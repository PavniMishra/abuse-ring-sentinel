"""
feature_engineering.py

Combines:
  - Graph features (component size, fraud-in-component, distance-to-fraud)
  - Account-level features (account age, transaction velocity, amount anomaly)
into a single flat table ready for model training.
"""

import sys
import os
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graph"))
from build_graph import build_graph, load_accounts  # noqa: E402
from ring_detection import run_all  # noqa: E402


def load_config(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def account_age_days(created_at: str) -> float:
    created = datetime.fromisoformat(created_at)
    return (datetime.now() - created).total_seconds() / 86400.0


def compute_account_level_features(accounts_df: pd.DataFrame, transactions_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    velocity_window_hours = cfg["features"]["velocity_window_hours"]

    txn_counts = transactions_df.groupby("account_id").size().rename("txn_count")
    txn_amount_mean = transactions_df.groupby("account_id")["amount"].mean().rename("amount_mean")
    txn_amount_std = transactions_df.groupby("account_id")["amount"].std().fillna(0).rename("amount_std")

    accounts_df = accounts_df.copy()
    accounts_df["account_age_days"] = accounts_df["created_at"].apply(account_age_days)
    accounts_df["is_new_account"] = (
        accounts_df["account_age_days"] < cfg["features"]["new_account_days_threshold"]
    ).astype(int)

    accounts_df = accounts_df.merge(txn_counts, on="account_id", how="left")
    accounts_df = accounts_df.merge(txn_amount_mean, on="account_id", how="left")
    accounts_df = accounts_df.merge(txn_amount_std, on="account_id", how="left")
    accounts_df["txn_count"] = accounts_df["txn_count"].fillna(0)
    accounts_df["amount_mean"] = accounts_df["amount_mean"].fillna(0)
    accounts_df["amount_std"] = accounts_df["amount_std"].fillna(0)

    # crude velocity proxy: txns per day since account creation
    accounts_df["velocity_per_day"] = accounts_df["txn_count"] / accounts_df["account_age_days"].clip(lower=1)

    return accounts_df


def build_feature_table(cfg: dict) -> pd.DataFrame:
    accounts_df = load_accounts(cfg)
    transactions_df = pd.read_csv(cfg["paths"]["transactions_csv"])

    G = build_graph(accounts_df, cfg)
    graph_features = run_all(G, max_distance=cfg["graph"]["max_dijkstra_distance"])
    graph_df = pd.DataFrame.from_dict(graph_features, orient="index").reset_index()
    graph_df = graph_df.rename(columns={"index": "account_id"})

    account_features_df = compute_account_level_features(accounts_df, transactions_df, cfg)

    merged = account_features_df.merge(graph_df, on="account_id", how="left")

    # clean up: cap infinite distance to a large finite number for the model
    max_dist_cap = cfg["graph"]["max_dijkstra_distance"] * 10
    merged["distance_to_fraud"] = merged["distance_to_fraud"].replace(
        [float("inf")], max_dist_cap
    )
    merged["component_size"] = merged["component_size"].fillna(1)
    merged["fraud_in_component"] = merged["fraud_in_component"].fillna(0)

    return merged


if __name__ == "__main__":
    cfg = load_config()
    features_df = build_feature_table(cfg)
    features_df.to_csv(cfg["paths"]["features_csv"], index=False)
    print(f"Wrote {len(features_df)} rows to {cfg['paths']['features_csv']}")
    print(features_df.columns.tolist())
