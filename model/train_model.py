"""
train_model.py

Trains a classifier on the feature table. Held-out test set is split
BEFORE any evaluation happens (see evaluate.py) — no peeking.
"""

import os
import sys
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from feature_engineering import build_feature_table  # noqa: E402


FEATURE_COLUMNS = [
    "account_age_days",
    "is_new_account",
    "txn_count",
    "amount_mean",
    "amount_std",
    "velocity_per_day",
    "component_size",
    "fraud_in_component",
    "distance_to_fraud",
]
LABEL_COLUMN = "is_fraud"


def load_config(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(cfg):
    m_cfg = cfg["model"]
    if m_cfg["type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=m_cfg["n_estimators"],
            max_depth=m_cfg["max_depth"],
            random_state=m_cfg["random_seed"],
        )
    return RandomForestClassifier(
        n_estimators=m_cfg["n_estimators"],
        max_depth=m_cfg["max_depth"],
        random_state=m_cfg["random_seed"],
        class_weight="balanced",  # fraud is rare — don't let the model ignore it
    )


def train(cfg):
    features_df = build_feature_table(cfg)
    features_df.to_csv(cfg["paths"]["features_csv"], index=False)

    X = features_df[FEATURE_COLUMNS]
    y = features_df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test, acc_train, acc_test = train_test_split(
        X, y, features_df["account_id"],
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_seed"],
        stratify=y,
    )

    model = get_model(cfg)
    model.fit(X_train, y_train)

    with open(cfg["paths"]["model_path"], "wb") as f:
        pickle.dump(model, f)

    # Persist the held-out test split so evaluate.py never touches train data.
    test_df = X_test.copy()
    test_df["is_fraud"] = y_test
    test_df["account_id"] = acc_test
    test_path = cfg["paths"]["model_path"].replace("model.pkl", "held_out_test.csv")
    test_df.to_csv(test_path, index=False)

    print(f"Trained {cfg['model']['type']} on {len(X_train)} rows.")
    print(f"Held-out test set: {len(X_test)} rows -> {test_path}")
    print(f"Model saved to {cfg['paths']['model_path']}")

    return model, test_df


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
