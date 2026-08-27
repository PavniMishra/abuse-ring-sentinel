"""
build_graph.py

Builds a weighted, undirected graph of accounts.
Two accounts get an edge if they share device_id, ip, or bank_fp.
Edge weight = strength of the strongest shared attribute (from config.yaml).
Higher weight = stronger/rarer signal (a shared bank fingerprint matters
far more than a shared public IP).
"""

from collections import defaultdict

import networkx as nx
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_graph(accounts_df: pd.DataFrame, cfg: dict, min_weight: float = 0.0) -> nx.Graph:
    """Builds the account graph. min_weight filters out edges below that
    strength — pass min_weight=0.5 (roughly) to get a "strong signals only"
    graph (bank fingerprint / device), excluding weak coincidental IP
    overlap. Default (0.0) includes every edge."""
    weights = cfg["graph"]["edge_weights"]
    G = nx.Graph()

    for _, row in accounts_df.iterrows():
        G.add_node(
            row["account_id"],
            is_fraud=int(row["is_fraud"]),
            created_at=row["created_at"],
        )

    attribute_specs = [
        ("device_id", weights["shared_device_id"]),
        ("ip", weights["shared_ip"]),
        ("bank_fp", weights["shared_bank_fp"]),
    ]

    for attr, weight in attribute_specs:
        if weight < min_weight:
            continue  # skip this attribute type entirely if too weak
        buckets = defaultdict(list)
        for _, row in accounts_df.iterrows():
            buckets[row[attr]].append(row["account_id"])

        for attr_value, members in buckets.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if G.has_edge(a, b):
                        if weight > G[a][b]["weight"]:
                            G[a][b]["weight"] = weight
                            G[a][b]["shared_attr"] = attr
                    else:
                        G.add_edge(a, b, weight=weight, shared_attr=attr)

    return G


def load_accounts(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["accounts_csv"])


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    accounts_df = load_accounts(cfg)
    G = build_graph(accounts_df, cfg)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")