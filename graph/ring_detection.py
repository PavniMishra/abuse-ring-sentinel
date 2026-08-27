"""
ring_detection.py

v1 — Connected components: groups accounts into clusters via any shared
     attribute. Cheap, catches tight/obvious rings immediately.

v2 — Dijkstra distance-to-nearest-fraud: for accounts NOT in an obvious
     tight cluster, computes shortest weighted-path distance to the
     nearest known-fraud node. Edge weight = link strength, so we search
     over inverse-weight as "distance" (stronger link = shorter distance).
     This is what catches the deliberately "distributed" fraud accounts
     that connected-components alone misses (see build failure-recovery
     note in README).
"""

import networkx as nx


def connected_components_v1(G: nx.Graph) -> dict:
    """Returns {account_id: component_id, component_size, fraud_in_component}.

    IMPORTANT: fraud_in_component counts fraud OTHER than the node itself.
    Summing over the whole component (including the node) leaks the very
    label the model is trying to predict straight into a feature — for an
    isolated fraud node it would literally read "1 fraud in this component"
    (itself), and for an isolated legit node "0", trivially solving the
    classification problem without any real graph reasoning. Excluding
    self is what makes this a legitimate feature.
    """
    result = {}
    for component_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            others = [n for n in component if n != node]
            result[node] = {
                "component_id": component_id,
                "component_size": len(component),
                "fraud_in_component": sum(
                    G.nodes[n].get("is_fraud", 0) for n in others
                ),
            }
    return result


def _distance_graph(G: nx.Graph) -> nx.Graph:
    """Returns a copy where edge weight = 1 / strength, so nx.dijkstra
    treats strongly-linked accounts as 'close' and weakly-linked ones
    as 'far'."""
    H = G.copy()
    for u, v, data in H.edges(data=True):
        strength = max(data.get("weight", 0.01), 0.01)
        H[u][v]["distance"] = 1.0 / strength
    return H


def distance_to_fraud_v2(G: nx.Graph, max_distance: int = 5) -> dict:
    """For every node, shortest weighted distance to the nearest OTHER
    known-fraud node.

    IMPORTANT: a fraud node's distance to "nearest fraud" must exclude
    itself. Using every fraud node (including the one being scored) as a
    source means every fraud account trivially gets distance 0 to
    itself — that's the label leaking into the feature again, just via
    a different mechanism than fraud_in_component. For legit nodes this
    isn't an issue (they're never sources), so we only need the
    leave-one-out treatment for fraud nodes specifically.
    """
    H = _distance_graph(G)
    fraud_nodes = [n for n, d in H.nodes(data=True) if d.get("is_fraud") == 1]
    fraud_set = set(fraud_nodes)

    if not fraud_nodes:
        return {n: {"distance_to_fraud": float("inf"), "reachable_within_cutoff": False} for n in H.nodes}

    # Legit nodes: safe to use the full fraud source set in one pass.
    lengths = nx.multi_source_dijkstra_path_length(
        H, sources=fraud_nodes, weight="distance", cutoff=max_distance
    )

    # Fraud nodes: leave-one-out. For each fraud node f, only the OTHER
    # fraud nodes count as sources.
    fraud_lengths = {}
    for f in fraud_nodes:
        other_sources = [n for n in fraud_nodes if n != f]
        if not other_sources:
            fraud_lengths[f] = float("inf")
            continue
        loo_lengths = nx.multi_source_dijkstra_path_length(
            H, sources=other_sources, weight="distance", cutoff=max_distance
        )
        fraud_lengths[f] = loo_lengths.get(f, float("inf"))

    result = {}
    for node in H.nodes:
        if node in fraud_set:
            dist = fraud_lengths.get(node, float("inf"))
        else:
            dist = lengths.get(node, float("inf"))
        result[node] = {
            "distance_to_fraud": dist,
            "reachable_within_cutoff": dist < float("inf"),
        }
    return result


def run_all(G_strong: nx.Graph, G_full: nx.Graph, max_distance: int = 5) -> dict:
    """Combines v1 + v2 into one feature dict keyed by account_id.

    v1 (connected components) runs on G_strong — strong signals only
    (bank fingerprint / device), so it reflects what a simple clustering
    pass would actually catch: the obvious, tightly-linked rings.

    v2 (distance-to-fraud) runs on G_full — including weak IP edges —
    so it can reach the distributed fraud accounts that v1 cannot see
    at all, which is the entire point of adding it.
    """
    cc = connected_components_v1(G_strong)
    dist = distance_to_fraud_v2(G_full, max_distance=max_distance)

    all_nodes = set(G_strong.nodes) | set(G_full.nodes)
    combined = {}
    for node in all_nodes:
        node_cc = cc.get(node, {"component_id": -1, "component_size": 1, "fraud_in_component": 0})
        combined[node] = {**node_cc, **dist.get(node, {})}
    return combined


if __name__ == "__main__":
    import os
    import yaml
    from build_graph import build_graph, load_accounts

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    accounts_df = load_accounts(cfg)
    G_strong = build_graph(accounts_df, cfg, min_weight=0.5)
    G_full = build_graph(accounts_df, cfg, min_weight=0.0)
    results = run_all(G_strong, G_full, max_distance=cfg["graph"]["max_dijkstra_distance"])

    n_large_components = sum(1 for v in results.values() if v["component_size"] > 1)
    print(f"Accounts in a multi-node component (strong signals only): {n_large_components}")
    n_near_fraud = sum(1 for v in results.values() if v["distance_to_fraud"] < float("inf"))
    print(f"Accounts within distance cutoff of a fraud node (incl. weak IP links): {n_near_fraud}")