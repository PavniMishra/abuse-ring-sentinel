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
    """Returns {account_id: component_id, component_size}."""
    result = {}
    for component_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            result[node] = {
                "component_id": component_id,
                "component_size": len(component),
                "fraud_in_component": sum(
                    G.nodes[n].get("is_fraud", 0) for n in component
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
    """For every node, shortest weighted distance to ANY known-fraud node.
    Uses multi-source Dijkstra (single pass, not one Dijkstra call per
    fraud node) for efficiency.
    """
    H = _distance_graph(G)
    fraud_nodes = [n for n, d in H.nodes(data=True) if d.get("is_fraud") == 1]

    if not fraud_nodes:
        return {n: {"distance_to_fraud": float("inf"), "hops_to_fraud": None} for n in H.nodes}

    # networkx multi_source_dijkstra_path_length handles this in one pass
    lengths = nx.multi_source_dijkstra_path_length(
        H, sources=fraud_nodes, weight="distance", cutoff=max_distance
    )

    result = {}
    for node in H.nodes:
        dist = lengths.get(node, float("inf"))
        result[node] = {
            "distance_to_fraud": dist,
            "reachable_within_cutoff": node in lengths,
        }
    return result


def run_all(G: nx.Graph, max_distance: int = 5) -> dict:
    """Combines v1 + v2 into one feature dict keyed by account_id."""
    cc = connected_components_v1(G)
    dist = distance_to_fraud_v2(G, max_distance=max_distance)

    combined = {}
    for node in G.nodes:
        combined[node] = {**cc.get(node, {}), **dist.get(node, {})}
    return combined


if __name__ == "__main__":
    import os
    import yaml
    from build_graph import build_graph, load_accounts

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))  # so relative csv paths resolve

    accounts_df = load_accounts(cfg)
    G = build_graph(accounts_df, cfg)
    results = run_all(G, max_distance=cfg["graph"]["max_dijkstra_distance"])

    n_large_components = sum(1 for v in results.values() if v["component_size"] > 1)
    print(f"Accounts in a multi-node component: {n_large_components}")
    n_near_fraud = sum(1 for v in results.values() if v["distance_to_fraud"] < float("inf"))
    print(f"Accounts within distance cutoff of a fraud node: {n_near_fraud}")
