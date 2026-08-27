"""
test_ring_detection.py

Sanity tests for the core graph logic. Run with: pytest tests/
"""

import os
import sys

import networkx as nx
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graph"))
from ring_detection import connected_components_v1, distance_to_fraud_v2, run_all  # noqa: E402


def make_test_graph():
    """3-node tight ring (all fraud) + 1 isolated legit node
    + 1 legit node weakly linked (2 hops) to the ring."""
    G = nx.Graph()
    G.add_node("f1", is_fraud=1)
    G.add_node("f2", is_fraud=1)
    G.add_node("f3", is_fraud=1)
    G.add_node("legit_isolated", is_fraud=0)
    G.add_node("legit_bridge", is_fraud=0)
    G.add_node("legit_far", is_fraud=0)

    G.add_edge("f1", "f2", weight=1.0)
    G.add_edge("f2", "f3", weight=1.0)
    G.add_edge("f3", "legit_bridge", weight=0.3)
    G.add_edge("legit_bridge", "legit_far", weight=0.3)
    return G


def test_connected_components_groups_the_ring_together():
    G = make_test_graph()
    result = connected_components_v1(G)
    assert result["f1"]["component_id"] == result["f2"]["component_id"] == result["f3"]["component_id"]
    assert result["legit_isolated"]["component_id"] != result["f1"]["component_id"]


def test_connected_components_fraud_count_correct():
    G = make_test_graph()
    result = connected_components_v1(G)
    # fraud_in_component EXCLUDES the node itself — for f1, that's f2 and
    # f3 only, i.e. 2, not 3.
    assert result["f1"]["fraud_in_component"] == 2


def test_isolated_node_has_component_size_one():
    G = make_test_graph()
    result = connected_components_v1(G)
    assert result["legit_isolated"]["component_size"] == 1


def test_distance_to_fraud_increases_with_hops():
    G = make_test_graph()
    result = distance_to_fraud_v2(G, max_distance=10)
    # f1's nearest OTHER fraud node is f2, one edge away — NOT itself (0).
    assert result["f1"]["distance_to_fraud"] > 0
    assert result["legit_bridge"]["distance_to_fraud"] < result["legit_far"]["distance_to_fraud"]


def test_isolated_node_unreachable_within_cutoff():
    G = make_test_graph()
    result = distance_to_fraud_v2(G, max_distance=10)
    assert result["legit_isolated"]["distance_to_fraud"] == float("inf")
    assert result["legit_isolated"]["reachable_within_cutoff"] is False


def test_run_all_merges_both_feature_sets():
    G = make_test_graph()
    combined = run_all(G, G, max_distance=10)
    assert "component_size" in combined["f1"]
    assert "distance_to_fraud" in combined["f1"]


def test_empty_graph_does_not_crash():
    G = nx.Graph()
    assert connected_components_v1(G) == {}
    assert distance_to_fraud_v2(G) == {}


def test_v1_misses_weak_only_link_but_v2_catches_it():
    """The core claim of the project: an account connected to fraud ONLY
    via a weak edge is invisible to v1 (strong-only graph) but reachable
    via v2's distance metric on the full graph."""
    G_full = nx.Graph()
    G_full.add_node("fraud1", is_fraud=1)
    G_full.add_node("weak_linked", is_fraud=1)
    G_full.add_edge("fraud1", "weak_linked", weight=0.3)

    G_strong = nx.Graph()
    G_strong.add_node("fraud1", is_fraud=1)
    G_strong.add_node("weak_linked", is_fraud=1)

    combined = run_all(G_strong, G_full, max_distance=10)
    assert combined["weak_linked"]["component_size"] == 1
    assert combined["weak_linked"]["distance_to_fraud"] < float("inf")


def test_no_label_leakage_via_self_inclusion():
    """Regression test for a real bug found in this project: both
    fraud_in_component and distance_to_fraud originally summed/searched
    over EVERY fraud node including the one being scored, so an isolated
    fraud node trivially got fraud_in_component=1 (itself) and
    distance_to_fraud=0 (itself) — the label leaking straight into the
    features."""
    G = nx.Graph()
    G.add_node("lone_fraud", is_fraud=1)
    G.add_node("lone_legit", is_fraud=0)

    cc = connected_components_v1(G)
    assert cc["lone_fraud"]["fraud_in_component"] == 0

    dist = distance_to_fraud_v2(G, max_distance=5)
    assert dist["lone_fraud"]["distance_to_fraud"] == float("inf")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))