"""
visualize_ring.py

Renders the account graph as a static, high-resolution image for the
pitch deck: fraud nodes in red, legit nodes in gray, edge thickness
proportional to link strength (bank fingerprint > device > IP).

Only draws the CONNECTED subgraph (nodes with at least one edge).
With 800 accounts, the ~85% that are fully isolated (no shared
attribute with anyone) would just be visual noise — the story this
image needs to tell is about the clusters, not the empty space.
"""

import os
import sys
import math

import matplotlib
matplotlib.use("Agg")  # no display needed, just render to file
import matplotlib.pyplot as plt
import networkx as nx
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "graph"))
from build_graph import build_graph, load_accounts  # noqa: E402


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def draw_graph(G: nx.Graph, out_path: str, title: str = "Account Relationship Graph"):
    # Only the connected part of the graph carries a story.
    connected_nodes = [n for n in G.nodes if G.degree(n) > 0]
    H = G.subgraph(connected_nodes).copy()

    if H.number_of_nodes() == 0:
        print("No connected nodes to draw — graph is empty of edges.")
        return

    # Lay out each connected cluster in its OWN grid cell, rather than one
    # global spring layout. A single global layout pulls unrelated rings
    # into the same visual space via weak IP edges and produces an
    # unreadable "hairball" — separating clusters spatially is what
    # actually makes individual rings legible in a pitch slide.
    components = sorted(nx.connected_components(H), key=len, reverse=True)
    n_cols = math.ceil(math.sqrt(len(components)))
    cell_size = 3.0
    pos = {}
    for idx, component in enumerate(components):
        sub = H.subgraph(component)
        sub_pos = nx.spring_layout(sub, seed=42, k=0.9)
        row, col = divmod(idx, n_cols)
        offset_x, offset_y = col * cell_size, -row * cell_size
        # normalize each cluster to fit within its cell before offsetting
        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        x_range = max(xs) - min(xs) or 1
        y_range = max(ys) - min(ys) or 1
        for node, (x, y) in sub_pos.items():
            norm_x = (x - min(xs)) / x_range
            norm_y = (y - min(ys)) / y_range
            pos[node] = (offset_x + norm_x * cell_size * 0.8,
                         offset_y + norm_y * cell_size * 0.8)

    node_colors = [
        "#e05858" if H.nodes[n].get("is_fraud") == 1 else "#9aa5b1"
        for n in H.nodes
    ]
    node_sizes = [
        220 if H.nodes[n].get("is_fraud") == 1 else 120
        for n in H.nodes
    ]
    # Strong edges (bank/device) drawn solid; weak IP-only edges drawn
    # thin and faint so they read as context, not as the main structure.
    strong_edges = [(u, v) for u, v in H.edges if H[u][v].get("weight", 0) >= 0.5]
    weak_edges = [(u, v) for u, v in H.edges if H[u][v].get("weight", 0) < 0.5]

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#111111")
    ax.set_facecolor("#111111")

    nx.draw_networkx_edges(H, pos, edgelist=weak_edges, width=0.6,
                            edge_color="#444444", alpha=0.35, style="dashed", ax=ax)
    nx.draw_networkx_edges(H, pos, edgelist=strong_edges, width=2.2,
                            edge_color="#8a8a8a", alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes,
                            linewidths=0.5, edgecolors="#222222", ax=ax)

    ax.set_title(title, color="#f0e6d2", fontsize=18, pad=20)
    ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Known fraud',
               markerfacecolor='#e05858', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Legitimate',
               markerfacecolor='#9aa5b1', markersize=8),
        Line2D([0], [0], color='#8a8a8a', lw=2.2, label='Strong link (bank fp / device)'),
        Line2D([0], [0], color='#444444', lw=1, linestyle='dashed', label='Weak link (shared IP)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#1c1c1c',
              edgecolor='#333333', labelcolor='#f0e6d2', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

    n_fraud = sum(1 for n in H.nodes if H.nodes[n].get("is_fraud") == 1)
    n_legit = H.number_of_nodes() - n_fraud
    print(f"Saved {out_path}")
    print(f"Drew {H.number_of_nodes()} connected accounts ({n_fraud} fraud, {n_legit} legit), "
          f"{H.number_of_edges()} edges across {len(components)} clusters")


if __name__ == "__main__":
    cfg = load_config()
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    accounts_df = load_accounts(cfg)
    G = build_graph(accounts_df, cfg)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    draw_graph(G, os.path.join(out_dir, "account_graph.png"),
               title="Abuse-Ring Sentinel — Account Relationship Graph")