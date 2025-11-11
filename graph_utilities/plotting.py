"""random_graph_nx_positions_3d.py

- random_graph_nx(...) builds the graph *and* assigns 3D positions ('pos': (x, y, z)).
- plot_graph_nx(...) projects to 2D by ignoring z.
"""

from __future__ import annotations

import random
import math
from typing import Tuple, Optional, Any, Dict, cast

import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_nx(G: nx.Graph,
                  show_edge_weights: bool = False,
                  figsize: Tuple[int, int] = (6, 6)) -> None:
    """
    Plot graph using 3D positions stored in G.nodes[node]['pos'],
    projecting to 2D by ignoring z.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    pos3d = nx.get_node_attributes(G, "pos")
    if pos3d and len(pos3d) == G.number_of_nodes():
        pos = {k: (v[0], v[1]) for k, v in pos3d.items()}
    else:
        pos = nx.spring_layout(G, seed=42, iterations=200)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", edgecolors="k", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)

    weights = nx.get_edge_attributes(G, "weight")
    if weights:
        max_w = max(abs(w) for w in weights.values()) if weights else 1.0
        lw_list = [max(0.5, 2.0 * (weights.get((u, v), weights.get((v, u), 1.0)) / max_w))
                   for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=cast(Any, lw_list))
    else:
        nx.draw_networkx_edges(G, pos)

    if show_edge_weights and weights:
        edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.tight_layout()
    plt.show()
