from __future__ import annotations

from typing import cast
from networkx.classes.reportviews import DegreeView
from graph_utilities import random_graph_nx, plot_graph_nx


if __name__ == "__main__":
    # define seed for reproducibility
    seed = None

    # parameters for random graph generation
    n_vertices = 14
    p = 0.15
    weighted = True
    weight_range = (0.5, 4.0)
    ensure_connected = True
    pos_layout = "sphere"
    compact_scale = 0.6
    min_node_dist = 0.13

    # construct random graph
    G = random_graph_nx(
        n=n_vertices,
        p=p,
        seed=seed,
        weighted=weighted,
        weight_range=weight_range,
        ensure_connected=ensure_connected,
        pos_layout=pos_layout,
        compact_scale=compact_scale,
        min_node_dist=min_node_dist
    )
    print(f"Obtained G: n={G.number_of_nodes()} m={G.number_of_edges()}")
    print("Odd-degree vertices:", [v for v, d in cast(DegreeView, G.degree()) if d % 2 == 1])

    # plot graph
    plot_graph_nx(G, show_edge_weights=True)
