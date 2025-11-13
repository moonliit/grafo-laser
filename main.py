from __future__ import annotations

from graph_utilities.random_graph import random_graph, random_eulerian
from graph_utilities.mesh_to_graph import obj_to_graph
from chinese_postman_route import chinese_postman_route
from graph_utilities.plotting import animate_walk_simple


if __name__ == "__main__":
    # construct graph
    """G = random_graph(
        n=8,
        p=0.15,
        seed=None,
        weighted=True,
        weight_range=(0.5, 4.0),
        ensure_connected=True,
        pos_layout="sphere",
        compact_scale=0.6,
        min_node_dist=0.13
    )"""
    G = obj_to_graph("mesh/cube.obj")

    # get walk and plot it
    walk, total_weight = chinese_postman_route(G)
    anim = animate_walk_simple(
        G,
        walk,
        interval=300,
        arrow_every=2,
        cmap_name="plasma",
        repeat_offset_scale=0.015,
        repeat_linewidth_base=2.4,
        repeat_linewidth_step=0.9,
        growth_factor=0.45,
        fade=True,
        fade_steps=len(walk),
        fade_alpha_min=0.15,
        annotate_nodes_during_play=True,
        annotate_first_visit_only=False,
        projection="perspective"
    )
