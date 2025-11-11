from __future__ import annotations

import random
import math
from typing import Tuple, Optional, Any, Dict
from typing import cast

import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.reportviews import DegreeView

def random_graph_nx(n: int,
                    p: float = 0.2,
                    seed: Optional[int] = None,
                    weighted: bool = False,
                    weight_range: Tuple[float, float] = (1.0, 1.0)) -> nx.Graph:
    """Return a networkx.Graph: undirected, simple, optionally weighted."""
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                w = random.uniform(weight_range[0], weight_range[1]) if weighted else 1.0
                G.add_edge(i, j, weight=w)
    return G


def enforce_min_distance(pos: Dict[int, Tuple[float, float]],
                         min_dist: float = 0.15,
                         iterations: int = 50,
                         shrink_factor: float = 0.9) -> Dict[int, Tuple[float, float]]:
    """
    Iteratively push nodes apart if they are closer than min_dist.
    - pos: dict node -> (x,y)
    - min_dist: desired minimum distance between any two nodes
    - iterations: number of relaxation iterations
    - shrink_factor: multiply min_dist by this each iteration to help convergence
    Returns modified pos (new dict).
    """
    nodes = list(pos.keys())
    # Convert to mutable floats
    coords = {u: [pos[u][0], pos[u][1]] for u in nodes}
    eps = 1e-6
    cur_min = min_dist
    for it in range(iterations):
        moved = 0
        for i in range(len(nodes)):
            u = nodes[i]
            ux, uy = coords[u]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                vx, vy = coords[v]
                dx = ux - vx
                dy = uy - vy
                dist = math.hypot(dx, dy)
                if dist < eps:
                    # jitter tiny random direction if identical positions
                    angle = random.random() * 2.0 * math.pi
                    dx = math.cos(angle) * 1e-3
                    dy = math.sin(angle) * 1e-3
                    dist = math.hypot(dx, dy)
                if dist < cur_min:
                    # How much to move: half of the overlap for each node
                    overlap = (cur_min - dist) / 2.0
                    # normalized vector
                    nxorm = dx / dist
                    nyorm = dy / dist
                    # move u away along +n, v along -n
                    coords[u][0] += nxorm * overlap
                    coords[u][1] += nyorm * overlap
                    coords[v][0] -= nxorm * overlap
                    coords[v][1] -= nyorm * overlap
                    moved += 1
        # ease the constraint slowly to avoid oscillation
        cur_min *= shrink_factor
        if moved == 0:
            break
    # convert back to tuples
    return {u: (coords[u][0], coords[u][1]) for u in nodes}


def plot_graph_nx(G: nx.Graph,
                  layout: str = "spring",
                  seed: Optional[int] = None,
                  show_edge_weights: bool = False,
                  compact_scale: float = 0.6,
                  compact_k: Optional[float] = 0.12,
                  min_node_dist: float = 0.12,
                  figsize: Tuple[int, int] = (6, 6)) -> None:
    """
    Draw the networkx graph with a compact node placement and enforced minimum distance.
    - layout: 'spring' | 'circle' | 'random'
    - min_node_dist: minimum distance between any pair of nodes (in layout coordinates)
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=compact_k, scale=compact_scale, iterations=200)
    elif layout == "random":
        rng = random.Random(seed)
        pos = {n: (rng.uniform(-compact_scale, compact_scale), rng.uniform(-compact_scale, compact_scale))
               for n in G.nodes()}
    else:  # circle
        pos = {}
        num_nodes = G.number_of_nodes()
        r = compact_scale
        for i, node in enumerate(G.nodes()):
            theta = 2.0 * math.pi * i / max(1, num_nodes)
            pos[node] = (r * 0.85 * math.cos(theta), r * 0.85 * math.sin(theta))

    # enforce a minimum pairwise distance so the whole group isn't collapsed
    if min_node_dist is not None and min_node_dist > 0.0:
        pos = enforce_min_distance(pos, min_dist=min_node_dist, iterations=50, shrink_factor=0.92)

    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", edgecolors="k", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # prepare edge widths mapped from weights (if exist)
    weights = nx.get_edge_attributes(G, "weight")
    if weights:
        max_w = max(abs(w) for w in weights.values()) if weights else 1.0
        # compute per-edge linewidths in same order as G.edges()
        lw_list = [max(0.5, 2.0 * (weights.get((u, v), weights.get((v, u), 1.0)) / max_w))
                   for u, v in G.edges()]
        # cast to Any so type-checkers won't complain about list->float mismatch
        nx.draw_networkx_edges(G, pos, width=cast(Any, lw_list))
    else:
        nx.draw_networkx_edges(G, pos)

    if show_edge_weights and weights:
        # create human-readable labels
        edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- configure here ---
    n_vertices = 18
    p = 0.18
    seed = None # Se permite definir un seed para reproducibilidad
    weighted = True
    weight_range = (0.5, 4.0)
    layout = "spring"         # 'spring' | 'circle' | 'random'
    show_edge_weights = True
    compact_scale = 0.6
    compact_k = 0.12
    min_node_dist = 0.12      # tune: increase to force more spacing
    # ------------------------

    G = random_graph_nx(n_vertices, p=p, seed=seed, weighted=weighted, weight_range=weight_range)
    print(f"Generated G: n={G.number_of_nodes()} m={G.number_of_edges()}")
    print("Odd-degree vertices:", [v for v, d in cast(DegreeView, G.degree()) if d % 2 == 1])
    plot_graph_nx(G, layout=layout, seed=seed, show_edge_weights=show_edge_weights,
                  compact_scale=compact_scale, compact_k=compact_k, min_node_dist=min_node_dist)
