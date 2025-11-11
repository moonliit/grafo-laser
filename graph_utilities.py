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


def enforce_min_distance_3d(pos: Dict[int, Tuple[float, float, float]],
                            min_dist: float = 0.15,
                            iterations: int = 50,
                            shrink_factor: float = 0.9) -> Dict[int, Tuple[float, float, float]]:
    """Iteratively push nodes apart in 3D if they are closer than min_dist."""
    nodes = list(pos.keys())
    coords = {u: [*pos[u]] for u in nodes}
    eps = 1e-6
    cur_min = min_dist
    for _ in range(iterations):
        moved = 0
        for i in range(len(nodes)):
            u = nodes[i]
            ux, uy, uz = coords[u]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                vx, vy, vz = coords[v]
                dx = ux - vx
                dy = uy - vy
                dz = uz - vz
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < eps:
                    angle = random.random() * 2.0 * math.pi
                    dx = math.cos(angle) * 1e-3
                    dy = math.sin(angle) * 1e-3
                    dz = math.cos(angle) * 1e-3
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < cur_min:
                    overlap = (cur_min - dist) / 2.0
                    nxorm = dx / dist
                    nyorm = dy / dist
                    nzorm = dz / dist
                    coords[u][0] += nxorm * overlap
                    coords[u][1] += nyorm * overlap
                    coords[u][2] += nzorm * overlap
                    coords[v][0] -= nxorm * overlap
                    coords[v][1] -= nyorm * overlap
                    coords[v][2] -= nzorm * overlap
                    moved += 1
        cur_min *= shrink_factor
        if moved == 0:
            break
    result = {u: tuple(coords[u]) for u in nodes}
    result = cast(Dict[int, Tuple[float, float, float]], result)
    return result


def random_graph_nx(n: int,
                    p: float = 0.2,
                    seed: Optional[int] = None,
                    weighted: bool = False,
                    weight_range: Tuple[float, float] = (1.0, 1.0),
                    ensure_connected: bool = True,
                    pos_layout: str = "random",  # 'random' | 'sphere' | 'cube'
                    compact_scale: float = 0.6,
                    min_node_dist: float = 0.12
                    ) -> nx.Graph:
    """
    Return a networkx.Graph with edges (possibly weighted) and node attribute 'pos'
    containing a 3D position tuple (x, y, z) for each node.

    Layout options:
      - 'random': uniform in a cube [-scale, scale]
      - 'sphere': evenly around a sphere surface
      - 'cube': same as random but less uniform
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add random edges
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.uniform(*weight_range) if weighted else 1.0
                G.add_edge(i, j, weight=w)

    # Ensure connectivity
    if ensure_connected and n > 0:
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            comp_nodes = [list(c) for c in comps]
            reps = [rng.choice(nodes) for nodes in comp_nodes]
            for a, b in zip(reps, reps[1:]):
                if not G.has_edge(a, b):
                    w = rng.uniform(*weight_range) if weighted else 1.0
                    G.add_edge(a, b, weight=w)

    # --- Position generation (3D) ---
    if pos_layout == "sphere":
        pos = {}
        for i, node in enumerate(G.nodes()):
            theta = 2 * math.pi * (i / n)
            phi = math.acos(2 * rng.random() - 1)
            x = compact_scale * math.sin(phi) * math.cos(theta)
            y = compact_scale * math.sin(phi) * math.sin(theta)
            z = compact_scale * math.cos(phi)
            pos[node] = (x, y, z)
    else:  # random / cube
        pos = {n: (rng.uniform(-compact_scale, compact_scale),
                   rng.uniform(-compact_scale, compact_scale),
                   rng.uniform(-compact_scale, compact_scale))
               for n in G.nodes()}

    if min_node_dist > 0:
        pos = enforce_min_distance_3d(pos, min_dist=min_node_dist, iterations=50, shrink_factor=0.92)

    nx.set_node_attributes(G, pos, name="pos")
    return G


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
