from __future__ import annotations

import random
import math
from typing import Tuple, Optional, Dict, cast
import networkx as nx
from networkx.classes.reportviews import DegreeView


def enforce_min_distance_3d(
    pos: Dict[int, Tuple[float, float, float]],
    min_dist: float = 0.15,
    iterations: int = 50,
    shrink_factor: float = 0.9
) -> Dict[int, Tuple[float, float, float]]:
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


def random_graph(
    n: int,
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


def random_eulerian(
    n: int,
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
    Generate a random undirected simple graph (nx.Graph) and make it Eulerian
    by adding new edges (no multiedges). Returns an nx.Graph where every
    vertex has even degree.

    NOTE: In pathological cases (e.g. a complete graph where no new edges
    exist to add) it may be impossible to make the graph Eulerian without
    allowing parallel edges; in that case the function raises RuntimeError
    and suggests using a MultiGraph-based generator.

    Parameters same semantics as your spec.
    """
    rng = random.Random(seed)

    # 1) build base simple graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.uniform(*weight_range) if weighted else 1.0
                G.add_edge(i, j, weight=w)

    # 2) ensure connectivity by linking components (if requested)
    if ensure_connected and n > 0:
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            comp_nodes = [list(c) for c in comps]
            reps = [rng.choice(nodes) for nodes in comp_nodes]
            for a, b in zip(reps, reps[1:]):
                if not G.has_edge(a, b):
                    w = rng.uniform(*weight_range) if weighted else 1.0
                    G.add_edge(a, b, weight=w)

    # 3) optionally assign 3D positions (kept minimal here; you can adapt from previous code)
    pos: Dict[int, Tuple[float, float, float]] = {}
    if pos_layout == "sphere":
        for idx, node in enumerate(G.nodes()):
            theta = 2 * math.pi * (idx / max(1, n))
            phi = math.acos(2 * rng.random() - 1)
            x = compact_scale * math.sin(phi) * math.cos(theta)
            y = compact_scale * math.sin(phi) * math.sin(theta)
            z = compact_scale * math.cos(phi)
            pos[node] = (x, y, z)
    else:  # random / cube
        for node in G.nodes():
            pos[node] = (
                rng.uniform(-compact_scale, compact_scale),
                rng.uniform(-compact_scale, compact_scale),
                rng.uniform(-compact_scale, compact_scale),
            )
    nx.set_node_attributes(G, {n: tuple(p) for n, p in pos.items()}, name="pos")

    # 4) make the graph Eulerian (all degrees even) by adding edges (no multiedges)
    odd = [v for v, d in cast(DegreeView, G.degree()) if d % 2 == 1]
    if not odd:
        return G  # already Eulerian

    # compute shortest-path distances between odd vertices (for matching)
    # (we'll use their distances as matching weights, but adding edges later is local)
    distances: Dict[int, Dict[int, float]] = {}
    for u in odd:
        lengths = nx.single_source_dijkstra_path_length(G, u, weight="weight")
        distances[u] = {v: lengths.get(v, float("inf")) for v in odd}

    # Build complete graph K on odd vertices with weights = shortest path distances
    K = nx.Graph()
    for i, u in enumerate(odd):
        for v in odd[i + 1:]:
            K.add_edge(u, v, weight=distances[u].get(v, float("inf")))

    from networkx.algorithms.matching import min_weight_matching

    # minimum-weight perfect matching on K
    mate = min_weight_matching(K, weight="weight")
    pairs = [(u, v) for u, v in mate]

    # helper: try to find a pivot w so that edges (u,w) and (w,v) are both absent
    def find_pivot_for_pair(u: int, v: int) -> Optional[int]:
        # prefer low-degree nodes for pivot to reduce collisions
        candidates = sorted(G.nodes(), key=lambda x: cast(int, G.degree(x)))
        for w in candidates:
            if w == u or w == v:
                continue
            if not G.has_edge(u, w) and not G.has_edge(w, v):
                return w
        return None

    # Now apply pairs: for each (u,v) try to add single edge u-v; if exists, try pivot 2-edge trick.
    for u, v in pairs:
        if not G.has_edge(u, v):
            # safe to add direct edge
            w = rng.uniform(*weight_range) if weighted else 1.0
            G.add_edge(u, v, weight=w)
            continue

        # direct edge exists; try to add two edges via pivot w so we don't create parallel edges
        pivot = find_pivot_for_pair(u, v)
        if pivot is not None:
            w1 = rng.uniform(*weight_range) if weighted else 1.0
            w2 = rng.uniform(*weight_range) if weighted else 1.0
            G.add_edge(u, pivot, weight=w1)
            G.add_edge(pivot, v, weight=w2)
            continue

        # If we reach here, couldn't add direct edge (exists) nor find a pivot.
        # This is rare but possible in dense graphs. In that case, making the graph Eulerian
        # using only *new* simple edges is not straightforward. We abort and suggest MultiGraph.
        raise RuntimeError(
            "Unable to produce a simple (non-multi) Eulerian augmentation for this graph.\n"
            "This can happen for dense/complete graphs where no new distinct edges exist to add.\n"
            "Consider using a MultiGraph-based generator (duplicate existing edges) instead."
        )

    # sanity check: all degrees even
    odd_after = [v for v, d in cast(DegreeView, G.degree()) if d % 2 == 1]
    if odd_after:
        # Shouldn't happen; raise if it does
        raise RuntimeError(f"Failed to make graph Eulerian; odd vertices remain: {odd_after}")

    return G
