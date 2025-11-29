from __future__ import annotations

import networkx as nx
from networkx.algorithms.matching import min_weight_matching
from networkx.classes.reportviews import DegreeView
from typing import cast, Tuple, Dict, List, Iterable


Pair = Tuple[int, int]
Pairing = List[Pair]
PathWithLen = Tuple[List[int], float]


def _edge_weight_safe(G: nx.Graph, a: int, b: int, default: float = 1.0) -> float:
    """Return a representative weight for edge (a,b) in G (handles Graph or MultiGraph)."""
    data = G.get_edge_data(a, b)
    if data is None:
        return default
    if G.is_multigraph():
        # data is a dict mapping keys -> attr-dicts
        try:
            # prefer the smallest explicit weight among parallel edges
            weights = []
            for key, attrs in data.items():
                if isinstance(attrs, dict):
                    weights.append(float(attrs.get("weight", default)))
                else:
                    # unexpected shape, skip
                    continue
            return min(weights) if weights else default
        except Exception:
            return default
    else:
        # data is an attr-dict
        try:
            return float(data.get("weight", default))
        except Exception:
            return default


def get_eulerian_circuit(G: nx.Graph) -> PathWithLen:
    """
    Return an Eulerian circuit (vertex list, closed) and its total weight.
    Works for Graph and MultiGraph.
    """
    total_weight = 0.0
    circuit: List[int] = []
    first_edge = True

    for edge in nx.eulerian_circuit(G):
        # networkx.eulerian_circuit yields (u,v) for Graph and (u,v,key) for MultiGraph
        if len(edge) == 3:
            u, v, k = edge
            # extract weight from multigraph specific storage
            try:
                w = float(G[u][v][k].get("weight", 1.0))
            except Exception:
                w = _edge_weight_safe(G, u, v, default=1.0)
        else:
            u, v = edge
            # for Graph G[u][v] is an attr-dict; for MultiGraph this can still be a mapping
            try:
                w = _edge_weight_safe(G, u, v, default=1.0)
            except Exception:
                w = 1.0

        if first_edge:
            circuit.append(u)
            first_edge = False
        circuit.append(v)
        total_weight += float(w)

    # ensure closed walk property
    if circuit and circuit[0] != circuit[-1]:
        circuit.append(circuit[0])

    return circuit, total_weight


def chinese_postman_route(G: nx.Graph) -> PathWithLen:
    assert nx.is_connected(G)

    # if graph is eulerian, optimal solution is an eulerian circuit
    if nx.is_eulerian(G):
        return get_eulerian_circuit(G)

    # 1. get odd vertices
    vertices_degrees = cast(DegreeView, G.degree())
    odd_vertices = [v for v, d in vertices_degrees if d % 2 != 0]
    k = len(odd_vertices)

    if k == 0:
        return get_eulerian_circuit(G)

    # 2. computer every pairwise dijkstra from odd vertices
    dist: Dict[int, Dict[int, float]] = {}   # dist[src][dst] = distance
    pred: Dict[int, Dict[int, List[int]]] = {}  # pred[src][dst] = path list

    for src in odd_vertices:
        lengths, paths = nx.single_source_dijkstra(G, src, weight="weight")
        # keep only required targets (odds)
        dist[src] = {dst: float(lengths[dst]) for dst in odd_vertices if dst in lengths and dst != src}
        pred[src] = {dst: paths[dst] for dst in odd_vertices if dst in paths and dst != src}

    # 3. build complete metric graph on odd vertices
    K = nx.Graph()
    K.add_nodes_from(odd_vertices)
    for i, u in enumerate(odd_vertices):
        for v in odd_vertices[i+1:]:
            d = dist[u].get(v, float("inf"))
            K.add_edge(u, v, weight=d)

    # 4. min-weight perfect matching (Blossom)
    matching = min_weight_matching(K, weight="weight")
    matching_pairs = list(matching)  # set of unordered pairs

    # 5. create multigraph from G
    MG = nx.MultiGraph()
    MG.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        MG.add_edge(u, v, **dict(data))

    # 6. duplicate edges along each matched shortest path
    for u, v in matching_pairs:
        path: List[int] = []
        # try fetching precomputed path
        if v in pred.get(u, {}):
            path = pred[u][v]
        elif u in pred.get(v, {}):
            path = pred[v][u]
        else:
            # fallback: compute on the fly
            path = nx.shortest_path(G, u, v, weight="weight")

        # duplicate each edge along the path
        if path and len(path) >= 2:
            for a, b in zip(path, path[1:]):
                w = _edge_weight_safe(G, a, b, default=1.0)
                MG.add_edge(a, b, weight=w)

    assert nx.is_eulerian(MG)
    return get_eulerian_circuit(MG)
