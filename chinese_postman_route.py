from __future__ import annotations

import networkx as nx
from networkx.classes.reportviews import DegreeView
from typing import cast, Tuple, List, Iterable


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


def generate_pairings(nodes: List[int]) -> Iterable[Pairing]:
    if not nodes:
        yield []
        return

    first = nodes[0]
    # try pairing `first` with each other node
    for i in range(1, len(nodes)):
        second = nodes[i]
        rest = nodes[1:i] + nodes[i+1:]

        # recursively get pairings of the rest
        for rest_pairing in generate_pairings(rest):
            yield [(first, second)] + rest_pairing


def evaluate_pairings(
    G: nx.Graph,
    odd_vertices: List[int],
) -> List[Tuple[Pairing, float, List[PathWithLen]]]:
    if len(odd_vertices) == 0:
        return []

    nodes = list(odd_vertices)
    results = []
    for pairing in generate_pairings(nodes):
        total = 0.0
        paths_acc: List[PathWithLen] = []
        for (u, v) in pairing:
            path = nx.shortest_path(G, u, v, weight="weight")
            length = nx.shortest_path_length(G, u, v, weight="weight")
            total += length
            paths_acc.append((path, length))
        results.append((pairing, total, paths_acc))

    results.sort(key=lambda t: t[1])
    return results


def chinese_postman_route(G: nx.Graph) -> PathWithLen:
    # if graph is eulerian, optimal solution is an eulerian circuit
    if nx.is_eulerian(G):
        return get_eulerian_circuit(G)

    vertices_degrees = cast(DegreeView, G.degree())
    odd_vertices = [v for v, d in vertices_degrees if d % 2 != 0]

    if not odd_vertices:
        return get_eulerian_circuit(G)

    pairings = evaluate_pairings(G, odd_vertices)
    _, _, best_paths = pairings[0]

    # create multigraph from G
    MG = nx.MultiGraph()
    MG.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        MG.add_edge(u, v, **dict(data))

    # duplicate edges along each shortest path in best_paths
    for path, _ in best_paths:
        # path is a list of vertices [a, ..., b]
        if not path or len(path) < 2:
            continue
        for a, b in zip(path, path[1:]):
            w = _edge_weight_safe(G, a, b, default=1.0)
            MG.add_edge(a, b, weight=w)

    circuit, total_weight = get_eulerian_circuit(MG)
    return circuit, total_weight
