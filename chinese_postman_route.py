from __future__ import annotations

import networkx as nx
from typing import cast, Tuple, List
from networkx.classes.reportviews import DegreeView


def get_eulerian_circuit(G: nx.Graph) -> Tuple[List[int], float]:
    total_weight = 0.0
    circuit: List[int] = []

    first_edge = True
    for edge in nx.eulerian_circuit(G):
        u, v = edge
        w = G[u][v].get("weight", 1.0)

        # build vertex-list representation (closed walk)
        if first_edge:
            circuit.append(u)
            first_edge = False
        circuit.append(v)

        total_weight += float(w)

    # ensure closed walk property (first == last)
    if circuit and circuit[0] != circuit[-1]:
        circuit.append(circuit[0])

    return circuit, total_weight


def chinese_postman_route(G: nx.Graph) -> Tuple[List[int], float]:
    vertices_degrees = cast(DegreeView, G.degree())
    even_vertices = [v for v, d in vertices_degrees if d % 2 == 0]
    odd_vertices = [v for v, d in vertices_degrees if d % 2 != 0]

    # if graph is eulerian, optimal solution is an eulerian circuit
    if nx.is_eulerian(G):
        return get_eulerian_circuit(G)

    # TODO: finish this
    return get_eulerian_circuit(G) # placeholder
