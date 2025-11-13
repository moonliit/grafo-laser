import networkx as nx
from typing import Set, Any

def _largest_connected_component_nodes(G: nx.Graph) -> Set[Any]:
    """
    Return the set of nodes in the largest connected component of G.
    If G is empty, returns an empty set.
    """
    if G.number_of_nodes() == 0:
        return set()
    # connected_components yields components as sets (largest first if reverse sorted)
    # but we can pick max by length:
    comps = nx.connected_components(G)
    largest = max(comps, key=len)  # returns a set of nodes
    return set(largest)


def connect_normalize(G: nx.Graph) -> nx.Graph:
    """
    Return a new NetworkX Graph that is the induced subgraph corresponding
    to the largest connected component of G. Node and edge attributes are preserved.
    """
    nodes = _largest_connected_component_nodes(G)
    if not nodes:
        return G.__class__()  # return empty graph of same type? returning Graph() is fine
    # Use G.subgraph(nodes).copy() to get an independent copy (not a view)
    return G.subgraph(nodes).copy()
