import math
import networkx as nx

def douglas_peucker(points, eps):
    """
    points: list of (x,y) or (x,y,z)
    eps: tolerance
    returns simplified list of points
    """

    if len(points) <= 2:
        return points[:]

    # Distance from point p to segment p0–p1
    def perp_dist(p, p0, p1):
        x, y = p[0], p[1]
        x1, y1 = p0[0], p0[1]
        x2, y2 = p1[0], p1[1]

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)

        # projection parameter
        t = ((x - x1)*dx + (y - y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        projx = x1 + t * dx
        projy = y1 + t * dy

        return math.hypot(x - projx, y - projy)

    # Find furthest point
    maxdist = -1
    index = -1
    p0, p1 = points[0], points[-1]

    for i in range(1, len(points)-1):
        d = perp_dist(points[i], p0, p1)
        if d > maxdist:
            maxdist = d
            index = i

    # If within tolerance → send endpoints only
    if maxdist <= eps:
        return [points[0], points[-1]]

    # Recurse
    left = douglas_peucker(points[:index+1], eps)
    right = douglas_peucker(points[index:], eps)
    return left[:-1] + right


def extract_linear_chains(G):
    """
    Extract maximal simple paths composed only of degree-2 nodes.
    Returns a list of ordered node lists.
    """
    chains = []
    visited = set()

    # Endpoint nodes (deg != 2) start chains
    endpoints = [n for n in G.nodes() if G.degree(n) != 2]

    for start in endpoints:
        for nbr in G.neighbors(start):
            if (start, nbr) in visited or (nbr, start) in visited:
                continue
            path = [start, nbr]
            visited.add((start, nbr))
            prev = start
            curr = nbr

            # walk through degree-2 corridor
            while G.degree(curr) == 2:
                nxts = [x for x in G.neighbors(curr) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if (curr, nxt) in visited:
                    break
                visited.add((curr, nxt))
                path.append(nxt)
                prev, curr = curr, nxt

            if len(path) > 1:
                chains.append(path)

    return chains


def extract_cycles(G):
    """
    Returns cycles as ordered node lists.
    NetworkX cycle_basis gives each cycle as an unordered ring.
    We sort them deterministically and rotate to smallest node.
    """
    cycles = []
    for cyc in nx.cycle_basis(G):
        if len(cyc) < 3:
            continue

        # make a deterministic cycle by rotating to smallest node-id
        cyc = cyc[:]  # copy
        min_node = min(cyc)
        idx = cyc.index(min_node)
        cyc = cyc[idx:] + cyc[:idx]

        # ensure it is cyclic (append first)
        cyc.append(cyc[0])
        cycles.append(cyc)
    return cycles


def simplify_polyline_node_list(G, node_list, eps):
    """Turns node IDs → point list, applies DP, returns simplified node ID list."""
    pts = [G.nodes[n]["pos"] for n in node_list]
    spts = douglas_peucker(pts, eps)

    # map positions back to closest nodes (safe for pixel graphs)
    # build reverse index
    pos_to_node = {tuple(G.nodes[n]["pos"]): n for n in G.nodes()}

    simplified_nodes = []
    for p in spts:
        n = pos_to_node.get((p[0], p[1], p[2]) if len(p) == 3 else (p[0], p[1]))
        if n is None:
            # fallback: brute nearest
            best = None
            bestd = 1e18
            for nid, dpos in pos_to_node.items():
                dx = dpos[0] - p[0]
                dy = dpos[1] - p[1]
                d = dx*dx + dy*dy
                if d < bestd:
                    bestd = d
                    best = nid
            n = best
        simplified_nodes.append(n)

    return simplified_nodes


def rebuild_graph_from_polylines(polylines, G):
    """Build a new graph from the simplified polylines."""
    NG = nx.Graph()
    # add all used nodes
    used = set()
    for line in polylines:
        used.update(line)

    for n in used:
        NG.add_node(n, pos=G.nodes[n]["pos"])

    # add edges according to polylines
    for line in polylines:
        for i in range(len(line)-1):
            u, v = line[i], line[i+1]
            if not NG.has_edge(u, v):
                w = float(math.hypot(
                    NG.nodes[u]["pos"][0] - NG.nodes[v]["pos"][0],
                    NG.nodes[u]["pos"][1] - NG.nodes[v]["pos"][1]
                ))
                NG.add_edge(u, v, weight=w)

    return NG


def optimize_graph(G: nx.Graph, tol=1.0, collapse_cycles=True):
    """
    Combined:
      - Simplify linear chains (degree-2 corridors)
      - Simplify cycles by breaking at smallest node and DP-ing ring
    """
    if G.number_of_nodes() <= 2:
        return G

    polylines = []

    # 1. Extract and simplify linear chains
    chains = extract_linear_chains(G)
    for ch in chains:
        if len(ch) < 3:
            polylines.append(ch)
        else:
            polylines.append(
                simplify_polyline_node_list(G, ch, tol)
            )

    # 2. Extract and simplify cycles
    if collapse_cycles:
        cycles = extract_cycles(G)
        for cyc in cycles:
            # cycle looks like [..., start, ... , start]
            scyc = simplify_polyline_node_list(G, cyc, tol)

            # ensure cycle closes
            if scyc[0] != scyc[-1]:
                scyc.append(scyc[0])

            polylines.append(scyc)

    # 3. Rebuild graph from all simplified polylines
    NG = rebuild_graph_from_polylines(polylines, G)
    return NG
