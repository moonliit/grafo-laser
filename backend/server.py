from typing import Dict, Any, List, Tuple, Optional
import math
import networkx as nx
from fastapi import Body, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chinese_postman_route import chinese_postman_route
import uvicorn
import math
from itertools import combinations

# Create FastAPI app
app = FastAPI(title="Draw→Polylines→Graph")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# iterative Douglas-Peucker (non-recursive)
def iterative_douglas_peucker(pts: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    n = len(pts)
    if n <= 2:
        return pts[:]  # nothing to do

    keep = [False] * n
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = pts[a]
        bx, by = pts[b]

        # find max distance
        maxd = -1.0
        maxi = -1
        dx = bx - ax
        dy = by - ay
        denom = dx*dx + dy*dy

        for i in range(a + 1, b):
            px, py = pts[i]
            if denom == 0.0:
                # a and b same point
                d = math.hypot(px - ax, py - ay)
            else:
                t = ((px - ax)*dx + (py - ay)*dy) / denom
                if t < 0.0:
                    projx, projy = ax, ay
                elif t > 1.0:
                    projx, projy = bx, by
                else:
                    projx = ax + t*dx
                    projy = ay + t*dy
                d = math.hypot(px - projx, py - projy)

            if d > maxd:
                maxd = d
                maxi = i

        if maxd > tol and maxi != -1:
            keep[maxi] = True
            stack.append((a, maxi))
            stack.append((maxi, b))

    # build result
    return [pts[i] for i, k in enumerate(keep) if k]


def _round_key(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    """
    Create a discretized key for approximate deduplication of coordinates.
    Adjust ndigits if you need coarser / finer merging.
    """
    return (round(x, ndigits), round(y, ndigits))


# --- Geometry helpers for intersection + splitting ---

EPS = 1e-9

def _on_segment(P: Tuple[float,float], Q: Tuple[float,float], R: Tuple[float,float]) -> bool:
    """Return True if Q lies on segment PR (bounding box test)."""
    return (min(P[0], R[0]) - EPS <= Q[0] <= max(P[0], R[0]) + EPS and
            min(P[1], R[1]) - EPS <= Q[1] <= max(P[1], R[1]) + EPS)

def segment_intersection(A: Tuple[float,float], B: Tuple[float,float],
                         C: Tuple[float,float], D: Tuple[float,float]) -> Optional[Dict[str, float]]:
    """
    Compute intersection between segment AB and CD.
    If intersects (including endpoints), return dict {x,y,t,u} where
    t is parameter on AB (0..1) and u on CD (0..1).
    Return None if no intersection.
    Handles proper intersections and endpoint touches; for colinear overlapping
    we return endpoint intersections when an endpoint lies on the other segment.
    """
    ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
    r_x = bx - ax; r_y = by - ay
    s_x = dx - cx; s_y = dy - cy

    denom = r_x * s_y - r_y * s_x
    if abs(denom) < EPS:
        # parallel or colinear
        # check colinearity by cross((C-A), r)
        cross = (cx - ax) * r_y - (cy - ay) * r_x
        if abs(cross) < EPS:
            # colinear: consider endpoints that lie on the other segment
            # return the first endpoint that lies on the other segment
            # (we only need points where segments touch)
            if _on_segment(A, C, B):
                t = ( (C[0]-ax)/r_x ) if abs(r_x) > abs(r_y) and abs(r_x) > EPS else ((C[1]-ay)/r_y if abs(r_y)>EPS else 0.0)
                return {"x": C[0], "y": C[1], "t": max(0.0, min(1.0, t)), "u": 0.0}
            if _on_segment(A, D, B):
                t = ( (D[0]-ax)/r_x ) if abs(r_x) > abs(r_y) and abs(r_x) > EPS else ((D[1]-ay)/r_y if abs(r_y)>EPS else 1.0)
                return {"x": D[0], "y": D[1], "t": max(0.0, min(1.0, t)), "u": 1.0}
            if _on_segment(C, A, D):
                u = ( (A[0]-cx)/s_x ) if abs(s_x) > abs(s_y) and abs(s_x) > EPS else ((A[1]-cy)/s_y if abs(s_y)>EPS else 0.0)
                return {"x": A[0], "y": A[1], "t": 0.0, "u": max(0.0, min(1.0, u))}
            if _on_segment(C, B, D):
                u = ( (B[0]-cx)/s_x ) if abs(s_x) > abs(s_y) and abs(s_x) > EPS else ((B[1]-cy)/s_y if abs(s_y)>EPS else 1.0)
                return {"x": B[0], "y": B[1], "t": 1.0, "u": max(0.0, min(1.0, u))}
            return None
        return None

    inv = 1.0 / denom
    cx_ax = cx - ax; cy_ay = cy - ay
    t = (cx_ax * s_y - cy_ay * s_x) * inv
    u = (cx_ax * r_y - cy_ay * r_x) * inv

    if t >= -EPS and t <= 1.0 + EPS and u >= -EPS and u <= 1.0 + EPS:
        ix = ax + t * r_x
        iy = ay + t * r_y
        return {"x": ix, "y": iy, "t": max(0.0, min(1.0, t)), "u": max(0.0, min(1.0, u))}
    return None

def split_segment_by_parameters(A: Tuple[float,float], B: Tuple[float,float], t_params: List[float]) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    """Given endpoints A,B and a list of t params in [0,1], return subsegments between consecutive sorted unique t's."""
    ts = sorted(set(max(0.0, min(1.0, float(t))) for t in t_params))
    # ensure endpoints present
    if len(ts) == 0 or abs(ts[0]) > EPS:
        ts.insert(0, 0.0)
    if abs(ts[-1] - 1.0) > EPS:
        ts.append(1.0)
    out = []
    for i in range(len(ts)-1):
        t0, t1 = ts[i], ts[i+1]
        if t1 - t0 < 1e-9:
            continue
        ax, ay = A; bx, by = B
        a_pt = (ax + (bx-ax)*t0, ay + (by-ay)*t0)
        b_pt = (ax + (bx-ax)*t1, ay + (by-ay)*t1)
        if math.hypot(b_pt[0]-a_pt[0], b_pt[1]-a_pt[1]) < 1e-9:
            continue
        out.append((a_pt, b_pt))
    return out

# Build segments from strokes (stroke is list of (x,y))
def build_segment_list(strokes: List[List[Tuple[float,float]]]) -> List[Dict[str,Any]]:
    segs = []
    for si, s in enumerate(strokes):
        for i in range(len(s)-1):
            A = s[i]; B = s[i+1]
            # skip degenerate
            if math.hypot(B[0]-A[0], B[1]-A[1]) < 1e-9:
                continue
            segs.append({"a": A, "b": B, "stroke_index": si, "seg_index": i})
    return segs

def split_all_segments_at_intersections(segs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Given a list of segments (each with .a and .b), compute intersection parameters for each segment,
    then split each into subsegments. Return list of subsegments as dicts {'a':(x,y),'b':(x,y), 'stroke_index', 'seg_index'}.
    """
    n = len(segs)
    if n == 0:
        return []

    # prepare t lists
    t_lists: List[List[float]] = [ [0.0, 1.0] for _ in range(n) ]

    # pairwise intersection (O(n^2))
    for i in range(n):
        Ai = segs[i]["a"]; Bi = segs[i]["b"]
        for j in range(i+1, n):
            Aj = segs[j]["a"]; Bj = segs[j]["b"]
            I = segment_intersection(Ai, Bi, Aj, Bj)
            if I:
                ti = float(I["t"])
                uj = float(I["u"])
                # add only if not near-duplicate
                if not any(abs(ti - v) < 1e-9 for v in t_lists[i]):
                    t_lists[i].append(ti)
                if not any(abs(uj - v) < 1e-9 for v in t_lists[j]):
                    t_lists[j].append(uj)

    # now split each segment
    out = []
    for idx, seg in enumerate(segs):
        A = seg["a"]; B = seg["b"]
        ts = t_lists[idx]
        sub = split_segment_by_parameters(A, B, ts)
        for a_pt, b_pt in sub:
            out.append({
                "a": a_pt,
                "b": b_pt,
                "stroke_index": seg["stroke_index"],
                "original_seg_index": seg["seg_index"]
            })
    return out

# Build graph from subsegments (list of dicts with 'a' and 'b' tuples)
def build_graph_from_subsegments(subsegs: List[Dict[str,Any]], ndigits: int = 3) -> Tuple[nx.Graph, Dict[int, Tuple[float,float]]]:
    G = nx.Graph()
    coord_to_node: Dict[Tuple[float,float], int] = {}
    node_pos: Dict[int, Tuple[float,float]] = {}
    next_id = 0

    def _get_node_id(pt: Tuple[float,float]) -> int:
        nonlocal next_id
        key = _round_key(pt[0], pt[1], ndigits)
        nid = coord_to_node.get(key)
        if nid is None:
            nid = next_id
            next_id += 1
            coord_to_node[key] = nid
            node_pos[nid] = (float(pt[0]), float(pt[1]))
            G.add_node(nid, pos=(float(pt[0]), float(pt[1])))
        return nid

    for s in subsegs:
        a = s["a"]; b = s["b"]
        u = _get_node_id(a)
        v = _get_node_id(b)
        if u == v:
            continue
        w = math.hypot(b[0]-a[0], b[1]-a[1])
        if G.has_edge(u, v):
            # keep smallest weight
            existing = G[u][v].get("weight", float('inf'))
            if w < existing:
                G[u][v]["weight"] = float(w)
        else:
            G.add_edge(u, v, weight=float(w))

    return G, node_pos

# Place this somewhere near the top of server.py (after iterative_douglas_peucker and _round_key)
from typing import Dict, Any, List, Tuple, Optional
EPS_INTER = 1e-9

def _seg_intersect(A: Tuple[float,float], B: Tuple[float,float],
                   C: Tuple[float,float], D: Tuple[float,float]) -> Optional[Tuple[float,float,float,float]]:
    """
    Robust segment intersection for AB and CD (2D).
    Returns (ix, iy, t, u) where ix,iy are intersection coordinates,
    t is param on AB (0..1), u is param on CD (0..1), or None if no intersection.
    Handles proper intersections and endpoint-touching; for colinear overlapping it will return
    endpoint intersections only (we treat overlapping specially by checking endpoints).
    """
    ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
    r_x = bx - ax; r_y = by - ay
    s_x = dx - cx; s_y = dy - cy
    denom = r_x * s_y - r_y * s_x
    if abs(denom) < EPS_INTER:
        # parallel or colinear
        # check colinear by cross((C-A), r) == 0
        cross = (cx - ax) * r_y - (cy - ay) * r_x
        if abs(cross) < EPS_INTER:
            # colinear: we won't attempt to produce intersections for overlapping interiors,
            # but endpoints lying on other segment are intersections; return the first matching one.
            def on_seg(P, Q, R):
                # Q on PR bounding box
                return (min(P[0], R[0]) - EPS_INTER <= Q[0] <= max(P[0], R[0]) + EPS_INTER and
                        min(P[1], R[1]) - EPS_INTER <= Q[1] <= max(P[1], R[1]) + EPS_INTER)
            # test endpoints
            if on_seg(A, C, B):
                # C lies on AB
                # compute t for C on AB
                denom_r = (r_x*r_x + r_y*r_y) or 1.0
                t = ((C[0]-ax)*r_x + (C[1]-ay)*r_y) / denom_r
                return (C[0], C[1], max(0.0, min(1.0, t)), 0.0)
            if on_seg(A, D, B):
                denom_r = (r_x*r_x + r_y*r_y) or 1.0
                t = ((D[0]-ax)*r_x + (D[1]-ay)*r_y) / denom_r
                return (D[0], D[1], max(0.0, min(1.0, t)), 1.0)
            if on_seg(C, A, D):
                denom_s = (s_x*s_x + s_y*s_y) or 1.0
                u = ((A[0]-cx)*s_x + (A[1]-cy)*s_y) / denom_s
                return (A[0], A[1], 0.0, max(0.0, min(1.0, u)))
            if on_seg(C, B, D):
                denom_s = (s_x*s_x + s_y*s_y) or 1.0
                u = ((B[0]-cx)*s_x + (B[1]-cy)*s_y) / denom_s
                return (B[0], B[1], 1.0, max(0.0, min(1.0, u)))
        return None

    inv = 1.0 / denom
    cx_ax = cx - ax; cy_ay = cy - ay
    t = (cx_ax * s_y - cy_ay * s_x) * inv
    u = (cx_ax * r_y - cy_ay * r_x) * inv
    if t >= -EPS_INTER and t <= 1.0 + EPS_INTER and u >= -EPS_INTER and u <= 1.0 + EPS_INTER:
        ix = ax + t * r_x
        iy = ay + t * r_y
        return (ix, iy, max(0.0, min(1.0, t)), max(0.0, min(1.0, u)))
    return None

def keep_largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) == 1:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

def _euclidean(p: Tuple[float,float], q: Tuple[float,float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])

def connect_components(G: nx.Graph) -> nx.Graph:
    # Work on a copy so we don’t mutate input
    H = G.copy()

    components = [list(c) for c in nx.connected_components(H)]
    k = len(components)

    # Already connected → nothing to do
    if k <= 1:
        return H

    # Build meta-graph of components
    CG = nx.Graph()
    CG.add_nodes_from(range(k))  # component indices

    # For each pair of components, compute closest pair of nodes
    for i, j in combinations(range(k), 2):
        best_dist = float('inf')
        best_pair = None

        for u in components[i]:
            pu = H.nodes[u].get('pos')
            if pu is None:
                raise ValueError(f"Node {u} has no 'pos' attribute")

            for v in components[j]:
                pv = H.nodes[v].get('pos')
                if pv is None:
                    raise ValueError(f"Node {v} has no 'pos' attribute")

                d = _euclidean(pu, pv)
                if d < best_dist:
                    best_dist = d
                    best_pair = (u, v)

        # Add the shortest possible "bridge" edge between these two components
        CG.add_edge(i, j, weight=best_dist, pair=best_pair)

    # Now compute MST on the component-graph
    mst = nx.minimum_spanning_tree(CG)

    # Add those edges to H
    for i, j in mst.edges():
        u, v = CG[i][j]['pair']
        H.add_edge(u, v)

    return H

def create_graph_helper(payload: Dict[str, Any]) -> nx.Graph:
    """
    Build a NetworkX Graph from payload containing strokes.
    Payload keys:
      "strokes": list of strokes, each stroke is list of {x,y}
      "tol": DP tolerance (optional)
    Returns:
      NetworkX Graph with nodes having attribute 'pos'=(x,y,0.0) and edges having 'weight' (float)
    """
    strokes_raw = payload.get("strokes")
    if strokes_raw is None or not isinstance(strokes_raw, list):
        raise ValueError("missing or invalid 'strokes'")

    tol = float(payload.get("tol", 1.0))

    # Step A: validate & parse strokes as list of list of (x,y)
    parsed_strokes: List[List[Tuple[float, float]]] = []
    for si, s in enumerate(strokes_raw):
        if not isinstance(s, list):
            continue
        pts: List[Tuple[float, float]] = []
        for p in s:
            if not isinstance(p, dict): continue
            try:
                x = float(p.get("x", 0.0)); y = float(p.get("y", 0.0))
            except Exception:
                continue
            pts.append((x, y))
        if len(pts) >= 1:
            parsed_strokes.append(pts)

    if len(parsed_strokes) == 0:
        raise ValueError("no valid strokes found")

    # Step B: simplify each stroke with iterative Douglas-Peucker (preserve endpoints)
    simplified: List[List[Tuple[float, float]]] = []
    for pts in parsed_strokes:
        if len(pts) <= 2:
            simplified.append(pts[:])
        else:
            simp = iterative_douglas_peucker(pts, tol)
            if len(simp) < 2 and len(pts) >= 2:
                simp = [pts[0], pts[-1]]
            simplified.append(simp)

    # Step C: build raw segments list (each segment references which stroke and index it came from)
    # segment entry: dict { 'a':(x,y), 'b':(x,y), 'stroke_index': int, 'seg_index': int }
    segments = []
    for si, s in enumerate(simplified):
        for i in range(len(s) - 1):
            a = s[i]; b = s[i+1]
            # skip degenerate
            if math.hypot(b[0]-a[0], b[1]-a[1]) < 1e-9:
                continue
            segments.append({'a': a, 'b': b, 'stroke_index': si, 'seg_index': i})

    # Step D: if split_flag -> compute intersection t-lists per segment and split
    subsegments = []
    # prepare t lists
    tlists = [ [0.0, 1.0] for _ in range(len(segments)) ]

    # pairwise intersection detection
    for i in range(len(segments)):
        Ai = segments[i]['a']; Bi = segments[i]['b']
        for j in range(i+1, len(segments)):
            Aj = segments[j]['a']; Bj = segments[j]['b']
            res = _seg_intersect(Ai, Bi, Aj, Bj)
            if res is None:
                continue
            ix, iy, ti, uj = res
            # add intersection parameters to both lists (avoid duplicates)
            # use approx equal check
            def push_unique(lst, val):
                for v in lst:
                    if abs(v - val) <= 1e-8:
                        return
                lst.append(val)
            push_unique(tlists[i], ti)
            push_unique(tlists[j], uj)

    # now split each segment using its tlist
    for idx, seg in enumerate(segments):
        ts = sorted(tlists[idx])
        a = seg['a']; b = seg['b']
        for k in range(len(ts) - 1):
            t0 = ts[k]; t1 = ts[k+1]
            if t1 - t0 < 1e-7: continue
            ax = a[0] + (b[0] - a[0]) * t0
            ay = a[1] + (b[1] - a[1]) * t0
            bx = a[0] + (b[0] - a[0]) * t1
            by = a[1] + (b[1] - a[1]) * t1
            if math.hypot(bx-ax, by-ay) < 1e-9: continue
            subsegments.append({'a': (ax, ay), 'b': (bx, by),
                                'original_stroke_index': seg['stroke_index'],
                                'original_seg_index': seg['seg_index']})

    # Step E: build graph from subsegments (deduplicate nodes using rounding)
    ndigits = 3  # tuneable: how aggressively to merge near-identical coordinates
    coord_to_nid = {}
    node_pos = {}
    G = nx.Graph()
    next_id = 0
    def _get_nid(x: float, y: float) -> int:
        nonlocal next_id
        key = (round(x, ndigits), round(y, ndigits))
        nid = coord_to_nid.get(key)
        if nid is None:
            nid = next_id
            next_id += 1
            coord_to_nid[key] = nid
            node_pos[nid] = (float(x), float(y))
            G.add_node(nid, pos=(float(x), float(y), 0.0))
        return nid

    for s in subsegments:
        (ax, ay) = s['a']; (bx, by) = s['b']
        u = _get_nid(ax, ay)
        v = _get_nid(bx, by)
        if u == v: continue
        w = math.hypot(bx-ax, by-ay)
        if G.has_edge(u, v):
            # keep smallest weight
            G[u][v]['weight'] = float(min(G[u][v].get('weight', float('inf')), w))
        else:
            G.add_edge(u, v, weight=float(w))

    # optionally keep only largest connected component (you were doing that previously)
    if G.number_of_nodes() == 0:
        raise ValueError("graph empty after building")

    # Return resulting graph
    return G

@app.post("/upload_polylines")
async def upload_polylines(payload: Dict[str, Any] = Body(...)):
    """
    POST /upload_polylines
    Body JSON:
      {
        "strokes": [ [ {"x":float,"y":float}, ... ], ... ],
        "tol": <float>   # optional; tolerance for Douglas-Peucker (default 1.0)
      }
    Returns:
      nodes: { id: [x,y] }, edges: [[u,v,weight], ...], walk: [node_id,...], positions: [{x,y},...], total_weight
    """
    G = create_graph_helper(payload)
    tol = float(payload.get("tol", 1.0))
    keep_largest = bool(payload.get("keep_largest", True)) # if keep only largest component

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="empty graph")

    if keep_largest:
        G = keep_largest_component(G)
    else:
        G = connect_components(G)

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="empty graph")

    # 3) Compute chinese postman route (your existing routine)
    try:
        walk, total = chinese_postman_route(G)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chinese_postman_route failed: {e}")

    # 4) Prepare JSON response
    nodes = { str(n): [ float(G.nodes[n]['pos'][0]), float(G.nodes[n]['pos'][1]) ] for n in G.nodes() }
    edges = [ [int(u), int(v), float(G[u][v].get('weight', 1.0))] for u, v in G.edges() ]

    positions = []
    try:
        for n in walk:
            pos = G.nodes[n]['pos']
            positions.append({ 'x': float(pos[0]), 'y': float(pos[1]) })
    except Exception:
        positions = []

    json_resp = {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk": [ int(x) for x in walk ],
        "positions": positions,
        "total_weight": float(total),
        "tol": float(tol),
    }
    return JSONResponse(json_resp)


if __name__ == "__main__":
    # Run without reloader to avoid multi-process side-effects.
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
