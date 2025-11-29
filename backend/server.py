# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
from io import BytesIO
import networkx as nx
from PIL import Image
import math
import os
import uuid
import pickle
import tempfile

app = FastAPI(title="Draw→PixelGraph→CPP (single-step)")

# CORS for development - adjust allowed origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# optional store for debugging
STORE_DIR = os.path.join(tempfile.gettempdir(), "draw_graph_store")
os.makedirs(STORE_DIR, exist_ok=True)

def _maybe_store_graph(G: nx.Graph) -> str:
    gid = str(uuid.uuid4())
    path = os.path.join(STORE_DIR, f"{gid}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(G, f)
        return gid
    except Exception:
        return ""

# ---------- image -> binary -> pixel graph ----------
def image_to_binary(img: Image.Image, threshold: int = 128) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p < threshold else 0, mode='L')
    return bw

def build_graph_from_binary(bw_img: Image.Image) -> nx.Graph:
    arr = bw_img.load()
    w, h = bw_img.size
    G = nx.Graph()
    id_of = {}
    node_id = 0
    for y in range(h):
        for x in range(w):
            v = arr[x,y]
            if v == 255:
                id_of[(x,y)] = node_id
                G.add_node(node_id, pos=(float(x), float(y), 0.0))
                node_id += 1
    neigh = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for (x,y), nid in list(id_of.items()):
        for dx,dy in neigh:
            nx_ = x + dx; ny_ = y + dy
            if (nx_, ny_) in id_of:
                nid2 = id_of[(nx_, ny_)]
                if not G.has_edge(nid, nid2):
                    wgt = math.hypot(dx, dy)
                    G.add_edge(nid, nid2, weight=float(wgt))
    return G

def keep_largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) == 1:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

# ---------- solver import order ----------
try:
    from cpp_optimized_module import cpp_optimized as solver
except Exception:
    try:
        from cpp_module import chinese_postman_route as solver
    except Exception:
        def solver(G: nx.Graph):
            if nx.is_eulerian(G):
                circuit = []
                total = 0.0
                first = True
                for e in nx.eulerian_circuit(G):
                    if len(e) == 3:
                        u,v,k = e
                    else:
                        u,v = e
                    if first:
                        circuit.append(u); first=False
                    circuit.append(v)
                    total += float(G[u][v].get("weight", 1.0))
                if circuit and circuit[0] != circuit[-1]:
                    circuit.append(circuit[0])
                return circuit, total
            return [], 0.0

# ---------- single-step endpoint ----------
@app.post("/upload_and_compute")
async def upload_and_compute(
    image: UploadFile = File(...),
    keep_graph: bool = False,
    threshold: int = 128
):
    """
    Accept uploaded pixelated image and return parsed graph + computed walk.
    Query params:
      - keep_graph: if true store the graph on the server and return graph_id
      - threshold: grayscale threshold (0..255), default 128
    """
    content = await image.read()
    try:
        pil = Image.open(BytesIO(content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    bw = image_to_binary(pil, threshold=int(threshold))
    G = build_graph_from_binary(bw)
    G = keep_largest_component(G)

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="no black pixels detected")

    graph_id = ""
    if bool(keep_graph):
        graph_id = _maybe_store_graph(G)

    try:
        from chinese_postman_route import chinese_postman_route
        walk, total = chinese_postman_route(G)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"solver failed: {e}")

    nodes = {str(n): [float(G.nodes[n]["pos"][0]), float(G.nodes[n]["pos"][1])] for n in G.nodes()}
    edges = [[int(u), int(v), float(G[u][v].get("weight", 1.0))] for u,v in G.edges()]

    resp = {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk": [int(x) for x in walk],
        "total_weight": float(total),
    }
    if graph_id:
        resp["graph_id"] = graph_id
    return JSONResponse(resp)


# POST /upload_and_compute_pixels
from fastapi import Body

@app.post("/upload_and_compute_pixels")
async def upload_and_compute_pixels(payload: Dict[str, Any] = Body(...), keep_graph: bool = False):
    """
    Accept raw pixels JSON:
      { "pixels": [[0|1,...], ...] }
    Returns nodes/edges and computed walk (single-step).
    """
    pixels = payload.get("pixels")
    if pixels is None:
        raise HTTPException(status_code=400, detail="missing 'pixels' in JSON body")

    # validate rectangular shape
    if not isinstance(pixels, list) or len(pixels) == 0 or not isinstance(pixels[0], list):
        raise HTTPException(status_code=400, detail="'pixels' must be 2D list")

    h = len(pixels)
    w = len(pixels[0])
    for row in pixels:
        if len(row) != w:
            raise HTTPException(status_code=400, detail="non-rectangular 'pixels' array")

    # build graph: each pixel with value truthy (1) becomes node with pos (x,y,0)
    G = nx.Graph()
    id_of = {}
    node_id = 0
    for r in range(h):
        for c in range(w):
            val = pixels[r][c]
            try:
                bit = 1 if int(val) else 0
            except Exception:
                bit = 1 if bool(val) else 0
            if bit:
                id_of[(c, r)] = node_id
                G.add_node(node_id, pos=(float(c), float(r), 0.0))
                node_id += 1

    # connect 8-neighbors
    neigh = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
    for (x,y), nid in list(id_of.items()):
        for dx,dy in neigh:
            nx_ = x + dx; ny_ = y + dy
            if (nx_, ny_) in id_of:
                nid2 = id_of[(nx_, ny_)]
                if not G.has_edge(nid, nid2):
                    wgt = math.hypot(dx, dy)
                    G.add_edge(nid, nid2, weight=float(wgt))

    # keep largest connected component
    G = keep_largest_component(G)

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="empty graph (no pixels set)")

    # optionally store graph
    graph_id = ""
    if keep_graph:
        graph_id = _maybe_store_graph(G)

    # compute walk
    try:
        from chinese_postman_route import chinese_postman_route
        walk, total = chinese_postman_route(G)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"solver failed: {e}")

    # prepare nodes/edges for JSON
    nodes = {str(n): [float(G.nodes[n]["pos"][0]), float(G.nodes[n]["pos"][1])] for n in G.nodes()}
    edges = [[int(u), int(v), float(G[u][v].get("weight", 1.0))] for u, v in G.edges()]

    resp = {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk": [int(x) for x in walk],
        "total_weight": float(total)
    }
    if graph_id:
        resp["graph_id"] = graph_id
    return JSONResponse(resp)
