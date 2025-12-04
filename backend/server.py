# server.py
from typing import Dict, Any
import math
import numpy as np
import networkx as nx

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from graph_utilities.optimize_graph import optimize_graph
from chinese_postman_route import chinese_postman_route

# Create FastAPI app
app = FastAPI(title="Draw→PixelGraph→CPP (single-step)")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

def keep_largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) == 1:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


# POST /upload_and_compute_pixels
@app.post("/upload_and_compute_pixels")
async def upload_and_compute_pixels(payload: Dict[str, Any] = Body(...)):
    """
    Accept raw pixels JSON:
      { "pixels": [[0|1,...], ...] }
    Returns nodes/edges and computed walk (single-step).
    """
    pixels = payload.get("pixels")
    tol = payload.get("tol")
    if pixels is None:
        raise HTTPException(status_code=400, detail="missing 'pixels' in JSON body")
    if tol is None:
        raise HTTPException(status_code=400, detail="missing 'tol' in JSON body")

    # validate rectangular shape
    if not isinstance(pixels, list) or len(pixels) == 0 or not isinstance(pixels[0], list):
        raise HTTPException(status_code=400, detail="'pixels' must be 2D list")

    h = len(pixels)
    w = len(pixels[0])
    for row in pixels:
        if len(row) != w:
            raise HTTPException(status_code=400, detail="non-rectangular 'pixels' array")

    # build graph: each truthy pixel becomes a node with pos (x,y,0)
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

    # optimize graph: collapse straight chains
    try:
        G = optimize_graph(G, tol=tol, collapse_cycles=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"graph optimization failed: {e}")

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="empty graph (no pixels set)")

    # keep largest connected component again (safety)
    G = keep_largest_component(G)

    if G.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="empty graph (no pixels set)")

    # compute walk (chinese postman) - expects graph with weights
    try:
        walk, total = chinese_postman_route(G)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chinese_postman_route failed: {e}")

    # prepare nodes/edges for JSON
    nodes = {str(n): [float(G.nodes[n]["pos"][0]), float(G.nodes[n]["pos"][1])] for n in G.nodes()}
    edges = [[int(u), int(v), float(G[u][v].get("weight", 1.0))] for u, v in G.edges()]

    # prepare positions for possible MQTT publishing use (walk may repeat start)
    try:
        positions = [{"x": float(G.nodes[n]["pos"][0]), "y": float(G.nodes[n]["pos"][1])} for n in walk]
        if positions:
            positions.pop()  # drop final duplicated return-to-start if present
    except Exception:
        positions = []

    json_resp = {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk": [int(x) for x in walk],
        "positions": positions,
        "total_weight": float(total),
        "tol": tol
    }
    return JSONResponse(json_resp)


if __name__ == "__main__":
    # Run without reloader to avoid multi-process side-effects.
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
