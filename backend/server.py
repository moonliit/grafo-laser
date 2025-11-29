from chinese_postman_route import chinese_postman_route
from typing import Dict, Any
import networkx as nx
import math

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import paho.mqtt.client as mqtt
import tomllib
import json

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

# Get MQT info
with open("mqtt.toml", "rb") as f:
    config = tomllib.load(f)

broker = config["broker"]
topics = config["topics"]
options = config["options"]

# Create MQTT client
client = mqtt.Client(
    client_id=broker["client_id"],
    clean_session=broker["clean_session"]
)

# TLS if requested
if broker["use_tls"]:
    client.tls_set()
    if broker["tls_insecure"]:
        client.tls_insecure_set(True)

# Connect
CONNECT_CLIENT = True
if CONNECT_CLIENT:
    client.connect(
        host=broker["host"],
        port=broker["port"],
        keepalive=broker["keepalive"]
    )
    client.loop_start()
else:
    client = None

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

    # compute walk
    walk, total = chinese_postman_route(G)

    # prepare nodes/edges for JSON
    nodes = {str(n): [float(G.nodes[n]["pos"][0]), float(G.nodes[n]["pos"][1])] for n in G.nodes()}
    edges = [[int(u), int(v), float(G[u][v].get("weight", 1.0))] for u, v in G.edges()]

    json_resp = {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "walk": [int(x) for x in walk],
        "total_weight": float(total)
    }

    # prepare positions for MQTT
    positions = [{"x": float(G.nodes[n]["pos"][0]), "y": float(G.nodes[n]["pos"][1])} for n in walk]
    positions.pop()

    if client is not None:
        mqtt_payload = json.dumps(positions)
        mqtt_result = client.publish(
            topic=topics["publish"],
            payload=mqtt_payload,
            qos=options["qos"],
            retain=options["retain"]
        )
        print(f"Published frame to {topics["publish"]}")
        print(f"result: {mqtt_result.rc}")
    else:
        print(f"Frame: {positions}")

    return JSONResponse(json_resp)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
