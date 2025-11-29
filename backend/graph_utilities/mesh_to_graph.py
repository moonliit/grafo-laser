import networkx as nx


def obj_to_graph(obj_filepath):
    """
    Converts an OBJ mesh file into a NetworkX graph.

    Args:
        obj_filepath (str): The path to the OBJ file.

    Returns:
        networkx.Graph: A NetworkX graph representing the mesh.
    """
    G = nx.Graph()
    vertices = []

    with open(obj_filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                # Parse vertex coordinates
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append((x, y, z))
                # Add node to graph with position as attribute
                G.add_node(len(vertices) - 1, pos=(x, y, z))
            elif parts[0] == 'f':
                # Parse face and add edges
                # OBJ indices are 1-based, convert to 0-based
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                for i in range(len(face_indices)):
                    node1 = face_indices[i]
                    node2 = face_indices[(i + 1) % len(face_indices)]
                    G.add_edge(node1, node2)
    return G
