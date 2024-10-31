import sys
import utils 
from collections import Counter

import pandas as pd
import numpy as np

from scipy.spatial.distance import euclidean

import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu
from geomapi.nodes import PointCloudNode
import geomapi.tools as tl

import topologicpy as tp
from topologicpy.Graph import Graph
from topologicpy.Dictionary import Dictionary

from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex

from topologicpy.Vector import Vector
from topologicpy.Plotly import Plotly

import matplotlib.pyplot as plt
import plotly.graph_objects as go




def load_ttl_tech_graph(graph_path):
    # Parse the RDF graph from the provided path
    graph = Graph().parse(graph_path)

    # Convert the RDF graph into nodes (Assuming `tl.graph_to_nodes` is defined elsewhere)
    nodes = tl.graph_to_nodes(graph)

    # Categorize nodes by type and extract their class_id and object_id
    node_tech_groups = {

        'lights_nodes': [n for n in nodes if 'Lights' in n.subject and isinstance(n, PointCloudNode)],
        'radiators_nodes': [n for n in nodes if 'Radiators' in n.subject and isinstance(n, PointCloudNode)],
        'hvac_nodes': [n for n in nodes if 'HVAC' in n.subject and isinstance(n, PointCloudNode)]
    }

    # Print node counts for each category
    print(f'{len(node_tech_groups["lights_nodes"])} lights_nodes detected!')
    print(f'{len(node_tech_groups["radiators_nodes"])} radiators_nodes detected!')
    print(f'{len(node_tech_groups["hvac_nodes"])} hvac_nodes detected!')

    # Extract class_id and object_id using a loop
    class_object_ids = {}
    for node_type, nodes in node_tech_groups.items():
        class_object_ids[node_type] = [(n.class_id, n.object_id) for n in nodes if hasattr(n, 'class_id') and hasattr(n, 'object_id')]

    # Return the node groups and their associated class-object IDs
    return node_tech_groups, class_object_ids



# Create vertices from coordinates
def create_vertices(coords):
    return [Vertex.ByCoordinates(coord[0], coord[1], coord[2]) for coord in coords]

# Closed wire
def create_closed_wire(coords):
    vertices = create_vertices(coords)
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])  # Close the wire by adding the first vertex at the end
    wire = Wire.ByVertices(vertices)
    if wire is None:
        print(f"Error: Could not create wire for vertices: {coords}")
    return wire

# Align vertices
def snap_vertex(v1, v2, tolerance=1e-6):
    # Compare coordinates with a tolerance to snap
    return abs(v1[0] - v2[0]) < tolerance and abs(v1[1] - v2[1]) < tolerance and abs(v1[2] - v2[2]) < tolerance

# Ensure consistent and aligned vertices
def align_vertices(coords_b, coords_t, tolerance=1e-6):
    for i in range(len(coords_b)):
        if snap_vertex(coords_b[i], coords_t[i], tolerance):
            print(f"Snapping top vertex {coords_t[i]} to bottom vertex {coords_b[i]}")
            coords_t[i] = [coords_b[i][0], coords_b[i][1], coords_t[i][2]]  # Snap top to bottom, keep z-coordinate
    return coords_b, coords_t

# Remove duplicate faces
def remove_duplicate_faces(faces):
    unique_faces = []
    for face in faces:
        if face not in unique_faces:
            unique_faces.append(face)
        else:
            print(f"Removing duplicate face: {face}")
    return unique_faces

# Extract edges from faces and track them
def track_face_edges(faces):
    
    edge_counter = Counter()
    
    for face in faces:
        edges = Topology.Edges(face)  # Use Topology.Edges() to extract edges from the face
        for edge in edges:
            edge_vertices = Topology.Vertices(edge)  # Extract vertices from the edge
            edge_coords = tuple(sorted((tuple(edge_vertices[0].Coordinates()), tuple(edge_vertices[1].Coordinates()))))
            edge_counter[edge_coords] += 1
            print(f"Edge {edge_coords} occurs {edge_counter[edge_coords]} times.")
    
    return edge_counter


