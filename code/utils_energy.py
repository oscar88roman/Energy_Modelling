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