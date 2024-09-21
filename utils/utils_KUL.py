import os
import os.path

from pathlib import Path
import rdflib
from rdflib import Graph
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import Namespace, RDF, XSD
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import laspy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import open3d as o3d

import geomapi.tools as tl
from geomapi.utils import geometryutils as gmu
from geomapi.nodes import PointCloudNode

from scipy.spatial import KDTree, ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors

import context_KUL
import utils_KUL as kul

import json


# IMPORT POINT CLOUD
def load_point_cloud(file_name):

    laz = laspy.read(file_name)
    pcd = gmu.las_to_pcd(laz)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd_nodes = PointCloudNode(resource=pcd)
    normals = np.asarray(pcd.normals)

    return laz, pcd, pcd_nodes, normals

# IMPORT THE GEOMETRIC GRAPH
def load_graph(graph_path):

    # Parse the RDF graph from the provided path
    graph = Graph().parse(graph_path)

    # Convert the RDF graph into nodes
    nodes = tl.graph_to_nodes(graph)

    # Categorize nodes by type and extract their class_id and object_id
    node_groups = {
        'unassigned': [n for n in nodes if 'unassigned' in n.subject and isinstance(n, PointCloudNode)],
        'floor': [n for n in nodes if 'floors' in n.subject and isinstance(n, PointCloudNode)],
        'ceiling': [n for n in nodes if 'ceilings' in n.subject and isinstance(n, PointCloudNode)],
        'wall': [n for n in nodes if 'walls' in n.subject and isinstance(n, PointCloudNode)],
        'column': [n for n in nodes if 'columns' in n.subject and isinstance(n, PointCloudNode)],
        'door': [n for n in nodes if 'doors' in n.subject and isinstance(n, PointCloudNode)],
        'window': [n for n in nodes if 'windows' in n.subject and isinstance(n, PointCloudNode)],
        'level': [n for n in nodes if 'level' in n.subject and isinstance(n, PointCloudNode)]
    }

    # Extract class_id and object_id using a loop
    class_object_ids = {}
    for node_type, nodes in node_groups.items():
        class_object_ids[node_type] = [(n.class_id, n.object_id) for n in nodes if hasattr(n, 'class_id') and hasattr(n, 'object_id')]

    # Debugging information
    # for key, value in class_object_ids.items():
    #     print(f'{key.capitalize()} Class-Object IDs:', value)

    return node_groups, class_object_ids



def load_tt_graph(graph_path):
    # Parse the RDF graph from the provided path
    graph = Graph().parse(graph_path)

    # Convert the RDF graph into nodes (Assuming `tl.graph_to_nodes` is defined elsewhere)
    nodes = tl.graph_to_nodes(graph)

    # Categorize nodes by type and extract their class_id and object_id
    node_groups = {
        'floors_nodes': [n for n in nodes if 'Floors' in n.subject and isinstance(n, PointCloudNode)],
        'ceilings_nodes': [n for n in nodes if 'Ceilings' in n.subject and isinstance(n, PointCloudNode)],
        'walls_nodes': [n for n in nodes if 'Walls' in n.subject and isinstance(n, PointCloudNode)],
        'columns_nodes': [n for n in nodes if 'Columns' in n.subject and isinstance(n, PointCloudNode)],
        'windows_nodes': [n for n in nodes if 'Windows' in n.subject and isinstance(n, PointCloudNode)],
        'doors_nodes': [n for n in nodes if 'Doors' in n.subject and isinstance(n, PointCloudNode)],
        'lights_nodes': [n for n in nodes if 'Lights' in n.subject and isinstance(n, PointCloudNode)],
        'radiators_nodes': [n for n in nodes if 'Radiators' in n.subject and isinstance(n, PointCloudNode)],
        'hvac_nodes': [n for n in nodes if 'HVAC' in n.subject and isinstance(n, PointCloudNode)]
    }

    # Print node counts for each category
    print(f'{len(node_groups["floors_nodes"])} floors_nodes detected!')
    print(f'{len(node_groups["ceilings_nodes"])} ceilings_nodes detected!')
    print(f'{len(node_groups["walls_nodes"])} walls_nodes detected!')
    print(f'{len(node_groups["columns_nodes"])} columns_nodes detected!')
    print(f'{len(node_groups["windows_nodes"])} windows_nodes detected!')
    print(f'{len(node_groups["doors_nodes"])} doors_nodes detected!')
    print(f'{len(node_groups["lights_nodes"])} lights_nodes detected!')
    print(f'{len(node_groups["radiators_nodes"])} radiators_nodes detected!')
    print(f'{len(node_groups["hvac_nodes"])} hvac_nodes detected!')

    # Extract class_id and object_id using a loop
    class_object_ids = {}
    for node_type, nodes in node_groups.items():
        class_object_ids[node_type] = [(n.class_id, n.object_id) for n in nodes if hasattr(n, 'class_id') and hasattr(n, 'object_id')]

    # Return the node groups and their associated class-object IDs
    return node_groups, class_object_ids










# PAERSE ttl FILE
def parse_ttl_file(graph_path):
    
    g = rdflib.Graph()
    g.parse(graph_path, format='turtle')

    # Namespaces
    RDF = rdflib.RDF
    RDFS = rdflib.RDFS
    SCHEMA = rdflib.Namespace("http://schema.org/")
    
    node_types = {}

    for s, p, o in g:
        class_name = s.split('/')[-1]  # Get class name from URI

        if p == RDFS.subClassOf:
            node_types[class_name] = {'id': None, 'color': None}
        
        elif p == SCHEMA.id and class_name in node_types:
            node_types[class_name]['id'] = str(o)

        elif p == SCHEMA.color and class_name in node_types:
            node_types[class_name]['color'] = str(o)

    print("Parsed node types:", node_types)  # Debugging line

    return node_types


def parse_ttl_to_df(ttl_content):
    graph = rdflib.Graph()
    
    # Make sure ttl_content is a string or bytes
    if isinstance(ttl_content, (str, bytes)):
        try:
            # Parse the TTL content
            graph.parse(data=ttl_content, format="turtle")
        except Exception as e:
            raise RuntimeError(f"Error parsing TTL content: {e}")
    else:
        raise TypeError(f"Expected string or bytes, got {type(ttl_content)}")

    # Extract data from the graph and create a DataFrame
    data = []
    for subj, pred, obj in graph:
        data.append({'subject': subj, 'predicate': pred, 'object': obj})

    df = pd.DataFrame(data)

    return df


# PROCESS laz NODES
def process_laz_nodes(laz, node_type, node_groups):
    if node_type not in node_groups:
        raise ValueError(f"Node type '{node_type}' is not a valid category.")

    node_list = node_groups[node_type]

    # Convert LAS attributes to numpy arrays
    laz_classes = np.array(laz.classification)
    laz_objects = np.array(laz.user_data)  # Assuming 'objects' are stored in 'user_data'
    laz_xyz = np.array(laz.xyz)

    processed_nodes = []

    for n in node_list:
        # Extract indices of points corresponding to this node's class_id and object_id
        idx = np.where((laz_classes == n.class_id) & (laz_objects == n.object_id))[0]
        
        if len(idx) == 0:
            print(f'No points found for node with class_id {n.class_id} and object_id {n.object_id}')
            continue
        
        # Create a new Open3D PointCloud and set its points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(laz_xyz[idx])
        n.resource = pcd

        processed_nodes.append(n)

    print(f'{len(node_list)} {node_type} nodes processed!')
    return processed_nodes

# COLLECT NODES
def collect_nodes(laz, ttl_file_path, node_groups):
    # Parse the Turtle file to get node types
    node_types = parse_ttl_file(ttl_file_path)

    # Initialize dictionary to collect processed nodes
    collected_nodes = {key: [] for key in node_groups.keys()}

    # Process nodes based on the node types from the Turtle file
    for node_type, nodes in node_groups.items():
        if node_type in node_types:
            processed_nodes = process_laz_nodes(laz, node_type, node_groups)
            collected_nodes[node_type].extend(processed_nodes)
        else:
            print(f"Node type '{node_type}' not found in Turtle file.")

    return collected_nodes


def create_class_object_to_idx_mapping(class_object_ids):
    
    return {pair: idx for idx, pair in enumerate(class_object_ids)}


##________________________________________
def extract_objects_building(laz, graph_path):

    # Parse the RDF graph and convert nodes
    graph = Graph().parse(str(graph_path))
    nodes = tl.graph_to_nodes(graph)

    # Categorize nodes by type (floors, ceilings, walls, etc.)
    unassigned_nodes = [n for n in nodes if 'unassigned' in n.subject.lower() and isinstance(n, PointCloudNode)]
    floors_nodes = [n for n in nodes if 'floors' in n.subject.lower() and isinstance(n, PointCloudNode)]
    ceilings_nodes = [n for n in nodes if 'ceilings' in n.subject.lower() and isinstance(n, PointCloudNode)]
    walls_nodes = [n for n in nodes if 'walls' in n.subject.lower() and isinstance(n, PointCloudNode)]
    columns_nodes = [n for n in nodes if 'columns' in n.subject.lower() and isinstance(n, PointCloudNode)]
    doors_nodes = [n for n in nodes if 'doors' in n.subject.lower() and isinstance(n, PointCloudNode)]
    windows_nodes = [n for n in nodes if 'windows' in n.subject.lower() and isinstance(n, PointCloudNode)]
    levels_nodes = [n for n in nodes if 'levels' in n.subject.lower() and isinstance(n, PointCloudNode)]

    # Helper function to extract information for further processes
    def extract_info(node_list):

        data_list = []
        
        for n in node_list:

            idx = np.where((laz['classes'] == n.class_id) & (laz['objects'] == n.object_id))

            # Extract coordinates based on indices
            x_coords = laz['x'][idx]
            y_coords = laz['y'][idx]
            z_coords = laz['z'][idx]
            
            # Stack coordinates vertically
            coordinates = np.vstack((x_coords, y_coords, z_coords)).T

            # Set point cloud resource and calculate oriented bounding box (OBB)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(laz.xyz[idx])
        
            # Assign the point cloud to the node
            n.resource = pcd
            n.get_oriented_bounding_box()
            
            # Collect useful data such as indices, OBB, and color
            data = {
                'indices': idx,
                'oriented_bounding_box': n.orientedBoundingBox,
                'coordinates': coordinates,
                'resource': n.resource,
                'obb_color': n.orientedBoundingBox.color,
                'obb_center': n.orientedBoundingBox.center,
                'obb_extent': n.orientedBoundingBox.extent,
                'class_id': n.class_id,
                'object_id': n.object_id
            }
          
            data_list.append(data)

        return data_list

    # Extract data for each type of object
    clutter = extract_info(unassigned_nodes)
    print(len(clutter), 'clutter detected')

    floors = extract_info(floors_nodes)
    print(len(floors), "floors detected")

    ceilings = extract_info(ceilings_nodes)
    print(len(ceilings), "ceilings detected")

    walls = extract_info(walls_nodes)
    print(len(walls), 'walls detected')

    columns = extract_info(columns_nodes)
    print(len(columns), 'columns detected')

    doors = extract_info(doors_nodes)
    print(len(doors), 'doors detected')

    windows = extract_info(windows_nodes)
    print(len(windows), 'windows detected')

    levels = extract_info(levels_nodes)
    print(len(levels), "levels detected")

    # Return the extracted data for further processing
    return {
        'clutter': clutter,
        'floors': floors,
        'ceilings': ceilings,
        'walls': walls,
        'columns': columns,
        'doors': doors,
        'windows': windows,
        'levels': levels
    }




##______________________________LEVELS - FLOORS - CEILINGS_____________________________________

t_thickness_levels = 0.80
th_hull_area = 7.50


def planar_xy_hull(laz, node_ids, avg_z):
    
    points_2d = []
    
    for n in node_ids:
        # Find the points in the laz dataset that match the current node's class_id and object_id
        idx = np.where((laz['classes'] == n.class_id) & (laz['objects'] == n.object_id))
        if idx[0].size == 0:
            continue  # Skip if no points are found for the given node_id
        
        # Extract x and y values for the selected points
        x_values = laz.x[idx]
        y_values = laz.y[idx]
        
        # Project to 2D by combining x and y values
        projected_points = np.column_stack((x_values, y_values))
        points_2d.extend(projected_points)
    
    # Convert list of 2D points to a numpy array
    points_2d = np.array(points_2d)
    
    # Ensure there are enough points to compute a convex hull
    if points_2d.shape[0] < 3:
        print(f"Not enough points to compute convex hull for avg_z {avg_z}")
        return None
    
    try:
        # Compute the convex hull using the 2D points
        hull = ConvexHull(points_2d)
        
        # Calculate the area of the convex hull (volume property is the area for 2D hull)
        hull_area = hull.volume
        
        if hull_area >= th_hull_area:  # Only return the hull if the area is >= 1 m²
            return hull
        else:
            print(f"Hull area {hull_area:.2f} is less than 1 m² for avg_z {avg_z}")
            return None
    except QhullError:
        # Handle errors in case the convex hull computation fails
        print(f"QhullError for avg_z {avg_z}")
        return None

    
def load_levels(laz, graph_path):
    # Parse the graph
    graph = Graph().parse(str(graph_path))
    nodes = tl.graph_to_nodes(graph)

    # Separate nodes by type
    ceilings_nodes = [n for n in nodes if 'ceilings' in n.subject.lower() and isinstance(n, PointCloudNode)]
    floors_nodes = [n for n in nodes if 'floors' in n.subject.lower() and isinstance(n, PointCloudNode)]
    level_nodes = [n for n in nodes if 'level' in n.subject.lower() and isinstance(n, PointCloudNode)]

    # Initialize the tolerance
    t_floor = 0.05
    t_ceiling = 0.05

    # Initialize lists for merged floor and ceiling data
    floors_z = []
    ceilings_z = []
    
    # Lists to store z_min and z_max values for bounding box computation
    floors_z_bbox = []  # (z_min, z_max) for each floor
    ceilings_z_bbox = []  # (z_min, z_max) for each ceiling

    # Calculate average z-values and z_min, z_max for floors
    for n in floors_nodes:
        idx = np.where((laz.classes == n.class_id) & (laz.objects == n.object_id))
        z_values = laz.z[idx]
        if len(z_values) > 0:
            avg_z = np.mean(z_values)
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            # Discard this floor if the z_max - z_min is greater than 1
            if (z_max - z_min) > t_thickness_levels:
                continue
            
            merged = False
            for i, (existing_avg_z, floor_ids) in enumerate(floors_z):
                if abs(existing_avg_z - avg_z) <= t_floor:
                    new_avg_z = (existing_avg_z * len(floor_ids) + avg_z) / (len(floor_ids) + 1)
                    floors_z[i] = (new_avg_z, floor_ids + [n])
                    floors_z_bbox[i] = (min(floors_z_bbox[i][0], z_min), max(floors_z_bbox[i][1], z_max))
                    merged = True
                    break
            
            if not merged:
                floors_z.append((avg_z, [n]))
                floors_z_bbox.append((z_min, z_max))  # Store z_min and z_max for this floor

    # Calculate average z-values and z_min, z_max for ceilings
    for n in ceilings_nodes:
        idx = np.where((laz.classes == n.class_id) & (laz.objects == n.object_id))
        z_values = laz.z[idx]
        if len(z_values) > 0:
            avg_z = np.mean(z_values)
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            
            # Discard this ceiling if the z_max - z_min is greater than 1
            if (z_max - z_min) > t_thickness_levels:
                continue
            
            merged = False
            for i, (existing_avg_z, ceiling_ids) in enumerate(ceilings_z):
                if abs(existing_avg_z - avg_z) <= t_ceiling:
                    new_avg_z = (existing_avg_z * len(ceiling_ids) + avg_z) / (len(ceiling_ids) + 1)
                    ceilings_z[i] = (new_avg_z, ceiling_ids + [n])
                    ceilings_z_bbox[i] = (min(ceilings_z_bbox[i][0], z_min), max(ceilings_z_bbox[i][1], z_max))
                    merged = True
                    break
            
            if not merged:
                ceilings_z.append((avg_z, [n]))
                ceilings_z_bbox.append((z_min, z_max))  # Store z_min and z_max for this ceiling

    print(f'Find {len(ceilings_nodes)} ceilings after normalization {len(ceilings_z)}')
    print(f'Find {len(floors_nodes)} floors after normalization {len(floors_z)}')
    print(f'Find {len(level_nodes)} levels')

    # Compute convex hulls and bounding boxes
    floor_hulls = []
    floor_hull_vertices = []
    floor_bboxes = []

    for avg_z, floor_ids in floors_z:
        # Get the convex hull
        hull = planar_xy_hull(floor_ids, avg_z)
        
        if hull is None:  # hulls with area < 1m² are already discarded in planar_xy_hull
            continue
        
        floor_hulls.append(hull)
        
        # Get the vertices of the convex hull
        vertices = hull.points[hull.vertices]
        floor_hull_vertices.append(vertices)
        
        # Compute the 2D bounding box
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

        # Extend to 3D bounding box using z_min and z_max
        z_min, z_max = floors_z_bbox[floors_z.index((avg_z, floor_ids))]
        floor_bboxes.append([
            (min_x, min_y, z_min), (max_x, min_y, z_min),
            (max_x, max_y, z_min), (min_x, max_y, z_min),
            (min_x, min_y, z_max), (max_x, min_y, z_max),
            (max_x, max_y, z_max), (min_x, max_y, z_max)
        ])

    ceiling_hulls = []
    ceiling_hull_vertices = []
    ceiling_bboxes = []

    for avg_z, ceiling_ids in ceilings_z:
        # Get the convex hull
        hull = planar_xy_hull(ceiling_ids, avg_z)
        
        if hull is None:  
            continue
        
        ceiling_hulls.append(hull)

        # Get the vertices of the convex hull
        vertices = hull.points[hull.vertices]
        ceiling_hull_vertices.append(vertices)
        
        # Compute the 2D bounding box
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        
        # Extend to 3D bounding box using z_min and z_max
        z_min, z_max = ceilings_z_bbox[ceilings_z.index((avg_z, ceiling_ids))]
        ceiling_bboxes.append([
            (min_x, min_y, z_min), (max_x, min_y, z_min),
            (max_x, max_y, z_min), (min_x, max_y, z_min),
            (min_x, min_y, z_max), (max_x, min_y, z_max),
            (max_x, max_y, z_max), (min_x, max_y, z_max)
        ])

    # Compute average z-values for floors and ceilings
    floor_avg_z_values = [avg_z for avg_z, _ in floors_z]
    ceiling_avg_z_values = [avg_z for avg_z, _ in ceilings_z]

    floor_z_avg = np.mean(floor_avg_z_values) if floor_avg_z_values else None
    ceiling_z_avg = np.mean(ceiling_avg_z_values) if ceiling_avg_z_values else None


    # Calculate thickness for each floor and ceiling by comparing floor z_max with ceiling z_min
    thicknesses = []
    for floor_bbox, ceiling_bbox in zip(floors_z_bbox, ceilings_z_bbox):
        thickness = ceiling_bbox[0] - floor_bbox[1]  # z_min of ceiling - z_max of floor
        thicknesses.append(thickness)

    
    # Create and return the dictionary
    results = {
        'floors_nodes': floors_nodes,
        'ceilings_nodes': ceilings_nodes,
        'floor_z_avg': floor_z_avg,
        'ceiling_z_avg': ceiling_z_avg,
        'level_nodes': level_nodes,
        'floors_z_bbox': floors_z_bbox,
        'ceilings_z_bbox': ceilings_z_bbox,
        'thicknesses': thicknesses,
        'floor_hulls': floor_hulls,
        'floor_hull_vertices': floor_hull_vertices,
        'ceiling_hulls': ceiling_hulls,
        'ceiling_hull_vertices': ceiling_hull_vertices,
        'floor_bboxes': floor_bboxes,
        'ceiling_bboxes': ceiling_bboxes
    }

    return results


def levels_bbox(floor_bboxes, ceiling_bboxes):
    # Filter out None values
    filtered_floor_bboxes = [bbox for bbox in floor_bboxes if bbox is not None]
    filtered_ceiling_bboxes = [bbox for bbox in ceiling_bboxes if bbox is not None]

    # Convert lists to numpy arrays
    floor_bboxes_array = np.array(filtered_floor_bboxes, dtype=object)
    ceiling_bboxes_array = np.array(filtered_ceiling_bboxes, dtype=object)
    
    # Concatenate floor and ceiling bounding boxes
    levels_bboxes = np.concatenate((floor_bboxes_array, ceiling_bboxes_array), axis=0)
    
    return levels_bboxes




# def get_levels_bbox (laz, floors_z, ceilings_z, floor_z, ceiling_z, levels):
    # IF BBOX == floors_poisition
        # then:
        # take bbox and this floor

def select_nodes_intersecting_bounding_box(node: dict, nodelist: List[dict], u: float = 0.5, v: float = 0.5, w: float = 0.5) -> List[dict]:
    """Select nodes whose bounding boxes intersect with the source node's bounding box.
    
    Args:
        node (dict): Source node with bounding box information.
        nodelist (List[dict]): Target node list with bounding box information.
        u (float, optional): Offset in X. Defaults to 0.5m.
        v (float, optional): Offset in Y. Defaults to 0.5m.
        w (float, optional): Offset in Z. Defaults to 0.5m.
    
    Returns:
        List[dict]: Nodes with intersecting bounding boxes.
    """
    # Get the oriented bounding box of the source node
    box = node.get('oriented_bounding_box')
    
    if box is None:
        raise ValueError("The source node does not have an 'oriented_bounding_box' key.")

    # Expand the bounding box
    box = gmu.expand_box(box, u=u, v=v, w=w)

    # Get the oriented bounding boxes for all nodes in the nodelist
    boxes = [n.get('oriented_bounding_box') for n in nodelist]
    
    # Check for missing bounding boxes
    if None in boxes:
        raise ValueError("One or more nodes in the nodelist do not have an 'oriented_bounding_box' key.")
    
    boxes = np.array(boxes, dtype=object)  # Convert to numpy array for easier manipulation
    
    # Find intersections
    idx_list = gmu.get_box_intersections(box, boxes)
    selected_node_list = [nodelist[idx] for idx in idx_list]
    
    return selected_node_list




##________________WALLS____________________________

def compute_plane_from_points(p1, p2, p3):
    """Compute plane coefficients (a, b, c, d) from 3 non-collinear points."""
    # Vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    # Plane normal (a, b, c)
    normal = np.cross(v1, v2)
    a, b, c = normal

    # Plane offset (d)
    d = -np.dot(normal, p1)

    return a, b, c, d

def distance_from_plane(point, plane):
    """Compute distance from a point to the plane."""
    a, b, c, d = plane
    x, y, z = point
    return abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

def ransac_plane_fitting(points, distance_threshold=0.03, num_iterations=1000):
    """Fit a plane to a set of 3D points using RANSAC."""
    best_plane = None
    best_inliers = []

    num_points = len(points)

    for _ in range(num_iterations):
        # Randomly select 3 points
        indices = np.random.choice(num_points, 3, replace=False)
        p1, p2, p3 = points[indices]

        # Compute the plane model
        plane = compute_plane_from_points(p1, p2, p3)

        # Compute inliers
        distances = np.array([distance_from_plane(point, plane) for point in points])
        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_plane = plane
            best_inliers = inliers

    return best_plane, best_inliers


def check_dimension_equality(var1, var2):
    # Check if both variables are numpy arrays
    if isinstance(var1, np.ndarray) and isinstance(var2, np.ndarray):
        return var1.shape == var2.shape
    # Check if both are lists (or other sequences) and compare their lengths
    elif isinstance(var1, list) and isinstance(var2, list):
        return len(var1) == len(var2)
    # For other types of sequences or data structures
    try:
        return len(var1) == len(var2)
    except TypeError:
        return False


# wall_metadata = []

# for index, wall_data in enumerate(walls_data):
#     result = process_wall_data(laz, wall_data)

#     if result is not None:
#         # Unpack the result based on the actual number of values returned
#         (wall_data["object_id"], start_point_middle, end_point_middle, start_point_base, end_point_base,
#          start_point_top, end_point_top, base_center, top_center, min_z, max_z,
#          length, thickness, z_axes) = result  # Adjust unpacking to 13 values

#         # Calculate angle_degrees if necessary or set it to None if not provided
#         angle_degrees = None  # Set default value or calculate it if needed

#         # Print detailed information
#         print(f'Index wall: {wall_data["object_id"]}')
#         print(f'Start point middle: {start_point_middle}')
#         print(f'End point middle: {end_point_middle}')
#         print(f'Start point base: {start_point_base}')
#         print(f'End point base: {end_point_base}')
#         print(f'Start point top: {start_point_top}')
#         print(f'End point top: {end_point_top}')
#         print(f'Base center: {base_center}')
#         print(f'Top center: {top_center}')
#         print(f'Length: {length}')
#         print(f'Thickness: {thickness}')
#         print(f'Height (Z-Axis): {z_axes}')
#         if angle_degrees is not None:
#             print(f'Angle with Z-Axis: {angle_degrees:.2f} degrees')

#         # Form the metadata dictionary
#         metadata = {
#             'index': index,
#             'start_point_middle': start_point_middle,
#             'end_point_middle': end_point_middle,
#             'start_point_base': start_point_base,
#             'end_point_base': end_point_base,
#             'start_point_top': start_point_top,
#             'end_point_top': end_point_top,
#             'base_center': base_center,
#             'top_center': top_center,
#             'min_z': min_z,
#             'max_z': max_z,
#             'length': length,
#             'thickness': thickness,
#             'height': z_axes,
#             'angle_degrees': angle_degrees
#         }

#         wall_metadata.append(metadata)

#     else:
#         # If result is None, append a default or empty metadata entry if needed
#         wall_metadata.append({
#             'index': index,
#             'start_point_middle': None,
#             'end_point_middle': None,
#             'start_point_base': None,
#             'end_point_base': None,
#             'start_point_top': None,
#             'end_point_top': None,
#             'base_center': None,
#             'top_center': None,
#             'min_z': None,
#             'max_z': None,
#             'length': None,
#             'thickness': None,
#             'height': None,
#             'angle_degrees': None
#         })



def create_sections(points: np.ndarray, z_min: float, z_max: float):

    if points.shape[1] != 3:
        raise ValueError("Input array must have exactly 3 columns for x, y, and z coordinates.")
    
    # Create the base section by setting all z-coordinates to z_min
    base_section = points.copy()
    base_section[:, 2] = z_min
    
    # Create the top section by adding z_max to the original z-coordinates
    top_section = points.copy()
    top_section[:, 2] += z_max
    
    return base_section, top_section


def compute_plane_and_normal(points, distance_threshold=0.03, num_iterations=1000):
    # Compute plane model and normal vector
    plane, inliers = kul.ransac_plane_fitting(points, distance_threshold, num_iterations)
    normal = plane[:3]
    normal[2] = 0  # Project to 2D
    normal /= np.linalg.norm(normal)
    return plane, normal, inliers


def adjust_face_center(n):
    # Adjust face center to the correct height
    
    # Extract points from PointCloud and convert to numpy array
    face_points = np.asarray(n['resource']['points'].points)[n['inliers']]
    
    # Calculate the mean (center) of the face points
    face_center = np.mean(face_points, axis=0)
    
    # Adjust the z-coordinate based on the height and offset
    face_center[2] = n['base_constraint']['height'] + n['base_offset']
    
    return face_center



def update_sign_and_normal(n, face_center, normal):
    box_center = np.array(n['orientedBoundingBox']['center'])
    box_center[2] = n['base_constraint']['height'] + n['base_offset']
    sign = np.sign(np.dot(normal, face_center - box_center))
    normal *= -1 if sign == -1 else 1
    return sign, normal

def handle_thickness_adjustment(n, ceilings_nodes, floors_nodes, t_thickness):
    if n['orientedBoundingBox']['extent'][2] > t_thickness:
        return
    combined_list = ceilings_nodes + floors_nodes
    reference_points = np.concatenate([node['resource']['points'] for node in combined_list if 'resource' in node], axis=0)

    top_point = np.array(n['orientedBoundingBox']['center'])
    top_point[2] = n['base_constraint']['height'] + n['base_offset'] + n['height']
    bottom_point = np.array(n['orientedBoundingBox']['center'])
    
    idx, _ = kul.compute_nearest_neighbors(np.array([top_point, bottom_point]), reference_points)
    points = reference_points[idx[:, 0]]
    idx = idx[np.argmin(np.abs(np.einsum('i,ji->j', n['normal'], points - bottom_point)))]
    
    point = reference_points[idx]
    point[2] = n['base_constraint']['height'] + n['base_offset']
    sign = np.sign(np.dot(n['normal'], point - bottom_point))
    return sign

def segment_plane_and_adjust_height(walls_nodes, ceilings_nodes, floors_nodes, t_thickness):
    for n in walls_nodes:
        if 'resource' not in n or 'points' not in n['resource']:
            print(f"Error: Missing 'resource' or 'points' in node {n}")
            continue

        points = np.asarray(n['resource']['points'])
        
        # Compute plane and normal
        plane, normal, inliers = compute_plane_and_normal(points)
        
        # Adjust face center
        face_center = adjust_face_center(n)
        n['faceCenter'] = face_center

        # Compute and update the sign and normal
        sign, normal = update_sign_and_normal(n, face_center, normal)
        n['sign'] = sign
        n['normal'] = normal

        # Handle thickness adjustment
        thickness_sign = handle_thickness_adjustment(n, ceilings_nodes, floors_nodes, t_thickness)
        if thickness_sign is not None:
            n['sign'] = thickness_sign

        print(f'name: {n.get("name", "Unnamed")}, plane: {plane}, inliers: {len(inliers)}/{len(points)}')


def compute_plane_from_points(p1, p2, p3):
    """Compute plane coefficients (a, b, c, d) from 3 non-collinear points."""
    # Vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    # Plane normal (a, b, c)
    normal = np.cross(v1, v2)
    a, b, c = normal

    # Plane offset (d)
    d = -np.dot(normal, p1)

    return a, b, c, d

def distance_from_plane(point, plane):
    """Compute distance from a point to the plane."""
    a, b, c, d = plane
    x, y, z = point
    return abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)


def ransac_inliers_normals(points, distance_threshold=0.03, num_iterations=500, min_inliers=0.8):
    best_planes = None
    best_inliers = []
    
    num_points = points.shape[0]
    target_inliers_count = int(min_inliers * num_points)

    for i in range(num_iterations):
        # Randomly select 3 points to define a plane
        sample_indices = np.random.choice(num_points, 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Compute the plane from the 3 points
        plane = compute_plane_from_points(p1, p2, p3)

        if len(plane) < 4:
            raise ValueError("The plane returned from RANSAC does not have the expected format.")
        
        # Extract the normal vector from the plane model
        normal = np.array(plane[:3])
        normal[2] = 0  # Project to 2D (zero out the z-component)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector
        
        # Vectorized distance calculation for all points
        distances = np.abs(np.dot(points - p1, plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])

        # Get inliers that satisfy the distance threshold
        inliers = np.where(distances < distance_threshold)[0]

        # If this plane has more inliers, save it
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_planes = plane

            # Early stopping if the number of inliers is sufficient
            if len(inliers) > target_inliers_count:
                break

    return best_planes, best_inliers, normal

def adjust_face_center(inlier_points):
    # Calculate the min and max z values
    min_z = np.min(inlier_points[:, 2])
    max_z = np.max(inlier_points[:, 2])
    height = max_z - min_z

    # Calculate the mean (center) of the inlier points
    face_center = np.mean(inlier_points, axis=0)
    
    # Adjust the z-coordinate based on height (if needed)
    face_center[2] = height / 2

    return face_center


def compute_nearest_neighbors(query_points, data_points, n_neighbors = 3):

    # Initialize NearestNeighbors with the number of neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data_points)
    
    # Find the nearest neighbors
    distances, indices = nbrs.kneighbors(query_points)
    
    return indices, distances

# def create_sections(points: np.ndarray, z_min: float, z_max: float):

#     if points.shape[1] != 3:
#         raise ValueError("Input array must have exactly 3 columns for x, y, and z coordinates.")
    
#     # Create the base section by setting all z-coordinates to z_min
#     base_section = points.copy()
#     base_section[:, 2] = z_min
#     base_center = np.mean(base_section)
    
#     # Create the top section by adding z_max to the original z-coordinates
#     top_section = points.copy()
#     top_section[:, 2] += z_max
#     top_center = np.mean(top_section)

#     z_axis_vector = np.array([0, 0, 1])
#     z_axes = top_center - base_center
#     dot_product = np.dot(z_axes, z_axis_vector)
    
#     # Calculate the magnitude of the direction vector
#     magnitude_direction = np.linalg.norm(z_axes)
#     cos_theta = dot_product / magnitude_direction
#     angle_radians = np.arccos(cos_theta)
    
#     # Convert the angle to degrees
#     angle_degrees = np.degrees(angle_radians)
    
#     return base_section, top_section, base_center, top_center, z_axes, angle_degrees

def create_sections(points, min_z, max_z):
    # Assuming `create_sections` is defined elsewhere and returns appropriate values
    # Here, just placeholder values are returned
    base_section = {"z_min": min_z, "points": points[points[:, 2] == min_z].tolist()}
    top_section = {"z_max": max_z, "points": points[points[:, 2] == max_z].tolist()}
    base_center = {"z": min_z, "coordinates": np.mean(points[points[:, 2] == min_z], axis=0).tolist()}
    top_center = {"z": max_z, "coordinates": np.mean(points[points[:, 2] == max_z], axis=0).tolist()}
    z_axes = {"start": base_center["coordinates"], "end": top_center["coordinates"]}
    angle_degrees = 0  # Placeholder for angle calculation

    return base_section, top_section, base_center, top_center, z_axes, angle_degrees

def bbox_center(bbox, normal, face_center):

    bbox_center = bbox.center
    bbox_extent = bbox.extent

    center_x, center_y, center_z = bbox_center
    extent_x, extent_y, extent_z = bbox_extent
    print(f"center_x: {center_x}, center_y: {center_y}, center_z: {center_z} extent_x: {extent_x}, extent_y: {extent_y}, extent_z: {extent_z}")

    # Compute length, width, and height
    length = max(extent_x, extent_y, extent_z)
    thickness = min(extent_x, extent_y, extent_z)
    print(f"Length of the bounding box: {length} Thickness of the bounding box: {thickness}")

    bbox_center_array = np.array(bbox_center)
    sign = np.sign(np.dot(normal, face_center - bbox_center_array))
    print(f"Sign: {sign}")

    return bbox_center_array, sign, length, thickness



# for index, wall_data in enumerate(walls_data):

#     walls_metadata = [] 

#     points = np.vstack([np.array(data['coordinates']) for data in walls_data])  
#     decimated_points = points[::10]
#     print(f'Decimated points count: {len(decimated_points)}, Shape: {decimated_points.shape}, Type: {type(decimated_points)}')

#     best_planes, best_inliers, normal = ransac_inliers_normals(decimated_points, distance_threshold=0.06, num_iterations=10)
    
#     if isinstance(best_inliers, np.ndarray) and best_inliers.ndim == 1:
#         inlier_points = decimated_points[best_inliers]
#         print(f"Inliers points count: {len(inlier_points)}, Shape of best_inliers: {best_inliers.shape}")

#         if inlier_points.ndim == 1:
#             if len(inlier_points) % 3 != 0:
#                 raise ValueError("The length of inlier_points is not divisible by 3, cannot reshape to (N, 3).")
#             inlier_points = inlier_points.reshape(-1, 3)
        
#         elif inlier_points.ndim != 2 or inlier_points.shape[1] != 3:
#             raise ValueError(f"Unexpected shape of inlier_points. Expected (N, 3) but got: {inlier_points.shape}")

#         face_center = adjust_face_center(inlier_points)

#         # bbox features
#         bbox = wall_data['oriented_bounding_box']
#         bbox_center_array, sign, length, thickness = bbox_center(bbox, normal, face_center)

#         wall_data['normal'] = normal * (-1 if sign == -1 else 1)

#         min_z,  max_z  = (np.min(decimated_points[:, 2]), np.max(decimated_points[:,2]))
#         base_section, top_section, base_center, top_center, z_axes, angle_degrees = create_sections(decimated_points, min_z, max_z)
#         print(f'Base Section: {base_section} Base Center: {base_center} Top Section: {top_section} Top Center: {top_center} The center of the top section {angle_degrees} degrees ')




## RAYTRACING FOR EXTERNAL WALLS
def bbox_details(bbox):
    # Extract vertices
    min_corner = bbox.MinPoint()
    max_corner = bbox.MaxPoint()

    vertices = [
        [min_corner.X(), min_corner.Y(), min_corner.Z()],
        [min_corner.X(), min_corner.Y(), max_corner.Z()],
        [min_corner.X(), max_corner.Y(), min_corner.Z()],
        [min_corner.X(), max_corner.Y(), max_corner.Z()],
        [max_corner.X(), min_corner.Y(), min_corner.Z()],
        [max_corner.X(), min_corner.Y(), max_corner.Z()],
        [max_corner.X(), max_corner.Y(), min_corner.Z()],
        [max_corner.X(), max_corner.Y(), max_corner.Z()],
    ]
    
    # Define edges as pairs of vertex indices
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Compute the midpoints of each face
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # Bottom face
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[0], vertices[2], vertices[6], vertices[4]],  # Left face
        [vertices[1], vertices[3], vertices[7], vertices[5]],  # Right face
    ]
    
    face_midpoints = []
    for face in faces:
        midpoint = [
            sum(vertex[0] for vertex in face) / 4,
            sum(vertex[1] for vertex in face) / 4,
            sum(vertex[2] for vertex in face) / 4
        ]
        face_midpoints.append(midpoint)
    
    return vertices, edges, face_midpoints



def ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    
    ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Ensure the direction is normalized
    
    denominator = np.dot(ray_direction, plane_normal)
    if abs(denominator) < 1e-6:
        # The ray is parallel to the plane
        return None
    
    t = np.dot(plane_point - ray_origin, plane_normal) / denominator
    
    if t < 0:
        # The intersection is behind the ray's origin
        return None
    
    intersection_point = ray_origin + t * ray_direction

    return intersection_point

def bbox_walls():
    # Define the walls of the bounding box
    min_corner = np.array([0, 0, 0])
    max_corner = np.array([1, 1, 1])
    
    # Define the planes for each wall (face) of the bounding box
    planes = [
        (min_corner, [1, 0, 0]),  # x = min_corner[0]
        (max_corner, [-1, 0, 0]), # x = max_corner[0]
        (min_corner, [0, 1, 0]),  # y = min_corner[1]
        (max_corner, [0, -1, 0]), # y = max_corner[1]
        (min_corner, [0, 0, 1]),  # z = min_corner[2]
        (max_corner, [0, 0, -1]), # z = max_corner[2]
    ]
    
    return planes

# def trace_rays_from_face_midpoints(bbox):
#     vertices, edges, face_midpoints = bbox_details(bbox)
#     planes = bbox_walls()
    
#     intersections = []
    
#     for midpoint in face_midpoints:
#         for plane_point, plane_normal in planes:
#             intersection = ray_plane_intersection(midpoint, plane_normal, plane_point, plane_normal)
#             if intersection is not None:
#                 # Ensure intersection is within the bounds of the bounding box
#                 if all(min_corner[i] <= intersection[i] <= max_corner[i] for i in range(3)):
#                     intersections.append(intersection)
    
#     return intersections

# Example usage:
class MockBBox:
    def MinPoint(self):
        return MockPoint(0, 0, 0)
    
    def MaxPoint(self):
        return MockPoint(1, 1, 1)

class MockPoint:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
    
    def X(self):
        return self._x
    
    def Y(self):
        return self._y
    
    def Z(self):
        return self._z

bbox = MockBBox()
# intersections = trace_rays_from_face_midpoints(bbox)
# print("Intersections:", intersections)



# Entire pcd
def ray_tracing(file_name):

    laz = laspy.read(file_name)
    pcd = gmu.las_to_pcd(laz)
    bbox = gmu.get_oriented_bounding_box(pcd)

    vertices, edges, mid_face_points = bbox_details(bbox)

    
def intersect_line_2d(p0, p1, q0, q1,strict=True):

    # Direction vectors of the lines
    dp = p1 - p0
    dq = q1 - q0
    
    # Matrix and vector for the linear system
    A = np.vstack((dp, -dq)).T
    b = q0 - p0
    
    # Solve the linear system
    try:
        t, u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # The system is singular: lines are parallel or identical
        return None
    
    # Intersection point
    intersection = p0 + t * dp
    
    if strict:
    # Since the system has a solution, check if it lies within the line segments
        if np.allclose(intersection, q0 + u * dq):
            return intersection
        else:
            return None
    else:
        return intersection


def compute_normal(start_point, end_point):

    direction = np.array(end_point) - np.array(start_point)
    normal = np.array([-direction[1], direction[0], 0])

    normalized_normal = normal / np.linalg.norm(normal)
    
    return normal, normalized_normal



# def extract_z_levels(floor_bboxes, ceiling_bboxes):
#     # Filter out None values
#     filtered_floor_bboxes = [bbox for bbox in floor_bboxes if bbox is not None]
#     filtered_ceiling_bboxes = [bbox for bbox in ceiling_bboxes if bbox is not None]

#     # Convert lists to numpy arrays
#     floor_bboxes_array = np.array(filtered_floor_bboxes, dtype=object)
#     ceiling_bboxes_array = np.array(filtered_ceiling_bboxes, dtype=object)

#     # Extract z_min and z_max values from each bounding box
#     z_min_values = [bbox[0, 2] for bbox in floor_bboxes_array] + [bbox[0, 2] for bbox in ceiling_bboxes_array]
#     z_max_values = [bbox[0, 2] for bbox in floor_bboxes_array] + [bbox[0, 2] for bbox in ceiling_bboxes_array]

#     return z_min_values, z_max_values

# def find_closest_level(z_value, z_values, z_threshold):

#     for i, z in enumerate(z_values):
#         if isinstance(z, tuple):
#             z = z[0] 
#         if abs(z_value - z) < z_threshold:
#             return i, z
#     return None, None

# z_threshold = 0.25

# def associate_levels_and_walls(walls_data, floor_bboxes, ceiling_bboxes, laz):
  
#     z_min_values, z_max_values = extract_z_levels(floor_bboxes, ceiling_bboxes)

#     # Process each wall
#     for wall_data in walls_data:
#         wall_metadata = process_wall_data(laz, wall_data)
        
#         if wall_metadata is None:
#             continue

#         start_point_base = np.array(wall_metadata['start_point_base'])
#         end_point_base = np.array(wall_metadata['end_point_base'])
#         length = np.linalg.norm(end_point_base - start_point_base)

#         z_min_wall = float(wall_metadata['min_z'])
#         z_max_wall = float(wall_metadata['max_z'])
#         height_wall = float(wall_metadata['height'])
        
#         print(f'Wall {wall_metadata["object_id"]} length {length} height: {height_wall}')

#         # Find closest level for z_min_wall
#         min_level_idx, min_z_level = find_closest_level(z_min_wall, z_min_values, z_threshold)

#         if min_level_idx is not None:
#             print(f'Wall {wall_metadata["object_id"]} is close to floor level {min_level_idx} with z_min {min_z_level}')

#         # Find closest level for z_max_wall
#         max_level_idx, max_z_level = find_closest_level(z_max_wall, z_max_values, z_threshold)
#         if max_level_idx is not None:
#             print(f'Wall {wall_metadata["object_id"]} is close to ceiling level {max_level_idx} with z_max {max_z_level}')








## COLUMNS___________________
def load_columns_data(laz, columns_nodes, avg_z, normals):

    columns_points = {}
    columns_points_2d = {}

    for node in columns_nodes:

        idx = np.where((laz['classes'] == node.class_id) & (laz['objects'] == node.object_id))
        
        if len(idx[0]) > 0:

            columns_points[node.object_id] = np.vstack((laz.x[idx], laz.y[idx], laz.z[idx], np.asarray(normals)[idx, 0], np.asarray(normals)[idx, 1], np.asarray(normals)[idx, 2])).transpose() 

            # Enable this to filter the points at min - max height
            # z_values = columns_points[node.object_id][:, 2]
            # min_z = np.min(z_values)
            # max_z = np.max(z_values)
            
            # idx = np.where((laz['classes'] == node.class_id) & (laz['objects'] == node.object_id) & (laz.z > min_z + 0.1) & (laz.z < max_z - 0.1))

            # Place points at avg_z
            columns_points_2d[node.object_id] = np.vstack((laz.x[idx], laz.y[idx], np.full_like(laz.z[idx], avg_z), np.asarray(normals)[idx, 0], np.asarray(normals)[idx, 1], np.asarray(normals)[idx, 2])).transpose() 
        
    return columns_points, columns_points_2d






def extract_objects_building_(laz, graph_path, pcd):
    # Parse the RDF graph and convert nodes
    graph = Graph().parse(str(graph_path))
    nodes = tl.graph_to_nodes(graph)

    # Categorize nodes by type (floors, ceilings, walls, etc.)
    node_categories = {
        'unassigned': [n for n in nodes if 'unassigned' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'floors': [n for n in nodes if 'floors' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'ceilings': [n for n in nodes if 'ceilings' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'walls': [n for n in nodes if 'walls' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'columns': [n for n in nodes if 'columns' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'doors': [n for n in nodes if 'doors' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'windows': [n for n in nodes if 'windows' in n.subject.lower() and isinstance(n, PointCloudNode)],
        'levels': [n for n in nodes if 'levels' in n.subject.lower() and isinstance(n, PointCloudNode)]
    }

    # Helper function to extract information for further processes
    def extract_info(node_list):
        data_list = []
        for n in node_list:
            if not hasattr(n, 'resource') or 'points' not in getattr(n, 'resource', {}):
                print(f"Error: Missing 'resource' or 'points' in node {n.__dict__}")
                continue

            idx = np.where((laz['classes'] == n.class_id) & (laz['objects'] == n.object_id))

            # Extract coordinates based on indices
            x_coords = laz['x'][idx]
            y_coords = laz['y'][idx]
            z_coords = laz['z'][idx]
            
            # Stack coordinates vertically
            coordinates = np.vstack((x_coords, y_coords, z_coords)).T

            # Set point cloud resource and calculate oriented bounding box (OBB)
            n.resource = pcd
            n.get_oriented_bounding_box()
            
            # Collect useful data such as indices, OBB, and color
            data = {
                'indices': idx,
                'oriented_bounding_box': n.orientedBoundingBox,
                'coordinates': coordinates,
                'obb_color': n.orientedBoundingBox.color,
                'obb_center': n.orientedBoundingBox.center,
                'obb_extent': n.orientedBoundingBox.extent,
                'class_id': n.class_id,
                'object_id': n.object_id
            }
            data_list.append(data)
        return data_list

    # Extract information for each category
    extracted_data = {}
    for category, node_list in node_categories.items():
        extracted_data[category] = extract_info(node_list)

    return extracted_data






## ___________OBJ -- MESHES ____________
def load_obj_and_create_meshes(file_path: str) -> Dict[str, o3d.geometry.TriangleMesh]:
    """
    Loads an OBJ file and creates TriangleMeshes for each object group.

    Args:
        file_path (str): Path to the OBJ file.

    Returns:
        Dict[str, o3d.geometry.TriangleMesh]: A dictionary mapping object group names to their corresponding TriangleMeshes.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vertices = []
    faces = {}
    current_object = None

    for line in lines:
        if line.startswith('v '):
            parts = line.strip().split()
            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
            vertices.append(vertex)
        elif line.startswith('f '):
            if current_object is not None:
                parts = line.strip().split()
                face = [int(parts[1].split('/')[0]) - 1, int(parts[2].split('/')[0]) - 1, int(parts[3].split('/')[0]) - 1]
                faces[current_object].append(face)
        elif line.startswith('g '):
            current_object = line.strip().split()[1]
            if current_object not in faces:
                faces[current_object] = []

    meshes = {}
    for object_name, object_faces in faces.items():
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(object_faces)
        mesh.compute_vertex_normals()
        meshes[object_name] = mesh
    
    return meshes



#### _________________ENERGY PART_______________________
from typing import Dict, Any

def read_building_energy_system(file_path: str) -> Dict[str, Any]:

    # Initialize a dictionary to store information for each class
    data_classes = {
        "heat_pump": None,
        "radiators": None,
        "lighting_system": None,
        "hvac_system": None,
        "solar_panels": None,
        "renewable_energy_sources": None
    }
    
    try:
        # Read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Debug print the whole JSON data
        print("Loaded JSON data:", json.dumps(data, indent=4))

        # Check and store data for each class if available
        if 'building_energy_system' in data:
            building_energy_system = data['building_energy_system']

            # Debug print the building energy system data
            print("Building Energy System Data:", json.dumps(building_energy_system, indent=4))

            # Heat Pump
            if 'heat_pump' in building_energy_system:
                data_classes['heat_pump'] = building_energy_system['heat_pump']

            # Radiators
            if 'radiators' in building_energy_system:
                data_classes['radiators'] = building_energy_system['radiators']

            # Lighting System
            if 'lighting_system' in building_energy_system:
                data_classes['lighting_system'] = building_energy_system['lighting_system']

            # HVAC System
            if 'hvac_system' in building_energy_system:
                data_classes['hvac_system'] = building_energy_system['hvac_system']

            # Solar Panels
            if 'solar_panels' in building_energy_system:
                data_classes['solar_panels'] = building_energy_system['solar_panels']

            # Renewable Energy Sources
            if 'renewable_energy_sources' in building_energy_system:
                data_classes['renewable_energy_sources'] = building_energy_system['renewable_energy_sources']
                
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the JSON file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data_classes

#__________________MESH________________________________

def column_mesh(points, floor_z, ceiling_z, height, minimum_bounding_box, depth, width, output_folder, name):  
    base_edges = []  
    top_edges = []
    vertical_edges = []
    faces = []

    print ("Vertex obj base", floor_z)
    print ("Vertex obj end", ceiling_z)

    base_vertices = [
                [minimum_bounding_box[0][0], minimum_bounding_box[0][1], floor_z], 
                [minimum_bounding_box[1][0], minimum_bounding_box[1][1], floor_z], 
                [minimum_bounding_box[2][0], minimum_bounding_box[2][1], floor_z],
                [minimum_bounding_box[3][0], minimum_bounding_box[3][1], floor_z]
]
    print ('Base vertices: ', base_vertices)

    end_vertices = [
                    [minimum_bounding_box[0][0], minimum_bounding_box[0][1], ceiling_z], 
                    [minimum_bounding_box[1][0], minimum_bounding_box[1][1], ceiling_z], 
                    [minimum_bounding_box[2][0], minimum_bounding_box[2][1], ceiling_z],
                    [minimum_bounding_box[3][0], minimum_bounding_box[3][1], ceiling_z]
]

    base_vertices = np.array(base_vertices)
    print(base_vertices.shape)
    end_vertices = np.array(end_vertices)
    vertices = np.vstack ((base_vertices, end_vertices))

    A0 = np.array([minimum_bounding_box[0][0], minimum_bounding_box[0][1], floor_z]) 
    B0 = np.array([minimum_bounding_box[1][0], minimum_bounding_box[1][1], floor_z])
    C0 = np.array([minimum_bounding_box[2][0], minimum_bounding_box[2][1], floor_z])
    D0 = np.array([minimum_bounding_box[3][0], minimum_bounding_box[3][1], floor_z])

    A1 = np.array([minimum_bounding_box[0][0], minimum_bounding_box[0][1], ceiling_z]) 
    B1 = np.array([minimum_bounding_box[1][0], minimum_bounding_box[1][1], ceiling_z])
    C1 = np.array([minimum_bounding_box[2][0], minimum_bounding_box[2][1], ceiling_z])
    D1 = np.array([minimum_bounding_box[3][0], minimum_bounding_box[3][1], ceiling_z])

    base_edges= np.array([[A0, B0], [B0, C0], [C0, D0], [D0, A0]])
    top_edges = np.array([[A1, B1], [B1, C1], [C1, D1], [D1, A1]])
    vertical_edges = np.array([[A0, A1], [B0, B1], [C0, C1], [D0, D1]])
    edges = np.vstack((base_edges, top_edges, vertical_edges))

    # Compute faces
    face_a = np.array([A0, B1, A1])
    face_b = np.array([A0, B0, B1])
    face_c = np.array([B0, B1, C0])
    face_d = np.array([C0, C1, B1])
    face_e = np.array([C0, C1, D0])
    face_f = np.array([C1, D1, D0])
    face_g = np.array([A0, D0, D1])
    face_h = np.array([A0, A1, D1])

    # Faces
    faces = np.array((face_a, face_b, face_c, face_d, face_e, face_f, face_g, face_h))
    print ('Faces:', faces)
    print ('Faces:', faces.shape)
    
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    color_edges = 'red'
    lw_edges = 0.25
    markersize_vertex = 2
    color_points = 'red'
    markersize_points = 0.001
    points_column = 'blue'
    
    # Plot base vertices
    points = np.concatenate([points], axis=0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker = 'o', color = points_column, s = markersize_points, alpha = 0.90)

    # for vertices in base_vertices:
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.plot(x, y, z, marker='o', color = color_points, markersize = markersize_vertex)
   
    # Flatten the edges array
    x_edges = edges[:, :, 0].flatten()
    y_edges = edges[:, :, 1].flatten()
    z_edges = edges[:, :, 2].flatten()

    # Plot edges as scatter
    ax.scatter(x_edges, y_edges, z_edges, color= color_edges, lw = lw_edges)

    for face in faces:
        # Close the loop by repeating the first vertex
        face = np.append(face, [face[0]], axis=0)
        # Plot the face
        ax.plot(face[:, 0], face[:, 1], face[:, 2])

    # Set labels
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')
    plt.gca().set_aspect('equal', adjustable='box')

    # Show plot
    #plt.show()

    # Prepare file for obj file
    A0 = np.array(base_vertices[0]) #1
    B0 = base_vertices[1] #2
    C0 = base_vertices[2] #3
    D0 = base_vertices[3] #4
    A1 = end_vertices[0] #5
    B1 = end_vertices[1] #6
    C1 = end_vertices[2] #7
    D1 = end_vertices[3] #8
    print ('A:', A0)
    print (f'A0, {A0[0]} {A0[1]} {A0[2]}')

    faces_obj = [
        [3, 7, 8],
        [3, 8, 4],
        [1, 5, 6],
        [1, 6, 2],
        [7, 3, 2],
        [7, 2, 6],
        [4, 8, 5],
        [4, 5, 1],
        [8, 7, 6],
        [8, 6, 5],
        [3, 4, 1],
        [3, 1, 2]
    ]

    file_name = output_folder / f"{name}.obj"

    ratio = width / depth  

    if (depth < 0.25 or width < 0.25) or (depth > 1.0 and width > 1.0) or (ratio > 2) or (ratio < 0.5):
        return

    # Write the file if condition is met
    with open(file_name, "w") as f:
        for v in vertices:
            f.write(f'v {v[0]:.3f} {v[1]} {v[2]}\n')
        # Write faces
        for face in faces_obj:
            # Convert face vertices to strings without brackets
            face_str = ' '.join([str(v) for v in face])
            f.write(f'f {face_str}\n')
    print("Obj correctly generated!")


  


#__________________GMSH FEM _____________________________
# floor_bboxes = np.array((results['floor_bboxes']))
# ceiling_bboxes = np.array((results['ceiling_bboxes']))

# floor_bboxes = np.array(floor_bboxes) 


# def create_gmsh_mesh(floor_bboxes):
#     import gmsh
#     import numpy as np

#     # Initialize the GMSH API
#     gmsh.initialize()
#     gmsh.model.add("bounding_box")

#     # Define your bounding box levels
#     floor_bboxes = np.array(floor_bboxes)       

#     # Initialize lists
#     point_tags = []
#     line_tags = []
#     surface_tags = []
    
#     # Create points for each level
#     for level in floor_bboxes:
#         level_tags = []
#         for pt in level:
#             pt_tag = gmsh.model.geo.addPoint(pt[0], pt[1], pt[2])
#             level_tags.append(pt_tag)
#         point_tags.append(level_tags)
    
#     # Create lines for each level
#     for level_tags in point_tags:
#         level_lines = [
#             gmsh.model.geo.addLine(level_tags[0], level_tags[1]),
#             gmsh.model.geo.addLine(level_tags[1], level_tags[2]),
#             gmsh.model.geo.addLine(level_tags[2], level_tags[3]),
#             gmsh.model.geo.addLine(level_tags[3], level_tags[0]),
#             gmsh.model.geo.addLine(level_tags[4], level_tags[5]),
#             gmsh.model.geo.addLine(level_tags[5], level_tags[6]),
#             gmsh.model.geo.addLine(level_tags[6], level_tags[7]),
#             gmsh.model.geo.addLine(level_tags[7], level_tags[4]),
#             gmsh.model.geo.addLine(level_tags[0], level_tags[4]),
#             gmsh.model.geo.addLine(level_tags[1], level_tags[5]),
#             gmsh.model.geo.addLine(level_tags[2], level_tags[6]),
#             gmsh.model.geo.addLine(level_tags[3], level_tags[7])
#         ]
#         line_tags.append(level_lines)
    
#     # Debug: Print line tags to verify correctness
#     print("Line tags:", line_tags)
    
#     # Create surfaces for each level
#     for level_lines in line_tags:
#         try:
#             # Bottom face
#             bottom_face = gmsh.model.geo.addSurfaceFilling([level_lines[0], level_lines[1], level_lines[2], level_lines[3]])
#             # Top face
#             top_face = gmsh.model.geo.addSurfaceFilling([level_lines[4], level_lines[5], level_lines[6], level_lines[7]])
            
#             surfaces = [bottom_face, top_face]
#             surface_tags.append(surfaces)
#         except Exception as e:
#             print("Error creating surfaces:", e)
    
#     # Define side surfaces (connect corresponding edges of consecutive levels)
#     for i in range(len(surface_tags) - 1):
#         level1_lines = line_tags[i]
#         level2_lines = line_tags[i + 1]
#         for j in range(4):  # Each level has 4 side surfaces
#             try:
#                 gmsh.model.geo.addSurfaceFilling([
#                     level1_lines[j + 4],  # Vertical lines from level 1
#                     level1_lines[j + 5],  # Vertical
#                     level2_lines[j + 6],  # Vertical lines from level 2
#                     level2_lines[j + 7]   # Vertical lines from level 2
#                 ])
#             except Exception as e:
#                 print(f"Error creating side surface for level {i}: {e}")

#     # Define the volume
#     try:
#         surface_loop = gmsh.model.geo.addSurfaceLoop([s for sublist in surface_tags for s in sublist])
#         gmsh.model.geo.addVolume(surface_loop)
#     except Exception as e:
#         print("Error creating volume:", e)
    
#     # Synchronize and generate the mesh
#     gmsh.model.geo.synchronize()
#     gmsh.model.mesh.generate(3)
    
#     # Write the mesh to a file
#     gmsh.write("bounding_box.msh")
    
#     # Finalize the GMSH API
#     gmsh.finalize()

# if __name__ == "__main__":

#     create_gmsh_mesh(floor_bboxes)
#     print("Mesh created and saved successfully!")
