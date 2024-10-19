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

import geomapi.utils as ut
import geomapi.tools as tl
from geomapi.utils import geometryutils as gmu
from geomapi.nodes import PointCloudNode

from scipy.spatial import KDTree, ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors

import topologicpy as tp

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

def create_class_object_to_idx_mapping(class_object_ids):
    
    return {pair: idx for idx, pair in enumerate(class_object_ids)}

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
    lights_nodes = [n for n in nodes if 'lights' in n.subject.lower() and isinstance(n, PointCloudNode)]
    radiators_nodes = [n for n in nodes if 'radiators' in n.subject.lower() and isinstance(n, PointCloudNode)]
    hvac_nodes = [n for n in nodes if 'hvac' in n.subject.lower() and isinstance(n, PointCloudNode)]
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

    lights = extract_info(lights_nodes)
    print(len(lights), 'lights detected')

    radiators = extract_info(radiators_nodes)
    print(len(radiators), 'radiators detected')

    hvac = extract_info(hvac_nodes)
    print(len(hvac), 'hvac detected')
    
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
        'lights': lights,
        'radiators': radiators,
        'hvac': hvac,
        'levels': levels
    }


def extract_info_elements(node_list, laz):
    data_list = []
    for n in node_list:
        # Find indices for class_id and object_id in the point cloud data (laz)
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


#@___________________________MAIN GEOMETRIC OPERATIONS___________________________

def compute_bounding_box(points):
    
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2),np.cos (angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # Covariance matrix
    cov = np.cov(points, rowvar=False)
    
    # apply rotations to the Hull points
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, r

def rotate_points_2d(points, center, rotation_matrix):

        rotated_points = []

        for point in points: 

            shifted = point - center   
            rotated = np.dot(shifted, rotation_matrix)
            rotated_points.append(rotated + center)
            
        rotated_points = np.array(rotated_points)

        return rotated_points

def rotate_points_3d(points, center, rotation_matrix):
    # Ensure center is 3D by adding a z-component of 0 if necessary
    if center.shape[0] == 2:
        center = np.append(center, 0)
    
    rotated_points = []

    for point in points:
        # Ensure point is also 3D
        if point.shape[0] == 2:
            point = np.append(point, 0)

        # Shift the point by the center, rotate, and shift back
        shifted = point - center
        rotated = np.dot(shifted, rotation_matrix)
        rotated_points.append(rotated + center)

    return np.array(rotated_points)

def compute_best_coverage_segments(rotated_points, histogram_step, coordinate, key):

    cases_h = np.arange(min([p[coordinate] for p in rotated_points]), max([p[coordinate] for p in rotated_points]), histogram_step)

    case_b = cases_h [: len(cases_h) // 2]
    hist_b, case_b = np.histogram([p[coordinate] for p in rotated_points], bins=case_b)

    if len(hist_b) == 0:
        print(f'Column: {key} - could not compute features')
    else:
        (f'Column: {key} - could not compute features\n')
        return None, False
       
    case_d = cases_h [2*len(cases_h) // 2:]
    hist_d, case_d =  np.histogram([p[coordinate] for p in rotated_points], bins=case_d)

    if len(hist_d) == 0:
        print(f'Column: {key} - could not compute features')
    else:
        print(f'Column: {key} - could not compute features\n')
        return None, False

    max_index_case_b = np.argmax(hist_b)
    max_index_case_d = np.argmax(hist_d)

    print("case_b", case_b)
    print("case_d", case_d)
    coord_b = case_b[max_index_case_b] + histogram_step / 2
    coord_d = case_d[max_index_case_d] + histogram_step / 2

    print("Coordinate of minimum value in case_b along {coordinate} axis:", coord_b)
    print("Coordinate of maximum value in case_d along {coordinate} axis:", coord_d)

    # New bbox
    new_coordinate = (coord_b, coord_b, coord_d, coord_d)
    return new_coordinate, True

def json_export(output_folder, name, key, width, depth, height_column, center, rotation):

    #json_file_path = os.path.join(output_folder, f'{ut.get_filename(name)}_columns.json')
    json_file_path = os.path.join(output_folder, '_'.join(ut.get_filename(name).split('_')[:4]) + '_columns.json')

    # Check if the file exists and read its content
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            json_file_data = json.load(json_file)
    else:
        json_file_data = []

    # width = np.max([width, 0.10])
    # depth = np.max([depth, 0.10])
    ratio = width / depth    

    if (depth < 0.25 or width < 0.25) or (depth > 1.0 and width > 1.0) or (ratio > 2) or (ratio < 0.5):
        return

    # Construct the new object data
    obj = {
        "id": key,
        "width": width,
        "depth": depth,
        "height": height_column,
        "loc": [center[0], center[1], center[2]],
        "rotation": 0 # 0 for cvpr, rotation_rounded otherwise
    }

    # Append the new object to the list of objects
    json_file_data.append(obj)

    # Write the updated JSON data to file
    with open(json_file_path, "w") as json_file:
        json.dump(json_file_data, json_file, indent=4)

    print("JSON data written to file:", json_file_path)

    return obj

def object_mesh(points, floor_z, ceiling_z, height, minimum_bounding_box, depth, width, output_folder, name):  
    
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

def extract_min_max_z_from_bboxes(bboxes):
    
    z_values = []
    
    # Loop through each bounding box and extract z values
    for bbox in bboxes:
        bbox_z_values = [point[2] for point in bbox]  # Extract z-coordinates (3rd element of each point)
        z_values.extend(bbox_z_values)  # Add all z-values to the list
    
    if z_values:
        z_min = np.min(z_values)
        z_max = np.max(z_values)
    else:
        z_min = None
        z_max = None

    return z_min, z_max

def compute_3d_features(laz, wall_coordinates, rotated_boxes, rotation_matrix, graph_path, floor_bboxes, ceiling_bboxes):
    
    all_base_vertices = []  # Store base vertices for all walls
    all_top_vertices = []  # Store top vertices for all walls
    wall_properties = []  # Store other wall properties like length, thickness, etc.

    # Loop through each set of wall points
    for i, wall_points in enumerate(wall_coordinates):
    
        coordinates = wall_points[:, :3]
        print(f"Coordinates for Wall {i}:\n", coordinates)
    
        z_values = coordinates[:, 2]
        min_z = np.min(z_values)
        max_z = np.max(z_values)

        # Get the rotated box for the current wall (from rotated_boxes list)
        rotated_box = rotated_boxes[i]

        # Ensure rotated_box is 2D (x, y)
        if rotated_box.shape[1] != 2:
            raise ValueError(f"rotated_box for Wall {i} does not have the correct shape: {rotated_box.shape}")

        # Compute the center as the mean of the rotated box
        center_r = np.mean(rotated_box, axis=0)

        # Ensure the center is 2D and append z=0 for 3D
        center = np.array([center_r[0], center_r[1], 0])

        # Vertices of the bounding box
        A, B, C, D = rotated_box

        # Calculate edges of the bounding box
        a = B - A
        b = C - B
        c = D - C
        d = A - D

        # Stack the edges into an array
        edges = np.vstack([a, b, c, d])

        # Compute rotation angle
        angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        rotation = np.degrees(angle)

        # Dimensions 
        thickness = np.min(np.linalg.norm(edges, axis=1))  # Fix: Compute norm over axis 1 (for each edge)
        length = np.max(np.linalg.norm(edges, axis=1))

        # Extract z_min and z_max from both floor_bboxes and ceiling_bboxes
        floor_z_min, floor_z_max = extract_min_max_z_from_bboxes(floor_bboxes)  # Floor bounding boxes
        ceiling_z_min, ceiling_z_max = extract_min_max_z_from_bboxes(ceiling_bboxes)  # Ceiling bounding boxes

        # Determine z_base from floors and z_top from ceilings
        if floor_z_min is not None:
            z_base = floor_z_min
        else:
            z_base = min_z  # Fallback to min_z if no floor data

        if ceiling_z_max is not None:
            z_top = ceiling_z_max
        else:
            z_top = max_z  # Fallback to max_z if no ceiling data

        # Calculate the height of the wall
        height = z_top - z_base

        # Redefine the bounding box in 3D
        A_3d_base = np.append(A, z_base)
        B_3d_base = np.append(B, z_base)
        C_3d_base = np.append(C, z_base)
        D_3d_base = np.append(D, z_base)

        A_3d_top = np.append(A, z_top)
        B_3d_top = np.append(B, z_top)
        C_3d_top = np.append(C, z_top)
        D_3d_top = np.append(D, z_top)

        # Store the 3D bounding box vertices (base and top) for this wall
        base_vertices = np.array([A_3d_base, B_3d_base, C_3d_base, D_3d_base])
        top_vertices = np.array([A_3d_top, B_3d_top, C_3d_top, D_3d_top])

        all_base_vertices.append(base_vertices)  # Save base vertices for this wall
        all_top_vertices.append(top_vertices)    # Save top vertices for this wall

        # Save other wall properties
        wall_properties.append({
            'length': length,
            'thickness': thickness,
            'height': height,
            'center': center,
            'rotation': rotation
        })

        # Print out details of the wall for debugging
        print(f'Wall {i}: Thickness: {thickness}\n Length: {length}\n z_base: {z_base}, z_top: {z_top}\n Height: {height}\n Location: {center}\n Rotation: {rotation}')

    return all_base_vertices, all_top_vertices, wall_properties

def bounding_boxes(laz, wall_coordinates, walls_nodes):

    wall_coordinates = []
    wall_oriented_bboxes = []

    for wall in walls_nodes:
        
        idx = np.where((laz.classes == wall.class_id) & (laz.objects == wall.object_id))

        if idx is not None and len(idx[0]) > 0:  # Ensure valid and non-empty indices
            coord_x = laz.x[idx]
            coord_y = laz.y[idx]
            coord_z = laz.z[idx]

            coordinates_ = np.vstack((coord_x, coord_y, coord_z)).T
            wall_coordinates.append(coordinates_)

            # 1. Create an oriented bounding box for the wall
            point_cloud = o3d.geometry.PointCloud()  # Assuming open3d for point cloud manipulation
            point_cloud.points = o3d.utility.Vector3dVector(coordinates_)

            oriented_bounding_box = point_cloud.get_oriented_bounding_box()
            wall_oriented_bboxes.append(oriented_bounding_box)

            # 2. Compute the dominant plane using RANSAC
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.03,
                                                                ransac_n=3,
                                                                num_iterations=1000)

            # 3. Compute the 2D normal of the wall (normal vector from plane equation)
            normal = plane_model[:3]  # Extracting the normal vector from the plane model
            normal[2] = 0  # Set the z-component to zero for 2D normal
            normal /= np.linalg.norm(normal)  # Normalize the vector

            # 4. Determine the direction of the wall
            box_center = np.asarray(oriented_bounding_box.get_center())
            face_center = point_cloud.select_by_index(inliers).get_center()
            face_center[2] = box_center[2]  # Align face center's height with the bounding box center

            sign = np.sign(np.dot(normal, face_center - box_center))  # Use dot product to determine direction

            # If sign is negative, flip the normal
            normal *= -1 if sign == -1 else 1

            # Output the result for debugging
            print(f'Wall {wall.object_id}:')
            print(f' - Oriented Bounding Box Center: {box_center}')
            print(f' - Plane Model: {plane_model}')
            print(f' - 2D Normal: {normal}')
            print(f' - Wall Direction: {"outwards" if sign == 1 else "inwards"}')

    return wall_coordinates, wall_oriented_bboxes, walls_nodes


# @ _________________LEVELS___________________@#

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

def select_nodes_intersecting_bounding_box(node: dict, nodelist: List[dict], u: float = 0.5, v: float = 0.5, w: float = 0.5) -> List[dict]:

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

def compute_plane_and_normal(points, distance_threshold = 0.03, num_iterations = 1000):
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




##____________________GMSH MESH_______________________________

# Function to create 3D cuboid 
def create_cuboid(points, color):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    # Define faces using two triangles for each rectangular face
    faces = [
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [1, 2, 6], [1, 6, 5],  # Right face
        [2, 3, 7], [2, 7, 6],  # Back face
        [3, 0, 4], [3, 4, 7]   # Left face
    ]
    
    cuboid = go.Mesh3d(
        x=x, y=y, z=z,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color=color,
        opacity=0.5
    )
    return cuboid

