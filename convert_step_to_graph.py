# Final, Corrected Script for B-rep to Graph Conversion
# Requires pythonocc-core, networkx, numpy, tqdm

import sys
import pickle
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm

# --- pythonocc-core Imports ---
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCC.Core.TopoDS import topods, TopoDS_Face, TopoDS_Edge
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Dir
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE

class BRepGraphExtractor:
    def __init__(self, step_file_path, n_samples_proximity=64):
        self.step_file_path = step_file_path
        self.n_samples_proximity = n_samples_proximity

        self.shape = self._load_step_file()
        if not self.shape or self.shape.IsNull():
            raise ValueError(f"Could not load or process STEP file: {step_file_path}")

        self.topo_explorer = TopologyExplorer(self.shape, ignore_orientation=True)
        self.faces = list(self.topo_explorer.faces())
        self.edges = list(self.topo_explorer.edges())
        self._build_topology_maps()

    def _load_step_file(self):
        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_file_path)
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            return reader.OneShape()
        return None

    def _build_topology_maps(self):
        # Use Python's built-in hash instead of .HashCode()
        self.face_map = {hash(f): i for i, f in enumerate(self.faces)}
        self.edge_map = {hash(e): i for i, e in enumerate(self.edges)}
        self.edge_to_faces = {i: [] for i in range(len(self.edges))}

        for face_idx, face in enumerate(self.faces):
            for edge in self.topo_explorer.edges_from_face(face):
                edge_hash = hash(edge)
                if edge_hash in self.edge_map:
                    edge_idx = self.edge_map[edge_hash]
                    if face_idx not in self.edge_to_faces[edge_idx]:
                        self.edge_to_faces[edge_idx].append(face_idx)

    def _get_random_points_on_face(self, face_idx, num_points):
        face = self.faces[face_idx]
        surface = BRep_Tool.Surface(face)
        u_min, u_max, v_min, v_max = breptools_UVBounds(face)

        points = []
        normals = []

        if not all(np.isfinite([u_min, u_max, v_min, v_max])):
            return np.zeros((num_points, 3)), np.zeros((num_points, 3))

        classifier = BRepTopAdaptor_FClass2d(face, 1e-9)
        props = GeomLProp_SLProps(surface, 1, 1e-9)

        attempts = 0
        max_attempts = num_points * 100

        while len(points) < num_points and attempts < max_attempts:
            u = np.random.uniform(u_min, u_max)
            v = np.random.uniform(v_min, v_max)
            if classifier.Perform(gp_Pnt2d(u, v)) == TopAbs_IN:
                pnt = surface.Value(u, v)
                props.SetParameters(u, v)
                if props.IsNormalDefined():
                    normal = props.Normal()
                    if face.Orientation() == TopAbs_REVERSED:
                        normal.Reverse()
                    points.append(pnt.Coord())
                    normals.append(normal.Coord())
            attempts += 1

        while len(points) < num_points:
            points.append((0, 0, 0))
            normals.append((0, 0, 0))

        return np.array(points), np.array(normals)

    def _get_edge_convexity_and_angle(self, edge_idx):
        face_indices = self.edge_to_faces[edge_idx]
        if len(face_indices) != 2:
            return 2, np.pi

        face1 = self.faces[face_indices[0]]
        face2 = self.faces[face_indices[1]]
        edge = self.edges[edge_idx]

        curve_adaptor = BRepAdaptor_Curve(edge)
        mid_param = (curve_adaptor.FirstParameter() + curve_adaptor.LastParameter()) / 2.0
        pnt_on_edge = curve_adaptor.Value(mid_param)

        normals = []
        for face in [face1, face2]:
            surf = BRep_Tool.Surface(face)
            projector = GeomAPI_ProjectPointOnSurf(pnt_on_edge, surf)
            if projector.NbPoints() > 0:
                u, v = projector.LowerDistanceParameters()
                props = GeomLProp_SLProps(surf, u, v, 1, 1e-9)
                if props.IsNormalDefined():
                    normal = props.Normal()
                    if face.Orientation() == TopAbs_REVERSED:
                        normal.Reverse()
                    normals.append(gp_Dir(normal.X(), normal.Y(), normal.Z()))

        if len(normals) == 2:
            angle = normals[0].Angle(normals[1])
            if np.isclose(angle, np.pi, atol=1e-2):
                return 2, angle
            elif angle < np.pi:
                return 0, angle
            else:
                return 1, angle

        return 2, np.pi

    def extract_features(self):
        num_faces = len(self.faces)
        num_edges = len(self.edges)

        print("Step 1/4: Building Face Adjacency Graph (FAG)...")
        fag = nx.Graph()
        fag.add_nodes_from(range(num_faces))
        for edge_idx, face_indices in self.edge_to_faces.items():
            if len(face_indices) == 2:
                fag.add_edge(face_indices[0], face_indices[1])

        print("Step 2/4: Extracting Face (Node) and Edge Features...")
        face_features = {}
        face_types = np.zeros(num_faces, dtype=int)
        face_areas = np.zeros(num_faces)
        face_loop_counts = np.zeros(num_faces, dtype=int)
        face_centroids = np.zeros((num_faces, 3))
        face_convexity = np.zeros(num_faces, dtype=int)

        gprop = GProp_GProps()
        for i, face in enumerate(tqdm(self.faces, desc="Processing Faces")):
            adaptor = BRepAdaptor_Surface(face)
            face_types[i] = adaptor.GetType()
            brepgprop_SurfaceProperties(face, gprop)
            face_areas[i] = gprop.Mass()
            convex_edges = 0
            concave_edges = 0

            for edge in self.topo_explorer.edges_from_face(face):
                edge_hash = hash(edge)
                if edge_hash in self.edge_map:
                    edge_idx = self.edge_map[edge_hash]
                    conv, _ = self._get_edge_convexity_and_angle(edge_idx)
                    if conv == 0:
                        convex_edges += 1
                    elif conv == 1:
                        concave_edges += 1

            if convex_edges > 0 and concave_edges == 0:
                face_convexity[i] = 0
            elif concave_edges > 0 and convex_edges == 0:
                face_convexity[i] = 1
            else:
                face_convexity[i] = 2  # mixed or undefined

            centroid_pnt = gprop.CentreOfMass()
            face_centroids[i] = centroid_pnt.Coord()
            exp = TopExp_Explorer(face, TopAbs_WIRE)
            loop_count = 0
            while exp.More():
                loop_count += 1
                exp.Next()
            face_loop_counts[i] = loop_count

        face_features['type'] = face_types
        face_features['area'] = face_areas
        face_features['adj'] = np.array([len(list(fag.neighbors(i))) for i in range(num_faces)])
        face_features['loops'] = face_loop_counts
        face_features['centroid'] = face_centroids
        face_features['convexity'] = face_convexity

        edge_features = {}
        edge_lens = np.zeros(num_edges)
        edge_convs = np.zeros(num_edges, dtype=int)

        for i in tqdm(range(num_edges), desc="Processing Edges"):
            brepgprop_LinearProperties(self.edges[i], gprop)
            edge_lens[i] = gprop.Mass()
            conv, _ = self._get_edge_convexity_and_angle(i)
            edge_convs[i] = conv

        edge_features['len'] = edge_lens
        edge_features['conv'] = edge_convs

        print("Step 3/4: Computing Proximity Features...")
        dist_matrix = nx.floyd_warshall(fag)

        nodes = list(fag.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        num_nodes = len(nodes)

        dist_array = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
        for u, targets in dist_matrix.items():
            for v, dist in targets.items():
                dist_array[node_index[u], node_index[v]] = dist

        print("  - Computing A2: Spatial Positional Relations (this may be slow)...")
        bbox = Bnd_Box()
        brepbndlib_Add(self.shape, bbox)
        pmin, pmax = bbox.CornerMin(), bbox.CornerMax()
        diag_len = max(pmin.Distance(pmax), 1.0)

        d2_histograms = np.zeros((num_faces, num_faces, 64))
        a3_histograms = np.zeros((num_faces, num_faces, 64))

        all_face_points = [self._get_random_points_on_face(i, self.n_samples_proximity) for i in range(num_faces)]

        for i in tqdm(range(num_faces), desc="Computing A2"):
            for j in range(i, num_faces):
                points_i, normals_i = all_face_points[i]
                points_j, normals_j = all_face_points[j]

                dists = np.linalg.norm(points_i[:, None, :] - points_j[None, :, :], axis=-1)
                d2_hist, _ = np.histogram(dists.flatten() / diag_len, bins=64, range=(0, 1))
                d2_histograms[i, j] = d2_hist
                d2_histograms[j, i] = d2_hist

                ni = normals_i / (np.linalg.norm(normals_i, axis=1, keepdims=True) + 1e-9)
                nj = normals_j / (np.linalg.norm(normals_j, axis=1, keepdims=True) + 1e-9)
                cos_angles = np.clip(np.dot(ni, nj.T), -1.0, 1.0)
                angles = np.arccos(cos_angles).flatten()
                a3_hist, _ = np.histogram(angles, bins=64, range=(0, np.pi))
                a3_histograms[i, j] = a3_hist
                a3_histograms[j, i] = a3_hist

        proximity_A2 = np.concatenate([d2_histograms, a3_histograms], axis=-1)

        print("Step 4/4: Assembling Final Data Object...")
        graph_data = {
            'num_nodes': num_faces,
            'edge_index': np.array(list(fag.edges)).T,
            'face_features': face_features,
            'brep_edge_features': edge_features,
            'proximity_A1_shortest_path': dist_array,
            'proximity_A2_spatial_relations': proximity_A2,
        }

        print("\nFeature extraction complete.")
        return graph_data

def convert_step_to_graph(input_step: str, output_pkl: str):
    print(f"Processing '{input_step}'...")
    try:
        extractor = BRepGraphExtractor(input_step)
        graph_data = extractor.extract_features()

        with open(output_pkl, 'wb') as f:
            pickle.dump(graph_data, f)

        print(f"\n✅ Success! Graph data saved to '{output_pkl}'")
        print(f" -> Graph has {graph_data['num_nodes']} nodes (faces).")
        print(f" -> Graph has {graph_data['edge_index'].shape[1]} edges.")
        face_features = graph_data.get('face_features')
        if face_features is not None:
            print(f" -> Node features count: {len(face_features)}")
        edge_features = graph_data.get('brep_edge_features')
        if edge_features is not None:
            print(f" -> Edge features count: {len(edge_features)}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please ensure the STEP file is valid and the environment is set up correctly.")

# Keep CLI compatibility
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a B-rep STEP file to a graph representation.")
    parser.add_argument("input_step", type=str, help="Path to the input STEP file.")
    parser.add_argument("output_pkl", type=str, help="Path to save the output graph data pickle file.")
    args = parser.parse_args()

    convert_step_to_graph(args.input_step, args.output_pkl)
