from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_ManifoldSolidBrep
from OCC.Core.IFSelect import IFSelect_RetDone
import pandas as pd
import numpy as np

# -------------------- Utilities --------------------
def get_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods.Face(exp.Current()))
        exp.Next()
    return faces

def get_edges(shape):
    edges = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edges.append(topods.Edge(exp.Current()))
        exp.Next()
    return edges

def face_center(face):
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return np.array([(xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2])

def write_step(shape, filename):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_ManifoldSolidBrep)
    status = writer.Write(filename)
    assert status == IFSelect_RetDone, f"STEP export failed: {filename}"
    print(f"✅ Wrote {filename}")

# ------------ Stepwise Ground-Truth Mapping Tracker ------------
def map_faces_history(operation, old_face_list):
    mapping = {}
    for i, face in enumerate(old_face_list):
        mapped = []
        occ_list = operation.Modified(face)
        for k in range(occ_list.Length()):
            mapped.append(hash(topods.Face(occ_list.Value(k))))
        mapping[i] = mapped
    return mapping

# -------------------- Feature Modeling --------------------
def make_modified_shape_and_track(box):
    """Create a through-hole, fillet, and chamfer with ground-truth face mapping tracking."""
    face_maps = {}  # To track face maps at every stage

    # 1. Initial faces
    orig_faces = get_faces(box)
    face_maps['step_0_box'] = {i: [hash(f)] for i, f in enumerate(orig_faces)}

    # 2. Through-hole cut
    print("➡️ Creating through-hole...")
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5,10,0), gp_Dir(0,0,1)), 2.5, 30).Shape()
    cut = BRepAlgoAPI_Cut(box, cyl)
    cut_shape = cut.Shape()
    cut_faces = get_faces(cut_shape)
    # Track mapping box->cut
    face_maps['step_1_cut'] = map_faces_history(cut, orig_faces)

    # 3. Fillet on hole edges
    print("➡️ Applying fillet on hole edges...")
    fillet = BRepFilletAPI_MakeFillet(cut_shape)
    cut_edges = get_edges(cut_shape)
    added_fillet = 0
    for edge in cut_edges:
        # Add a fillet to *every* edge for demonstration (you can improve selectivity here!)
        fillet.Add(1.0, edge)
        added_fillet += 1
    if added_fillet:
        fillet.Build()
        fillet_shape = fillet.Shape()
        # Track mapping cut->fillet
        cut_faces_for_map = get_faces(cut_shape)
        face_maps['step_2_fillet'] = map_faces_history(fillet, cut_faces_for_map)
    else:
        fillet_shape = cut_shape
        face_maps['step_2_fillet'] = {i: [hash(f)] for i, f in enumerate(get_faces(fillet_shape))}
        print("⚠️ No valid edges found for fillet, skipped.")

    # 4. Chamfer on all top edges (again, for demonstration, use all edges)
    print("➡️ Applying chamfer on all edges as demo...")
    chamfer = BRepFilletAPI_MakeChamfer(fillet_shape)
    fillet_edges = get_edges(fillet_shape)
    added_chamfer = 0
    for edge in fillet_edges:
        chamfer.Add(0.8, edge)
        added_chamfer += 1
    if added_chamfer:
        chamfer.Build()
        chamfer_shape = chamfer.Shape()
        # Track mapping fillet->chamfer
        fillet_faces_for_map = get_faces(fillet_shape)
        face_maps['step_3_chamfer'] = map_faces_history(chamfer, fillet_faces_for_map)
    else:
        chamfer_shape = fillet_shape
        face_maps['step_3_chamfer'] = {i: [hash(f)] for i, f in enumerate(get_faces(chamfer_shape))}
        print("⚠️ No valid edges found for chamfer, skipped.")

    print("✅ Feature modeling & ground-truth mapping done.")
    return chamfer_shape, face_maps, orig_faces

# -------------------- Main --------------------
if __name__ == "__main__":
    print("➡️ Creating base box...")
    box = BRepPrimAPI_MakeBox(10, 20, 30).Shape()

    # --- Modeling with Provenance ---
    modified_box, face_maps, orig_faces = make_modified_shape_and_track(box)

    # --- Flatten final mapping: from original face index to final faces ---
    print("➡️ Building original-to-final mapping via provenance...")

    # Traverse through each mapping (box->cut->fillet->chamfer)
    map0 = face_maps['step_0_box']
    map1 = face_maps['step_1_cut']
    map2 = face_maps['step_2_fillet']
    map3 = face_maps['step_3_chamfer']

    # For each original face, trace its descendants through each modeling operation
    final_mapping = []
    for i_orig, hashes0 in map0.items():
        descendants = hashes0
        for m in (map1, map2, map3):
            next_descendants = []
            for h in descendants:
                # Find which face-index in prior step had this hash
                found_key = [k for k, L in m.items() if h in L]
                # Then get its descendants
                for key in found_key:
                    next_descendants += m[key]
            descendants = list(set(next_descendants))  # remove duplicates
        # For CSV, output all descendants (could be empty, one, or many faces)
        final_mapping.append({
            "original_face_id": i_orig,
            "descendant_face_hashes": descendants if descendants else [None]
        })

    # Export
    write_step(box, "block_original.step")
    write_step(modified_box, "block_modified.step")

    # Save mapping to CSV (expanding lists into columns)
    max_desc = max(len(row['descendant_face_hashes']) for row in final_mapping)
    dict_rows = []
    for row in final_mapping:
        d = {"original_face_id": row["original_face_id"]}
        for j in range(max_desc):
            d[f"descendant_face_hash_{j}"] = row["descendant_face_hashes"][j] if j < len(row["descendant_face_hashes"]) else None
        dict_rows.append(d)
    df = pd.DataFrame(dict_rows)
    df.to_csv("block_face_groundtruth_mapping.csv", index=False)
    print("✅ Ground-truth mapping exported as block_face_groundtruth_mapping.csv")