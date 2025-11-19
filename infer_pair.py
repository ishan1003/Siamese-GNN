#!/usr/bin/env python3
"""
infer_hybrid_gnn.py

Hybrid Hungarian + GNN inference for CAD face matching.

Usage (example):
python infer_hybrid_gnn.py \
  --pair 60_pair.json \
  --checkpoint siamese_infonce_null.pt \
  --xt XT_merged_new1_doubled_fixed.json \
  --out mapping_60.json \
  --topk 12 \
  --null_cost 0.5 \
  --device cpu

Notes:
- If the checkpoint doesn't store feat_mean/feat_std, pass --xt to compute identical z-score stats used in training.
- If you don't pass --xt and checkpoint lacks stats, script will fallback to row-wise L2 normalization.
"""

import argparse
import json
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn as nn

# --------------------------
# Model definitions (match training)
# --------------------------
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.g1 = SAGEConv(input_dim, hidden_dim)
        self.g2 = SAGEConv(hidden_dim, hidden_dim)
        self.g3 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.g1(x, edge_index))
        x = F.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)
        return x

class SiameseGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, proj_dim=64):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, data: Data):
        h = self.encoder(data.x, data.edge_index)
        z = F.normalize(self.proj(h), dim=1)
        return F.normalize(h, dim=1), z

# --------------------------
# Utilities: IO + preprocessing
# --------------------------
def load_pair_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def extract_feature_stats_from_xt(xt_path: str, feature_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std across all A+B embeddings in xt_path.
    Returns (mean, std) as numpy arrays.
    """
    with open(xt_path, 'r') as f:
        xt = json.load(f)
    feats = []
    for k,p in xt.items():
        for _, v in p.get("A_embeddings", {}).items():
            feats.append(v)
        for _, v in p.get("B_embeddings", {}).items():
            feats.append(v)
    if len(feats) == 0:
        raise RuntimeError("No embeddings found in XT.")
    arr = np.array(feats, dtype=np.float32)[:, feature_idx]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-9
    return mean, std

def build_matrices_from_pair(pair: dict, feature_idx: List[int],
                             feat_mean: np.ndarray=None, feat_std: np.ndarray=None):
    """
    Returns:
      A_ids (list[str]), B_ids (list[str]),
      A_mat (np.ndarray [nA,f]), B_mat (nB,f),
      A_edges list of [a,b], B_edges list of [a,b]
    """
    A_ids = sorted(pair.get('A_embeddings', {}).keys(), key=lambda x: int(x))
    B_ids = sorted(pair.get('B_embeddings', {}).keys(), key=lambda x: int(x))
    A_mat = np.array([pair['A_embeddings'][aid] for aid in A_ids], dtype=np.float32)[:, feature_idx] if len(A_ids)>0 else np.zeros((0,len(feature_idx)),dtype=np.float32)
    B_mat = np.array([pair['B_embeddings'][bid] for bid in B_ids], dtype=np.float32)[:, feature_idx] if len(B_ids)>0 else np.zeros((0,len(feature_idx)),dtype=np.float32)

    # Apply z-score normalization if stats provided; else leave raw (we will L2-normalize later)
    if feat_mean is not None and feat_std is not None:
        A_mat = (A_mat - feat_mean[None,:]) / feat_std[None,:]
        B_mat = (B_mat - feat_mean[None,:]) / feat_std[None,:]

    A_edges = pair.get('A_edges', [])
    B_edges = pair.get('B_edges', [])
    return A_ids, B_ids, A_mat, B_mat, A_edges, B_edges

def convert_edges_to_index(edges, id_list):
    id_to_idx = {int(i): idx for idx, i in enumerate(id_list)}
    out = []
    for a,b in edges:
        try:
            ai = id_to_idx[int(a)]; bi = id_to_idx[int(b)]
        except Exception:
            continue
        out.append([ai, bi]); out.append([bi, ai])
    if len(out) == 0:
        return np.zeros((2,0),dtype=np.int64)
    return np.array(out, dtype=np.int64).T

def build_pyg_data(x_mat: np.ndarray, edge_idx_np: np.ndarray):
    x = torch.tensor(x_mat, dtype=torch.float32)
    if edge_idx_np.size == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_idx_np, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# --------------------------
# Candidate selection (fast)
# --------------------------
def l2_row_normalize(mat: np.ndarray):
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return mat / norms

def candidate_selection_knn(A_mat: np.ndarray, B_mat: np.ndarray, topk: int):
    """
    Work in numpy. Uses cosine similarity via row-normalized dot.
    Returns: candidates: list of lists (for each A index, list of B indices)
    """
    nA = A_mat.shape[0]; nB = B_mat.shape[0]
    if nA==0:
        return []
    if nB==0:
        return [[] for _ in range(nA)]
    An = l2_row_normalize(A_mat)
    Bn = l2_row_normalize(B_mat)
    sims = An @ Bn.T
    topk = min(topk, nB)
    candidates = [list(np.argsort(-sims[i])[:topk]) for i in range(nA)]
    return candidates

# --------------------------
# Compute projector embeddings z (GNN)
# --------------------------
def compute_z_vectors(model: nn.Module, data: Data, device: torch.device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        _, z = model(data)
    return z.cpu()

# --------------------------
# Final Hungarian assignment with NULLs
# --------------------------
def final_assignment_with_null(zA: torch.Tensor, zB: torch.Tensor,
                               candidates: List[List[int]],
                               null_cost: float = 0.5,
                               cost_fn=None):
    """
    Corrected Hungarian assignment:
    - Only A-faces are rows.
    - Columns = B-faces + NULL columns.
    """

    zA_np = zA.cpu().numpy()
    zB_np = zB.cpu().numpy()

    nA, D = zA_np.shape
    nB = zB_np.shape[0]

    if cost_fn is None:
        cost_fn = lambda sim: 1.0 - float(sim)   # lower cost = better match

    # ----- CASE: empty side -----
    if nA == 0:
        return {}, list(range(nB))
    if nB == 0:
        return {i: -1 for i in range(nA)}, []

    # ----- Cost matrix: rows = A (nA), cols = B (nB) + A_NULL (nA) -----
    num_cols = nB + nA
    C = np.full((nA, num_cols), 100.0, dtype=np.float32)

    # Full similarity matrix
    sims_full = zA_np @ zB_np.T    # [nA, nB]

    # ----- Fill A → B costs (only candidate B indices allowed) -----
    for i in range(nA):
        for j in candidates[i]:
            sim = sims_full[i, j]
            C[i, j] = cost_fn(sim)

    # ----- Fill A → NULL costs -----
    for i in range(nA):
        null_col = nB + i
        C[i, null_col] = float(null_cost)

    # ----- Run Hungarian -----
    row_ind, col_ind = linear_sum_assignment(C)

    # ----- Build final mapping -----
    mapping = {}
    assigned_B = set()

    for a_idx, col in zip(row_ind, col_ind):
        if col < nB:
            # Matched to B-face
            mapping[a_idx] = col
            assigned_B.add(col)
        else:
            # Matched to NULL
            mapping[a_idx] = -1

    # B-faces never selected → NULL
    unassigned_B = [j for j in range(nB) if j not in assigned_B]

    return mapping, unassigned_B


# --------------------------
# High-level inference function
# --------------------------
def hybrid_infer_pair(pair: dict, checkpoint_path: str, xt_path_for_stats: str = None,
                      topk: int = 10, null_cost: float = 0.5, device: str = "cpu",
                      override_threshold: float = None, verbose: bool = True):
    """
    Returns: mappings list of [A_id, B_id or 'NULL'], meta dict
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

    use_features = cfg.get('use_features', None)
    if use_features is None:
        raise RuntimeError("Checkpoint config must contain 'use_features' (feature indices).")

    # compute feature stats if available in config else from xt_path
    feat_mean = cfg.get('feat_mean', None)
    feat_std = cfg.get('feat_std', None)
    if feat_mean is None or feat_std is None:
        if xt_path_for_stats is not None:
            fm, fs = extract_feature_stats_from_xt(xt_path_for_stats, use_features)
            feat_mean = fm; feat_std = fs
        else:
            feat_mean = None; feat_std = None

    # build matrices
    A_ids, B_ids, A_mat, B_mat, A_edges, B_edges = build_matrices_from_pair(pair, use_features,
                                                                           feat_mean=(np.array(feat_mean) if feat_mean is not None else None),
                                                                           feat_std=(np.array(feat_std) if feat_std is not None else None))
    # Candidate selection on normalized raw features (if z-score done) else L2 row-normalized
    if feat_mean is not None:
        A_feat_for_knn = A_mat.copy()
        B_feat_for_knn = B_mat.copy()
    else:
        # fallback L2 normalization on raw columns
        A_feat_for_knn = l2_row_normalize(A_mat) if A_mat.size else A_mat
        B_feat_for_knn = l2_row_normalize(B_mat) if B_mat.size else B_mat

    candidates = candidate_selection_knn(A_feat_for_knn, B_feat_for_knn, topk=topk)

    # build PyG data and model
    A_edges_idx = convert_edges_to_index(A_edges, A_ids) if len(A_edges)>0 else np.array([],dtype=np.int64)
    B_edges_idx = convert_edges_to_index(B_edges, B_ids) if len(B_edges)>0 else np.array([],dtype=np.int64)
    dataA = build_pyg_data(A_mat, A_edges_idx)
    dataB = build_pyg_data(B_mat, B_edges_idx)

    device_t = torch.device(device)
    in_dim = len(use_features)
    hid = cfg.get('encoder_hidden', 64)
    out_dim = cfg.get('encoder_out', 32)
    proj_dim = cfg.get('proj_dim', 64)

    model = SiameseGNN(in_dim, hid, out_dim, proj_dim)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device_t)
    model.eval()

    # compute z vectors
    zA = compute_z_vectors(model, dataA, device_t) if dataA.x.shape[0] > 0 else torch.empty((0,proj_dim))
    zB = compute_z_vectors(model, dataB, device_t) if dataB.x.shape[0] > 0 else torch.empty((0,proj_dim))

    # ensure normalized
    if zA.numel() > 0:
        zA = F.normalize(zA, dim=1)
    if zB.numel() > 0:
        zB = F.normalize(zB, dim=1)

    # Optionally compute threshold (midpoint of pos and null) using pair's GT if present in pair['mappings']
    threshold = override_threshold
    if threshold is None and 'mappings' in pair:
        # derive pos sims and null max sims for pair if mappings exist
        pos_sims = []
        null_max_sims = []
        mp = { (int(a) if a!="NULL" else -1): (int(b) if b!="NULL" else -1) for a,b in pair['mappings'] }
        for a_idx, b_idx in mp.items():
            if a_idx == -1: continue
            if b_idx == -1:
                if zB.shape[0] > 0 and zA.shape[0] > 0:
                    sims = (zA[a_idx:a_idx+1] @ zB.t()).cpu().numpy().reshape(-1)
                    null_max_sims.append(float(sims.max()))
            else:
                if zB.shape[0] > 0 and zA.shape[0] > 0:
                    sim = float((zA[a_idx] * zB[b_idx]).sum().item())
                    pos_sims.append(sim)
        if len(pos_sims)>0 or len(null_max_sims)>0:
            mean_pos = float(np.mean(pos_sims)) if len(pos_sims)>0 else 0.5
            mean_null = float(np.mean(null_max_sims)) if len(null_max_sims)>0 else -0.5
            threshold = float((mean_pos + mean_null) / 2.0)
        else:
            # fallback
            threshold = 0.53
    if threshold is None:
        threshold = 0.53

    # -------------------------------
    # DEBUG PRINT: Similarity scores
    # -------------------------------
    if zA.shape[0] > 0 and zB.shape[0] > 0:
        sim_matrix = (zA @ zB.T).cpu().numpy()   # [nA, nB]

        max_sim_each_A = sim_matrix.max(axis=1)
        mean_sim_each_A = sim_matrix.mean(axis=1)
        top5_sim_each_A = np.sort(sim_matrix, axis=1)[:, -5:]  # last 5 = highest

        print("\n=== SIMILARITY DEBUG ===")
        print("A_faces:", zA.shape[0], " B_faces:", zB.shape[0])
        print("Max similarity across all A:", float(max_sim_each_A.max()))
        print("Min similarity across all A:", float(max_sim_each_A.min()))
        print("Mean of max similarities:", float(max_sim_each_A.mean()))
        print("\nPer-A Max Similarity:")
        for i, s in enumerate(max_sim_each_A):
            print(f"A[{i}] max_sim = {s:.4f}")

        print("\nPer-A Top-5 Similarities:")
        for i, row in enumerate(top5_sim_each_A):
            print(f"A[{i}] top5 = {[float(x) for x in row]}")
        
        print("\nComplete Similarity Matrix (A x B):")
        print(sim_matrix)
        print("=== END DEBUG ===\n")
    else:
        print("\n(No similarity – one side is empty)\n")


    # final Hungarian assignment using z similarities & candidate set
    mapping_idx, unassigned_B = final_assignment_with_null(zA, zB, candidates, null_cost=null_cost)

    # create mappings in terms of IDs
    assigned_B_vals = set(mapping_idx.values()) - {-1}
    mappings = []
    for a_i, a_id in enumerate(A_ids):
        b_idx = mapping_idx.get(a_i, -1)
        if b_idx == -1:
            mappings.append([str(a_id), "NULL"])
        else:
            mappings.append([str(a_id), str(B_ids[b_idx])])
    for b_idx, b_id in enumerate(B_ids):
        if b_idx not in assigned_B_vals:
            mappings.append(["NULL", str(b_id)])

    meta = {
        "num_A": len(A_ids),
        "num_B": len(B_ids),
        "num_matched": int(sum(1 for v in mapping_idx.values() if v != -1)),
        "threshold": float(threshold),
        "null_cost": float(null_cost),
        "topk": int(topk)
    }
    if verbose:
        print(f"A: {len(A_ids)}, B: {len(B_ids)}, matched: {meta['num_matched']}, thr={meta['threshold']:.3f}, null_cost={meta['null_cost']}")

    return mappings, meta

# --------------------------
# CLI
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", required=True, help="JSON file for single pair")
    p.add_argument("--checkpoint", required=True, help="trained checkpoint (torch .pt)")
    p.add_argument("--xt", required=False, help="XT json used for training (to compute feat mean/std). Optional but recommended.")
    p.add_argument("--out", required=True, help="output mapping JSON path")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--null_cost", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threshold", type=float, default=None, help="override similarity threshold (optional)")
    args = p.parse_args()

    pair = load_pair_json(args.pair)
    mappings, meta = hybrid_infer_pair(pair, args.checkpoint, xt_path_for_stats=args.xt,
                                      topk=args.topk, null_cost=args.null_cost, device=args.device,
                                      override_threshold=args.threshold, verbose=True)
    out = {"mappings": mappings, "meta": meta}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved mapping to:", args.out)

if __name__ == "__main__":
    main()
