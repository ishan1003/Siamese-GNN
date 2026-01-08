# ---------- test_model_chunk1.py ----------
# Chunk 1/3: imports, utils, dataset, model (NULL classifier enabled)

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from scipy.optimize import linear_sum_assignment

# Hungarian will be used in chunk 2 (scipy import there)


# ---------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    # expected keys: model_state, config, feat_mean, feat_std
    return (
        ckpt["model_state"],
        ckpt["config"],
        torch.tensor(ckpt["feat_mean"], dtype=torch.float32),
        torch.tensor(ckpt["feat_std"], dtype=torch.float32),
    )


# ---------------------------------------------------
# EDGE CONVERSION
# ---------------------------------------------------
def convert_edges(edges, id_list):
    id_to_idx = {int(i): x for x, i in enumerate(id_list)}
    out = []
    for a, b in edges:
        if int(a) in id_to_idx and int(b) in id_to_idx:
            ai = id_to_idx[int(a)]
            bi = id_to_idx[int(b)]
            out.append([ai, bi])
            out.append([bi, ai])
    if not out:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(out, dtype=torch.long).t().contiguous()


# ---------------------------------------------------
# DATASET
# ---------------------------------------------------
class TestDataset:
    def __init__(self, json_path, feature_idx, feat_mean, feat_std):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.keys = sorted(self.data.keys(), key=lambda x: int(x))
        self.feature_idx = feature_idx
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.data[key]

        # A
        A_ids = sorted(pair["A_embeddings"].keys(), key=lambda x: int(x))
        xA = torch.tensor([pair["A_embeddings"][f] for f in A_ids], dtype=torch.float32)
        xA = (xA[:, self.feature_idx] - self.feat_mean) / self.feat_std
        edgeA = convert_edges(pair.get("A_edges", []), A_ids)

        # B
        B_ids = sorted(pair["B_embeddings"].keys(), key=lambda x: int(x))
        xB = torch.tensor([pair["B_embeddings"][f] for f in B_ids], dtype=torch.float32)
        xB = (xB[:, self.feature_idx] - self.feat_mean) / self.feat_std
        edgeB = convert_edges(pair.get("B_edges", []), B_ids)

        A_map = {int(a): i for i, a in enumerate(A_ids)}
        B_map = {int(b): i for i, b in enumerate(B_ids)}

        matches = []
        for a, b in pair["mappings"]:
            ai = -1 if a == "NULL" else A_map.get(int(a), -1)
            bi = -1 if b == "NULL" else B_map.get(int(b), -1)
            matches.append([ai, bi])

        return (
            key,
            Data(x=xA, edge_index=edgeA, xt=A_ids),
            Data(x=xB, edge_index=edgeB, xt=B_ids),
            torch.tensor(matches, dtype=torch.long),
        )


# ---------------------------------------------------
# MODEL DEFINITION (updated to return null logits)
# ---------------------------------------------------
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, hid)
        self.g3 = SAGEConv(hid, out_dim)

    def forward(self, x, e):
        x = F.relu(self.g1(x, e))
        x = F.relu(self.g2(x, e))
        return self.g3(x, e)


class SiameseGNN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid, out_dim)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(out_dim, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim),
        )

        # NULL classifier head (same as training)
        self.null_head = torch.nn.Sequential(
            torch.nn.Linear(out_dim + proj_dim + 1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, A, B):
        # encoder outputs (h: encoding before projection)
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)

        # projected embeddings used for similarity
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)

        # classifier logits (use same features as during training)
        featA = torch.cat([hA, zA], dim=1)
        featB = torch.cat([hB, zB], dim=1)
        logitsA = self.null_head(featA).squeeze(1)
        logitsB = self.null_head(featB).squeeze(1)

        # return embeddings and logits
        return zA, zB, logitsA, logitsB

# ---------- end of chunk 1 ----------

# ---------------------------------------------------
# HUNGARIAN ASSIGNMENT (raw matching, no NULL logic yet)
# ---------------------------------------------------
def hungarian_assign(zA, zB):
    """
    Hungarian match on full similarity matrix.
    NOTE: NULL will be handled later using classifier output.
    """
    sims = zA @ zB.t()  # cosine similarities
    # Hungarian minimizes cost, so convert similarity → cost
    cost = sims.max().item() - sims.cpu().numpy()
    row, col = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row, col)}


# ---------------------------------------------------
# BUILD FINAL PREDICTION MAP (Option A)
# Classifier decides NULL; Hungarian or argmax decides match.
# ---------------------------------------------------
def compute_pred_map(zA, zB, logitsA, use_hungarian, null_thr):
    """
    Option A rule:
    1. If classifier says NULL → force NULL
    2. Otherwise assign best B using Hungarian OR similarity argmax
    """
    N = zA.shape[0]

    # classifier probabilities
    p_null = torch.sigmoid(logitsA)

    sims = zA @ zB.t() if zB.shape[0] > 0 else None

    if use_hungarian and zB.shape[0] > 0:
        raw_map = hungarian_assign(zA, zB)
    else:
        raw_map = {a: int(sims[a].argmax()) for a in range(N)} if sims is not None else {}

    pred_map = {}
    for a in range(N):
        if p_null[a] > null_thr:
            pred_map[a] = -1
        else:
            pred_map[a] = raw_map.get(a, -1)

    return pred_map, sims, p_null



# ---------------------------------------------------
# TOP-1 / TOP-5 METRICS
# ---------------------------------------------------
def compute_top1_top5(sims, matches):
    """
    sims: [NA, NB] similarity matrix
    matches: tensor with ground truth alignments [(a_idx, b_idx)]
    """
    valid_A = []
    valid_B = []
    for a, b in matches.tolist():
        if a != -1 and b != -1:
            valid_A.append(a)
            valid_B.append(b)

    if len(valid_A) == 0:
        return None, None

    valid_A = torch.tensor(valid_A)
    valid_B = torch.tensor(valid_B)

    # Top-1 accuracy
    pred1 = sims.argmax(dim=1)[valid_A]
    top1 = (pred1 == valid_B).float().mean().item()

    # Top-5 accuracy
    k = min(5, sims.shape[1])
    top5_candidates = torch.topk(sims[valid_A], k, dim=1).indices
    top5 = torch.any(top5_candidates == valid_B.unsqueeze(1), dim=1).float().mean().item()

    return top1, top5


# ---------------------------------------------------
# STRUCTURAL EDGE CONSISTENCY (Option A)
# ---------------------------------------------------
def structural_edge_consistency(A, B, pred_map):
    """
    Option A structural metric:
    For each undirected edge (u,v) in A:
        If pred_map[u] = u', pred_map[v] = v'
        And (u',v') is an edge in B → count as consistent

    Returns:
        ratio = consistent_edges / total_checked_edges
    """
    edgeA = A.edge_index.t().tolist()
    edgeB = B.edge_index.t().tolist()

    # Make undirected edge sets
    undA = set()
    for u, v in edgeA:
        if u != v:
            undA.add((min(int(u), int(v)), max(int(u), int(v))))

    undB = set()
    for u, v in edgeB:
        if u != v:
            undB.add((min(int(u), int(v)), max(int(u), int(v))))

    total = 0
    consistent = 0

    for u, v in undA:
        if u not in pred_map or v not in pred_map:
            continue
        pu = pred_map[u]
        pv = pred_map[v]

        if pu == -1 or pv == -1:
            continue

        total += 1
        if (min(pu, pv), max(pu, pv)) in undB:
            consistent += 1

    return (consistent / total) if total > 0 else 0.0

# ---------- end of chunk 2 ----------

def test_model(ckpt_path, test_json, use_hungarian=False):

    print("\nLoading checkpoint...")
    state, cfg, feat_mean, feat_std = load_checkpoint(ckpt_path)
    NULL_THR = cfg.get("null_prob_threshold", 0.5)
    print(f"Using NULL classifier threshold = {NULL_THR:.3f}")


    print("Loading dataset...")
    ds = TestDataset(test_json, cfg["use_features"], feat_mean, feat_std)

    print("Building model...")
    model = SiameseGNN(
        in_dim=len(cfg["use_features"]),
        hid=cfg["encoder_hidden"],
        out_dim=cfg["encoder_out"],
        proj_dim=cfg["proj_dim"],
    )
    model.load_state_dict(state)
    model.eval()

    os.makedirs("model_pairs", exist_ok=True)

    # ---------------------------------------------------
    # Aggregated metrics
    # ---------------------------------------------------
    all_top1, all_top5 = [], []
    all_f1, all_unified = [], []
    all_struct = []

    print("\n--------------------------------------------")
    print("PER-MODEL ACCURACY SUMMARY")
    print("--------------------------------------------")

    with torch.no_grad():

        for key, A, B, m in ds:

            # Forward pass (now includes classifier logits)
            hA = model.encoder(A.x, A.edge_index)
            hB = model.encoder(B.x, B.edge_index)

            zA = F.normalize(model.proj(hA), dim=1)
            zB = F.normalize(model.proj(hB), dim=1)

            if zB.shape[0] > 0:
                sims_AB = zA @ zB.t()                  # [NA, NB]
                maxA = sims_AB.max(dim=1).values
                meanA = sims_AB.mean(dim=1)
                sim_gap_A = maxA - meanA
            else:
                sim_gap_A = torch.zeros(zA.size(0))


            logitsA = model.null_head(
                torch.cat([hA, zA, sim_gap_A.unsqueeze(1)], dim=1)
            ).squeeze(1)


            # ---------------------------------------------------
            # Prediction using NULL classifier (Option A)
            # ---------------------------------------------------
            pred_map, sims, p_null = compute_pred_map(
                zA, zB, logitsA, use_hungarian, NULL_THR
            )


            max_s = sims.max(dim=1).values

            # ---------------------------------------------------
            # Compute metrics (Top-1 / Top-5)
            # ---------------------------------------------------
            top1, top5 = compute_top1_top5(sims, m)

            # ---------------------------------------------------
            # Compute NULL-F1 + unified accuracy
            # ---------------------------------------------------
            TP = FP = FN = 0
            correct = 0
            total = 0

            for a, b in m.tolist():
                if a == -1:
                    continue

                pred_null = (pred_map[a] == -1)

                if b == -1:
                    TP += int(pred_null)
                    FN += int(not pred_null)
                    correct += int(pred_null)
                else:
                    FP += int(pred_null)
                    correct += int(not pred_null and pred_map[a] == b)

                total += 1

            prec = TP / (TP + FP + 1e-12)
            rec = TP / (TP + FN + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            unified = correct / total if total > 0 else 0.0

            # ---------------------------------------------------
            # Structural Edge Consistency (Option A)
            # ---------------------------------------------------
            struct_cons = structural_edge_consistency(A, B, pred_map)

            # Store aggregated results
            if top1 is not None:
                all_top1.append(top1)
                all_top5.append(top5)
            all_f1.append(f1)
            all_unified.append(unified)
            all_struct.append(struct_cons)

            # ---------------------------------------------------
            # Per-model console print
            # ---------------------------------------------------
            print(
                f"[Model {key}]  "
                f"Top1={100*(top1 or 0):.1f}%  "
                f"Top5={100*(top5 or 0):.1f}%  "
                f"NULL-F1={100*f1:.1f}%  "
                f"Unified={100*unified:.1f}%  "
                f"Struct={100*struct_cons:.1f}%"
            )

            # ---------------------------------------------------
            # Save JSON results
            # ---------------------------------------------------
            per_json = {
                "model_id": key,
                "hungarian": use_hungarian,

                "metrics": {
                    "top1": float(top1) if top1 is not None else None,
                    "top5": float(top5) if top5 is not None else None,
                    "null_f1": float(f1),
                    "unified_accuracy": float(unified),
                    "structural_edge_consistency": float(struct_cons),
                },

                "A_ids": [int(x) for x in A.xt],
                "B_ids": [int(x) for x in B.xt],
                "p_null": p_null.tolist(),
                "max_similarity_per_A": max_s.tolist(),

                "predicted_matches": [
                    {
                        "A_index": a,
                        "A_xt_id": int(A.xt[a]),
                        "pred_B_index": None if pred_map[a] == -1 else pred_map[a],
                        "pred_B_xt_id": None if pred_map[a] == -1 else int(B.xt[pred_map[a]]),
                        "similarity": float(max_s[a]),
                        "classifier_null_prob": float(p_null[a]),
                        "is_null": bool(pred_map[a] == -1),
                    }
                    for a, _ in m.tolist()
                    if a != -1
                ],
            }

            with open(f"model_pairs/model_{key}.json", "w") as f:
                json.dump(per_json, f, indent=2)

    # ---------------------------------------------------
    # Print aggregated stats
    # ---------------------------------------------------
    print("\n===================================")
    print("FINAL AGGREGATED METRICS")
    print("===================================")
    print(f"Using Hungarian:         {use_hungarian}")
    print(f"Top-1 Accuracy:          {100*np.mean(all_top1):.2f}%")
    print(f"Top-5 Accuracy:          {100*np.mean(all_top5):.2f}%")
    print(f"NULL F1 Score:           {100*np.mean(all_f1):.2f}%")
    print(f"Unified Accuracy:        {100*np.mean(all_unified):.2f}%")
    print(f"Struct Consistency:      {100*np.mean(all_struct):.2f}%")
    print("===================================\n")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--use_hungarian", action="store_true")

    args = parser.parse_args()
    test_model(args.checkpoint, args.test_json, args.use_hungarian)
