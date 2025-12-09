# ======================
# test_model.py (FULL SCRIPT with Hungarian + Structural Edge Consistency)
# ======================

import json, os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from scipy.optimize import linear_sum_assignment  # Hungarian Algorithm


# ---------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
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
# MODEL DEFINITION
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
        # Unused for metric but loaded for compatibility
        self.null_head = torch.nn.Sequential(
            torch.nn.Linear(out_dim + proj_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, A, B):
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)
        return zA, zB


# ---------------------------------------------------
# POS / NULL SIMILARITY COLLECTION
# ---------------------------------------------------
def collect_pos_null_sims(zA, zB, matches):
    pos_sims = []
    null_sims = []

    for a, b in matches.tolist():
        if a != -1 and b != -1:  # positive match
            pos_sims.append((zA[a] * zB[b]).sum().item())
        if a != -1 and b == -1:  # A should be null
            null_sims.append((zA[a] @ zB.t()).max().item())

    return pos_sims, null_sims


# ---------------------------------------------------
# HUNGARIAN ASSIGNMENT
# ---------------------------------------------------
def hungarian_assign(zA, zB, matches, thr):
    sims = zA @ zB.t()
    max_s = sims.max(dim=1).values

    # filter A faces above threshold
    active_A = [a for a, b in matches.tolist() if a != -1 and b != -1 and max_s[a] >= thr]
    if len(active_A) == 0:
        return {a: -1 for a, _ in matches.tolist()}

    active_A = torch.tensor(active_A)
    cost = sims.max().item() - sims[active_A]

    row, col = linear_sum_assignment(cost.cpu().numpy())

    pred = {int(a): -1 for a, _ in matches.tolist()}
    for i, r in enumerate(row):
        pred[int(active_A[r])] = int(col[i])

    # threshold-based NULL
    for a, _ in matches.tolist():
        if a != -1 and max_s[a] < thr:
            pred[a] = -1

    return pred


# ---------------------------------------------------
# TEST LOOP
# ---------------------------------------------------
def test_model(ckpt_path, test_json, use_hungarian=False):

    print("Loading checkpoint...")
    state, cfg, feat_mean, feat_std = load_checkpoint(ckpt_path)

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

    print("Estimating NULL threshold...")
    all_pos, all_null = [], []
    with torch.no_grad():
        for _, A, B, m in ds:
            zA, zB = model(A, B)
            ps, ns = collect_pos_null_sims(zA, zB, m)
            all_pos += ps
            all_null += ns

    mp = np.mean(all_pos) if all_pos else 0.5
    mn = np.mean(all_null) if all_null else -0.5
    thr = float(np.clip((mp + mn) / 2, -0.9, 0.9))
    print(f"Threshold = {thr:.4f}")

    all_top1 = []
    all_top5 = []
    all_f1 = []
    all_unified = []
    all_struct_consistency = []

    print("Evaluating... Hungarian =", use_hungarian)
    os.makedirs("model_outputs", exist_ok=True)
    os.makedirs("model_pairs", exist_ok=True)

    print("\n--------------------------------------------")
    print("PER-MODEL ACCURACY SUMMARY")
    print("--------------------------------------------")

    with torch.no_grad():
        for key, A, B, m in ds:
            zA, zB = model(A, B)
            sims = zA @ zB.t()  # cosine similarity (zA,zB are normalized)
            max_s = sims.max(dim=1).values

            if use_hungarian:
                pred_map = hungarian_assign(zA, zB, m, thr)
            else:
                pred_map = {
                    a: (-1 if max_s[a] < thr else sims[a].argmax().item())
                    for a, _ in m.tolist() if a != -1
                }

            TP = FP = FN = 0
            top1 = top5 = unified = None
            validA = []
            validB = []
            predsB = []

            for a, b in m.tolist():
                if a != -1 and b != -1:
                    validA.append(a)
                    validB.append(b)
                    predsB.append(pred_map[a])

            if validA:
                validA = torch.tensor(validA)
                validB = torch.tensor(validB)
                predB = torch.tensor(predsB)

                top1 = (predB == validB).float().mean().item()

                k = min(5, zB.shape[0])
                top5 = torch.any(
                    torch.topk(sims[validA], k, dim=1).indices
                    == validB.unsqueeze(1),
                    dim=1,
                ).float().mean().item()

            total = 0
            correct = 0
            for a, b in m.tolist():
                if a == -1:
                    continue
                pred_null = pred_map[a] == -1
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
            f1 = (
                2 * prec * rec / (prec + rec + 1e-12)
                if (TP + FP + FN) > 0
                else 0.0
            )
            unified = correct / total if total > 0 else 0.0

            # ---------- Structural Edge Consistency (Option A) ----------
            # Build undirected unique edge sets
            edgeA_list = A.edge_index.t().tolist()
            edgeB_list = B.edge_index.t().tolist()

            edgeA_undirected = set()
            for u, v in edgeA_list:
                if u == v:
                    continue
                if (v, u) in edgeA_undirected:
                    continue
                edgeA_undirected.add((int(u), int(v)))

            edgeB_undirected = set()
            for u, v in edgeB_list:
                if u == v:
                    continue
                if (v, u) in edgeB_undirected:
                    continue
                edgeB_undirected.add((int(u), int(v)))

            struct_total = 0
            struct_consistent = 0
            for ai, aj in edgeA_undirected:
                if ai not in pred_map or aj not in pred_map:
                    continue
                bi = pred_map[ai]
                bj = pred_map[aj]
                if bi == -1 or bj == -1:
                    continue
                struct_total += 1
                if (bi, bj) in edgeB_undirected or (bj, bi) in edgeB_undirected:
                    struct_consistent += 1

            struct_consistency = (
                struct_consistent / struct_total if struct_total > 0 else 0.0
            )

            if top1 is not None:
                all_top1.append(top1)
                all_top5.append(top5)
                all_f1.append(f1)
                all_unified.append(unified)
            all_struct_consistency.append(struct_consistency)

            # ---------- Per-model console print ----------
            print(
                f"[Model {key}]  "
                f"Top1={100*(top1 or 0):.1f}%  "
                f"Top5={100*(top5 or 0):.1f}%  "
                f"NULL-F1={100*f1:.1f}%  "
                f"Unified={100*unified:.1f}%  "
                f"StructCons={100*struct_consistency:.1f}%"
            )

            max_s_list = max_s.tolist()

            per_model_result = {
                "model_id": key,
                "hungarian_used": use_hungarian,
                "metric_details": {
                    "top1_accuracy": float(top1) if top1 is not None else None,
                    "top5_accuracy": float(top5) if top5 is not None else None,
                    "null_f1": float(f1),
                    "unified_accuracy": float(unified),
                    "null_threshold": float(thr),
                    "TP": int(TP),
                    "FP": int(FP),
                    "FN": int(FN),
                    "total_faces": int(total),
                    "structural_edge_consistency": float(struct_consistency),
                },
                "A_ids": [int(x) for x in A.xt],
                "B_ids": [int(x) for x in B.xt],
                "max_similarity_per_A": max_s_list,
                "predicted_matches": [
                    {
                        "A_index": a,
                        "A_xt_id": int(A.xt[a]),
                        "pred_B_index": None
                        if pred_map[a] == -1
                        else int(pred_map[a]),
                        "pred_B_xt_id": None
                        if pred_map[a] == -1
                        else int(B.xt[pred_map[a]]),
                        "similarity": float(max_s[a]),
                        "is_null": bool(pred_map[a] == -1),
                    }
                    for a, _ in m.tolist()
                    if a != -1
                ],
            }

            with open(f"model_pairs/model_{key}.json", "w") as f:
                json.dump(per_model_result, f, indent=2)

    print("\n========================================")
    print("TEST RESULTS (AGGREGATED)")
    print("========================================")
    print(f"Hungarian:                    {use_hungarian}")
    print(f"Top-1 Accuracy (per-model):   {100*np.mean(all_top1):.2f}%")
    print(f"Top-5 Accuracy (per-model):   {100*np.mean(all_top5):.2f}%")
    print(f"NULL F1 Score (per-model):    {100*np.mean(all_f1):.2f}%")
    print(f"Unified Accuracy (per-model): {100*np.mean(all_unified):.2f}%")
    print(f"Struct Edge Consistency:      {100*np.mean(all_struct_consistency):.2f}%")
    print("========================================")


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
