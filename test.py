# ======================
# test_model.py (COMPATIBLE + unified accuracy restored)
# ======================

import json, os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


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
    id_to_idx = {int(i): x for x,i in enumerate(id_list)}
    out = []
    for a,b in edges:
        if int(a) in id_to_idx and int(b) in id_to_idx:
            ai = id_to_idx[int(a)]
            bi = id_to_idx[int(b)]
            out.append([ai, bi])
            out.append([bi, ai])
    if not out:
        return torch.empty((2,0), dtype=torch.long)
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

    def __len__(self): return len(self.keys)

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

        # matches
        A_map = {int(a): i for i,a in enumerate(A_ids)}
        B_map = {int(b): i for i,b in enumerate(B_ids)}

        matches = []
        for a,b in pair["mappings"]:
            ai = -1 if a=="NULL" else A_map.get(int(a),-1)
            bi = -1 if b=="NULL" else B_map.get(int(b),-1)
            matches.append([ai,bi])

        return (
            key,
            Data(x=xA, edge_index=edgeA, xt=A_ids),
            Data(x=xB, edge_index=edgeB, xt=B_ids),
            torch.tensor(matches, dtype=torch.long),
        )


# ---------------------------------------------------
# MODEL (SAME AS TRAINING)
# ---------------------------------------------------
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, hid)
        self.g3 = SAGEConv(hid, out_dim)

    def forward(self, x, e):
        x = F.relu(self.g1(x,e))
        x = F.relu(self.g2(x,e))
        return self.g3(x,e)


class SiameseGNN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid, out_dim)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(out_dim, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim)
        )

        # from training script (even though not used here)
        self.null_head = torch.nn.Sequential(
            torch.nn.Linear(out_dim + proj_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, A, B):
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)

        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)

        # classifier (not used yet)
        featA = torch.cat([hA, zA], dim=1)
        featB = torch.cat([hB, zB], dim=1)
        nullA = self.null_head(featA).squeeze(1)
        nullB = self.null_head(featB).squeeze(1)

        return zA, zB, nullA, nullB


# ---------------------------------------------------
# METRICS + UNIFIED ACCURACY
# ---------------------------------------------------
def compute_metrics(zA, zB, matches, thr=0.5):
    sims = zA @ zB.t()
    max_s, _ = sims.max(dim=1)

    validA, validB = [], []
    for a,b in matches.tolist():
        if a!=-1 and b!=-1:
            validA.append(a)
            validB.append(b)

    if not validA:
        return None, None, 0.0, sims, max_s

    validA = torch.tensor(validA)
    validB = torch.tensor(validB)

    top1 = (torch.argmax(sims[validA], 1) == validB).float().mean().item()

    k = min(5, zB.shape[0])
    topk = torch.topk(sims[validA], k, dim=1).indices
    top5 = torch.any(topk == validB.unsqueeze(1), 1).float().mean().item()

    # NULL F1
    TP = FP = FN = 0
    for a,b in matches.tolist():
        if a==-1: continue
        pred_null = max_s[a] < thr

        if b==-1:
            TP += int(pred_null)
            FN += int(not pred_null)
        else:
            FP += int(pred_null)

    prec = TP/(TP+FP+1e-12)
    rec  = TP/(TP+FN+1e-12)
    f1   = 2*prec*rec/(prec+rec+1e-12)

    return top1, top5, f1, sims, max_s


def compute_unified_accuracy(zA, zB, matches, thr=0.5):
    sims = zA @ zB.t()
    max_s, max_idx = sims.max(dim=1)

    total = matches.shape[0]
    correct = 0

    for a,b in matches.tolist():
        pred_null = max_s[a] < thr

        if b == -1:
            correct += int(pred_null)
        else:
            correct += int((not pred_null) and max_idx[a].item() == b)

    return correct / total


# ---------------------------------------------------
# TEST LOOP
# ---------------------------------------------------
def test_model(ckpt_path, test_json):

    print("Loading checkpoint...")
    state_dict, cfg, feat_mean, feat_std = load_checkpoint(ckpt_path)

    print("Loading test dataset...")
    ds = TestDataset(test_json, cfg["use_features"], feat_mean, feat_std)

    print("Building model...")
    model = SiameseGNN(
        in_dim=len(cfg["use_features"]),
        hid=cfg["encoder_hidden"],
        out_dim=cfg["encoder_out"],
        proj_dim=cfg["proj_dim"]
    )
    model.load_state_dict(state_dict)
    model.eval()

    all_top1=[]; all_top5=[]; all_f1=[]; all_unified=[]
    os.makedirs("model_outputs", exist_ok=True)

    print("Running evaluation...\n")

    with torch.no_grad():
        for key, A, B, m in ds:
            zA, zB, nullA, nullB = model(A,B)

            t1,t5,f1,sims,max_s = compute_metrics(zA,zB,m)
            unified = compute_unified_accuracy(zA,zB,m)

            if t1 is not None:
                all_top1.append(t1)
                all_top5.append(t5)
                all_f1.append(f1)
                all_unified.append(unified)

            # per-model JSON
            out = {
                "model_id": key,
                "top1": float(t1) if t1 is not None else None,
                "top5": float(t5) if t5 is not None else None,
                "null_f1": float(f1),
                "unified_accuracy": float(unified),
                "A_ids": A.xt,
                "B_ids": B.xt,
                "max_similarity_per_A": max_s.tolist(),
                "predicted_matches": []
            }

            for i in range(zA.shape[0]):
                best_j = torch.argmax(sims[i]).item()
                best_sim = sims[i][best_j].item()
                pred_null = best_sim < 0.5

                out["predicted_matches"].append({
                    "A_index": i,
                    "A_xt_id": A.xt[i],
                    "pred_B_index": None if pred_null else best_j,
                    "pred_B_xt_id": None if pred_null else B.xt[best_j],
                    "similarity": best_sim,
                    "is_null": bool(pred_null)
                })

            with open(f"model_outputs/model_{key}.json", "w") as f:
                json.dump(out, f, indent=2)

    print("========================================")
    print("TEST RESULTS (AGGREGATED)")
    print("========================================")
    print(f"Top-1 Accuracy:      {100*np.mean(all_top1):.2f}%")
    print(f"Top-5 Accuracy:      {100*np.mean(all_top5):.2f}%")
    print(f"NULL F1 Score:       {100*np.mean(all_f1):.2f}%")
    print(f"Unified Accuracy:    {100*np.mean(all_unified):.2f}%")
    print("========================================")
    print("Per-model results saved into model_outputs/")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_json", required=True)
    args = parser.parse_args()

    test_model(args.checkpoint, args.test_json)
