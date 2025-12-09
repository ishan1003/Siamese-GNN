"""
train_siamese_v3_null_classifier.py

Siamese GNN v3 - final consistent version:
 - NULL decisions made by learned null_head (no manual similarity threshold)
 - Pair classifier + contrastive margin loss retained
 - Unified accuracy counts NULL predictions as correct
 - Same dataset format as your previous scripts
"""

import os
import json
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# -------------------------
# CONFIG
# -------------------------
CONFIG = {
    "json_path": r"C:\Users\Z0054udc\Downloads\Siamese GNN\XT_merged_new1.json",
    "use_features": [0,1,2,3,15],
    "embed_dim": 64,
    "proj_dim": 64,
    "lr": 1e-3,
    "wd": 1e-4,
    "epochs": 300,
    "margin": 0.4,
    "null_weight": 1.0,
    "pair_weight": 1.0,
    "contrast_weight": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "siamese_v3_null_best.pt",
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# -------------------------
# DATASET (same semantics as you had)
# -------------------------
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, feat_idx):
        with open(json_path, "r") as f:
            data = json.load(f)
        # allow both dict-of-idx and list
        if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
            self.keys = sorted(data.keys(), key=lambda x: int(x))
            self.data = data
        elif isinstance(data, list):
            self.keys = [str(i) for i in range(len(data))]
            self.data = {str(i): data[i] for i in range(len(data))}
        else:
            self.keys = list(data.keys())
            self.data = data

        self.feat_idx = feat_idx

        # compute mean/std over all node features used
        all_feats = []
        for k in self.keys:
            pair = self.data[k]
            for _, v in pair.get("A_embeddings", {}).items():
                all_feats.append(v)
            for _, v in pair.get("B_embeddings", {}).items():
                all_feats.append(v)
        arr = np.array(all_feats, dtype=np.float32)[:, feat_idx]
        self.mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(arr.std(axis=0) + 1e-9, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.data[key]

        # A
        A_ids = sorted(pair["A_embeddings"].keys(), key=lambda x: int(x))
        xA = torch.tensor([pair["A_embeddings"][aid] for aid in A_ids], dtype=torch.float32)
        xA = xA[:, self.feat_idx]
        xA = (xA - self.mean) / self.std

        # B
        B_ids = sorted(pair["B_embeddings"].keys(), key=lambda x: int(x))
        xB = torch.tensor([pair["B_embeddings"][bid] for bid in B_ids], dtype=torch.float32)
        xB = xB[:, self.feat_idx]
        xB = (xB - self.mean) / self.std

        # convert edges to edge_index 2xE
        def convert_edges(edges, id_list):
            id_to_idx = {int(id_): i for i, id_ in enumerate(id_list)}
            out = []
            for a, b in edges:
                if int(a) in id_to_idx and int(b) in id_to_idx:
                    ai = id_to_idx[int(a)]; bi = id_to_idx[int(b)]
                    out.append([ai, bi]); out.append([bi, ai])
            if len(out) == 0:
                return torch.empty((2, 0), dtype=torch.long)
            return torch.tensor(out, dtype=torch.long).t().contiguous()

        A_edge = convert_edges(pair.get("A_edges", []), A_ids)
        B_edge = convert_edges(pair.get("B_edges", []), B_ids)

        # matches: list of [ia, ib] with -1 for NULL
        A_map = {int(a): i for i, a in enumerate(A_ids)}
        B_map = {int(b): i for i, b in enumerate(B_ids)}
        matches = []
        for a, b in pair.get("mappings", []):
            ia = -1 if a == "NULL" else A_map.get(int(a), -1)
            ib = -1 if b == "NULL" else B_map.get(int(b), -1)
            matches.append([ia, ib])
        matches = torch.tensor(matches, dtype=torch.long)

        return Data(x=xA, edge_index=A_edge), Data(x=xB, edge_index=B_edge), matches


# -------------------------
# MODEL
# -------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, out_dim)


    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            # no edges: apply linear via conv's lin? fallback to linear mapping
            return self.g2(F.relu(self.g1.lin_l(x)))
        h = F.relu(self.g1(x, edge_index))
        h = self.g2(h, edge_index)
        return h


class SiameseGNNv3(nn.Module):
    def __init__(self, in_dim, embed_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, embed_dim, embed_dim)

        self.project = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # pair classifier for (A_i, B_j) pairs (used in loss only)
        self.pair_head = nn.Sequential(
            nn.Linear(2 * proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # null head predicts probability that a node is unmapped (NULL)
        self.null_head = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, A: Data, B: Data):
        hA = self.encoder(A.x, A.edge_index)     # [nA, embed_dim]
        hB = self.encoder(B.x, B.edge_index)     # [nB, embed_dim]
        zA = F.normalize(self.project(hA), dim=1)  # [nA, proj_dim]
        zB = F.normalize(self.project(hB), dim=1)  # [nB, proj_dim]
        return hA, hB, zA, zB


# -------------------------
# LOSSES & HELPERS
# -------------------------
def contrastive_margin_loss(zA, zB, pos_pairs, margin):
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=zA.device)
    sims = zA @ zB.t()   # [nA, nB]
    loss = 0.0
    cnt = 0
    for ia, ib in pos_pairs:
        if ia < 0 or ib < 0 or ia >= sims.size(0) or ib >= sims.size(1):
            continue
        pos_sim = sims[ia, ib]
        # hardest negative in B for this A
        neg_sim, _ = torch.max(torch.cat([sims[ia, :ib], sims[ia, ib+1:]]) if sims.size(1)>1 else torch.tensor([0.0], device=sims.device), dim=0)
        loss = loss + F.relu(margin + neg_sim - pos_sim)
        cnt += 1
    if cnt == 0:
        return torch.tensor(0.0, device=zA.device)
    return loss / cnt


def pair_classifier_loss(zA, zB, matches, pair_head, neg_ratio=1):
    # positive pairs from matches (ia != -1 and ib != -1)
    pos_pairs = [(int(a), int(b)) for a, b in matches.tolist() if a != -1 and b != -1]
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=zA.device)
    pos_logits = []
    for ia, ib in pos_pairs:
        feat = torch.cat([zA[ia], zB[ib]], dim=0)
        pos_logits.append(pair_head(feat))
    pos_logits = torch.cat(pos_logits, dim=0)

    # generate negative pairs (random) - ensure they're not positives
    neg_logits = []
    nA, nB = zA.size(0), zB.size(0)
    tries = 0
    while len(neg_logits) < min(len(pos_pairs)*neg_ratio, nA*nB) and tries < (len(pos_pairs)*10 + 100):
        ia = np.random.randint(0, nA)
        ib = np.random.randint(0, nB)
        if (ia, ib) not in pos_pairs:
            feat = torch.cat([zA[ia], zB[ib]], dim=0)
            neg_logits.append(pair_head(feat))
        tries += 1
    if len(neg_logits) == 0:
        return torch.tensor(0.0, device=zA.device)
    neg_logits = torch.cat(neg_logits, dim=0)

    labels_pos = torch.ones_like(pos_logits)
    labels_neg = torch.zeros_like(neg_logits)
    loss_pos = F.binary_cross_entropy_with_logits(pos_logits.squeeze(), labels_pos)
    loss_neg = F.binary_cross_entropy_with_logits(neg_logits.squeeze(), labels_neg)
    return loss_pos + loss_neg


def null_loss_from_head(zA, matches, null_head):
    # label = 1 for A nodes that map to NULL; else 0
    nA = zA.size(0)
    labels = torch.zeros(nA, device=zA.device)
    for a, b in matches.tolist():
        if a != -1 and b == -1:
            labels[int(a)] = 1.0
    logits = null_head(zA).squeeze()
    if logits.numel() == 0:
        return torch.tensor(0.0, device=zA.device)
    return F.binary_cross_entropy_with_logits(logits, labels)


# -------------------------
# PREDICTION (NULL uses null_head)
# -------------------------
def predict_using_null_head(zA, zB, null_head, device, null_threshold=0.5):
    """
    For each A node:
      - if null_prob > null_threshold -> predict -1
      - else predict best B by similarity
    Returns predictions tensor of shape [nA] with values in { -1, 0..nB-1 }
    Also returns null_probs and best_sim if required.
    """
    nA = zA.size(0)
    if zB.size(0) == 0:
        # all NULL (no B nodes)
        return torch.full((nA,), -1, dtype=torch.long, device=device), torch.sigmoid(null_head(zA).squeeze()), None

    sims = zA @ zB.t()  # [nA, nB]
    best_sim, best_idx = sims.max(dim=1)  # [nA], [nA]
    null_logits = null_head(zA).squeeze()
    null_probs = torch.sigmoid(null_logits)

    preds = torch.where(null_probs > null_threshold,
                        torch.full_like(best_idx, -1),
                        best_idx)
    return preds, null_probs.detach().cpu().numpy(), best_sim.detach().cpu().numpy()


# -------------------------
# METRICS
# -------------------------
def compute_metrics(preds, matches):
    """
    preds: tensor [nA] values in {-1, 0..nB-1}
    matches: Tensor [M,2] with entries (ia, ib) or -1
    We'll compute:
      - unified accuracy: for each A with any mapping (including NULL), whether pred == gt
      - NULL precision/recall/F1 (for A nodes that are labeled NULL)
      - Top-1 among mapped nodes (ignores NULL rows)
    """
    # build ground truth per A index
    gt_map = {}  # ia -> ib (could be -1)
    for a, b in matches.tolist():
        if a != -1:
            gt_map[int(a)] = int(b)
    if len(gt_map) == 0:
        return {"unified_acc": None, "null_f1": None, "top1": None}

    n_correct = 0
    n_total = 0
    # for null metrics
    TP = FP = FN = 0
    top1_count = 0
    top1_total = 0

    for ia, ib in gt_map.items():
        gt = ib  # could be -1
        pred = int(preds[ia].item())
        n_total += 1
        if pred == gt:
            n_correct += 1

        if gt == -1:
            # ground truth NULL
            if pred == -1:
                TP += 1
            else:
                FN += 1
        else:
            # ground truth mapped
            if pred == -1:
                FP += 1  # predicted null but actually mapped
            else:
                # top1 correct count handled by equality
                if pred == gt:
                    top1_count += 1
            top1_total += 1

    unified_acc = n_correct / n_total if n_total > 0 else None
    prec = TP / (TP + FP + 1e-12)
    rec = TP / (TP + FN + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
    top1 = top1_count / (top1_total + 1e-12) if top1_total > 0 else None

    return {"unified_acc": unified_acc, "null_f1": f1, "top1": top1}


# -------------------------
# TRAIN & EVAL
# -------------------------
def train():
    device = torch.device(CONFIG["device"])
    ds = PairDataset(CONFIG["json_path"], CONFIG["use_features"])
    print("Found", len(ds), "pairs")

    model = SiameseGNNv3(
        in_dim=len(CONFIG["use_features"]),
        embed_dim=CONFIG["embed_dim"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["wd"])

    best_unified = -1.0
    for ep in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for i in tqdm(range(len(ds)), desc=f"Train Epoch {ep}/{CONFIG['epochs']}"):
            A, B, matches = ds[i]
            A, B, matches = A.to(device), B.to(device), matches.to(device)

            hA, hB, zA, zB = model(A, B)

            # build pos pairs list for losses
            pos_pairs = [(int(a), int(b)) for a, b in matches.tolist() if a != -1 and b != -1]

            L_contrast = contrastive_margin_loss(zA, zB, pos_pairs, CONFIG["margin"]) * CONFIG["contrast_weight"]
            L_pair = pair_classifier_loss(zA, zB, matches, model.pair_head) * CONFIG["pair_weight"]
            L_null = null_loss_from_head(zA, matches, model.null_head) * CONFIG["null_weight"]

            loss = L_contrast + L_pair + L_null

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            total_loss += float(loss.item())

        # evaluate epoch
        model.eval()
        unified_accs = []
        null_f1s = []
        top1s = []
        with torch.no_grad():
            for j in range(len(ds)):
                A, B, matches = ds[j]
                A, B, matches = A.to(device), B.to(device), matches.to(device)
                hA, hB, zA, zB = model(A, B)
                preds, null_probs, best_sim = predict_using_null_head(zA, zB, model.null_head, device, null_threshold=0.5)
                metrics = compute_metrics(preds, matches)
                if metrics["unified_acc"] is not None:
                    unified_accs.append(metrics["unified_acc"])
                if metrics["null_f1"] is not None:
                    null_f1s.append(metrics["null_f1"])
                if metrics["top1"] is not None:
                    top1s.append(metrics["top1"])

        mean_unified = 100 * (np.mean(unified_accs) if len(unified_accs) > 0 else 0.0)
        mean_nullf1 = 100 * (np.mean(null_f1s) if len(null_f1s) > 0 else 0.0)
        mean_top1 = 100 * (np.mean(top1s) if len(top1s) > 0 else 0.0)

        print(f"Epoch {ep:03d} Loss={total_loss/len(ds):.6f} UnifiedAcc={mean_unified:.2f}% Top1={mean_top1:.2f}% NullF1={mean_nullf1:.2f}%")

        # save best by unified accuracy
        if mean_unified > best_unified:
            best_unified = mean_unified
            torch.save({
                "model_state": model.state_dict(),
                "config": CONFIG,
            }, CONFIG["save_path"])
            print(f"Saved best model (UnifiedAcc={mean_unified:.2f}%) -> {CONFIG['save_path']}")

    print("Training finished. Best UnifiedAcc:", best_unified)


if __name__ == "__main__":
    train()
