# ================================
# train_full_with_infonce_null_classifier.py
# ================================

import os
import json
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import SAGEConv

# -------------------------
# ====== CONFIG ===========
# -------------------------
CONFIG = {
    "xt_path": r"C:\Users\Z0054udc\Downloads\Siamese GNN\XT_merged_Synthetic_cleaned.json",

    "use_features": [0,1,2,3,4,5,6,20,21,22],  # indices of features to use

    "proj_dim": 64,
    "encoder_hidden": 64,
    "encoder_out": 32,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    "epochs": 130,
    "grad_accum_steps": 8,

    "temperature": 0.1,
    "null_margin": 0.2,
    "null_weight": 1.2,
    "struct_weight": 20,   # Increase it to make it stronger


    "device": "cpu",
    "seed": 42,

    "save_path": "siamese_infonce_null_classifier.pt",
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ---------------------------------------------------
#                DATASET
# ---------------------------------------------------
class SingleFileEmbeddingPairDataset(Dataset):
    def __init__(self, json_path, feature_idx):
        super().__init__(os.path.dirname(json_path))

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.keys = sorted(self.data.keys(), key=lambda x: int(x))
        self.feature_idx = feature_idx

        # compute mean/std from whole dataset
        all_feats = []
        for key in self.keys:
            pair = self.data[key]
            for _, v in pair.get("A_embeddings", {}).items():
                all_feats.append(v)
            for _, v in pair.get("B_embeddings", {}).items():
                all_feats.append(v)

        arr = np.array(all_feats, dtype=np.float32)[:, feature_idx]
        self.feat_mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        self.feat_std =  torch.tensor(arr.std(axis=0) + 1e-9, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.data[key]

        # ---------- A ----------
        A_ids = sorted(pair["A_embeddings"].keys(), key=lambda x: int(x))
        xA = torch.tensor([pair["A_embeddings"][aid] for aid in A_ids], dtype=torch.float32)
        xA = xA[:, self.feature_idx]
        xA = (xA - self.feat_mean) / self.feat_std

        # ---------- B ----------
        B_ids = sorted(pair["B_embeddings"].keys(), key=lambda x: int(x))
        xB = torch.tensor([pair["B_embeddings"][bid] for bid in B_ids], dtype=torch.float32)
        xB = xB[:, self.feature_idx]
        xB = (xB - self.feat_mean) / self.feat_std

        # ---------- edges ----------
        def convert_edges(edges, id_list):
            id_to_idx = {int(id_): i for i, id_ in enumerate(id_list)}
            out = []
            for e in edges:

                # ---------- detect malformed edge ----------
                if not isinstance(e, (list, tuple)) or len(e) != 2:
                    print(f"\n❌ Malformed edge in model {key}: {e}")
                    continue

                a, b = e

                # ---------- skip edges whose nodes don't exist ----------
                if int(a) not in id_to_idx or int(b) not in id_to_idx:
                    # debug print if you want:
                    # print(f"⚠️ Edge references missing node in model {key}: {e}")
                    continue

                ai = id_to_idx[int(a)]
                bi = id_to_idx[int(b)]
                out.append([ai, bi])
                out.append([bi, ai])

            if len(out) == 0:
                return torch.empty((2, 0), dtype=torch.long)

            return torch.tensor(out, dtype=torch.long).t().contiguous()



        A_edges = convert_edges(pair.get("A_edges", []), A_ids)
        B_edges = convert_edges(pair.get("B_edges", []), B_ids)

        # ---------- mappings ----------
        A_map = {int(a): i for i, a in enumerate(A_ids)}
        B_map = {int(b): i for i, b in enumerate(B_ids)}

        matches = []
        for a,b in pair["mappings"]:
            ia = -1 if a=="NULL" else A_map.get(int(a), -1)
            ib = -1 if b=="NULL" else B_map.get(int(b), -1)
            matches.append([ia,ib])
        matches = torch.tensor(matches, dtype=torch.long)

        return (
            Data(x=xA, edge_index=A_edges, xt=A_ids),
            Data(x=xB, edge_index=B_edges, xt=B_ids),
            matches
        )


# ---------------------------------------------------
#                MODEL
# ---------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, hid)
        self.g3 = SAGEConv(hid, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.g1(x,edge_index))
        x = F.relu(self.g2(x,edge_index))
        x = self.g3(x,edge_index)
        return x


class SiameseGNN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid, out_dim)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # -------- NULL classifier head (fixed) --------
        self.null_head = nn.Sequential(
            nn.Linear(out_dim + proj_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, A, B):
        # encoder embeddings
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)

        # projected embeddings
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)

        if zB.shape[0] > 0:
            sims_AB = zA @ zB.t()              # [NA, NB]
            maxA = sims_AB.max(dim=1).values
            meanA = sims_AB.mean(dim=1)
            sim_gap_A = maxA - meanA
        else:
            sim_gap_A = torch.zeros(zA.size(0), device=zA.device)

        if zA.shape[0] > 0:
            sims_BA = zB @ zA.t()              # [NB, NA]
            maxB = sims_BA.max(dim=1).values
            meanB = sims_BA.mean(dim=1)
            sim_gap_B = maxB - meanB
        else:
            sim_gap_B = torch.zeros(zB.size(0), device=zB.device)



        # classifier features
        null_feat_A = torch.cat([hA, zA, sim_gap_A.unsqueeze(1)], dim=1)
        null_feat_B = torch.cat([hB, zB, sim_gap_B.unsqueeze(1)], dim=1)



        null_logits_A = self.null_head(null_feat_A).squeeze(1)
        null_logits_B = self.null_head(null_feat_B).squeeze(1)

        return (
            F.normalize(hA, dim=1),
            F.normalize(hB, dim=1),
            zA, zB,
            null_logits_A, null_logits_B
        )


# ---------------------------------------------------
#      LOSSES
# ---------------------------------------------------
def info_nce(z1, z2, matches_pos, tau=0.1):
    if matches_pos.numel()==0:
        return z1.sum()*0.0
    sims = (z1 @ z2.t()) / tau
    anchors = matches_pos[:,0].long()
    targets = matches_pos[:,1].long()
    return F.cross_entropy(sims[anchors], targets)


# ---------------------------------------------------
#      INFO-NCE (masked identical pairs)
# ---------------------------------------------------
def info_nce_weighted(z1, z2, matches_pos, tau=0.1, sim_eps=0.995):
    if matches_pos.numel() == 0:
        return z1.sum() * 0.0

    anchors = matches_pos[:, 0].long()
    targets = matches_pos[:, 1].long()

    pos_sim = (z1[anchors] * z2[targets]).sum(dim=1)

    # weights: anchors vs learners
    weights = torch.ones_like(pos_sim)
    weights[pos_sim > sim_eps] = 0.1   # unmodified → anchor

    sims = (z1 @ z2.t()) / tau
    loss = F.cross_entropy(
        sims[anchors],
        targets,
        reduction="none"
    )

    return (loss * weights).sum() / weights.sum()



# ---------------------------------------------------
#   Structural Consistency Loss
# ---------------------------------------------------

def structural_consistency_loss(hA, hB, edgeA, edgeB, pred_map):
    edgeA_list = edgeA.t().tolist()
    edgeB_set = set(map(tuple, edgeB.t().tolist()))

    pairs = []
    for ai, aj in edgeA_list:
        bi = pred_map.get(int(ai), -1)
        bj = pred_map.get(int(aj), -1)
        if bi != -1 and bj != -1 and (bi, bj) in edgeB_set:
            pairs.append((aj, bj))

    if not pairs:
        return hA.sum() * 0.0   # zero but keeps graph connected

    idxA = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=hA.device)
    idxB = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=hB.device)

    # Stronger + normalized loss
    return (1 - F.cosine_similarity(hA[idxA], hB[idxB]).mean())

# ---------------------------------------------------
#       NULL THRESHOLD CALIBRATION
# ---------------------------------------------------

def calibrate_null_threshold(probs, labels):
    """
    probs: list of predicted NULL probabilities
    labels: list of GT NULL labels (1 = NULL, 0 = NOT NULL)
    """
    best_thr = 0.5
    best_f1 = -1

    for t in np.linspace(0.05, 0.95, 37):
        TP = FP = FN = 0
        for p, y in zip(probs, labels):
            pred_null = (p > t)
            if y == 1 and pred_null: TP += 1
            if y == 1 and not pred_null: FN += 1
            if y == 0 and pred_null: FP += 1

        prec = TP / (TP + FP + 1e-12)
        rec  = TP / (TP + FN + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    return best_thr, best_f1

# ---------------------------------------------------
#       FIND BEST NULL THRESHOLD
# ---------------------------------------------------

def find_best_null_threshold(probs, gt):
    best_thr = 0.5
    best_f1 = 0.0

    for thr in np.linspace(0.1, 0.9, 81):
        TP = FP = FN = 0
        for p, y in zip(probs, gt):
            pred = p > thr
            if y == 1 and pred: TP += 1
            if y == 1 and not pred: FN += 1
            if y == 0 and pred: FP += 1

        prec = TP / (TP + FP + 1e-12)
        rec  = TP / (TP + FN + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


# ---------------------------------------------------
#                  PREDICT FUNCTION
# ---------------------------------------------------
def predict(z1, z2, thr, margin=0.05):
    if z2.shape[0] == 0:
        return torch.full((z1.shape[0],), -1), None

    sims = z1 @ z2.t()
    max_s, idx = sims.max(dim=1)

    pred = torch.where(
        max_s > (thr + margin),
        idx,
        torch.full_like(idx, -1)
    )

    return pred, max_s



# ---------------------------------------------------
#   Metrics + threshold
# ---------------------------------------------------
def compute_top1_top5(z1, z2, matches):
    d = torch.cdist(z1, z2)

    validA = []
    validB = []
    for a,b in matches.tolist():
        if a!=-1 and b!=-1:
            validA.append(a); validB.append(b)

    if len(validA)==0:
        return None, None, [], []

    validA = torch.tensor(validA)
    validB = torch.tensor(validB)

    # top1
    minidx = d.argmin(dim=1)[validA]
    top1 = (minidx==validB).float().mean().item()

    # top5
    k = min(5, z2.shape[0])
    topk = torch.topk(-d[validA], k, dim=1).indices
    top5 = torch.any(topk==validB.unsqueeze(1), dim=1).float().mean().item()

    # pos sims
    pos_s = (z1[validA] * z2[validB]).sum(dim=1).detach().cpu().tolist()

    # null sims
    nullA = [a for a,b in matches.tolist() if b==-1]
    null_s = []
    if len(nullA)>0:
        sims = z1[nullA] @ z2.t()
        null_s = sims.max(dim=1).values.detach().cpu().tolist()

    return top1, top5, pos_s, null_s


# ---------------------------------------------------
#                TRAIN LOOP
# ---------------------------------------------------
def train():
    device = torch.device(CONFIG["device"])
    ds = SingleFileEmbeddingPairDataset(CONFIG["xt_path"], CONFIG["use_features"])

    print(f"Found {len(ds)} pairs")

    model = SiameseGNN(
        in_dim=len(CONFIG["use_features"]),
        hid=CONFIG["encoder_hidden"],
        out_dim=CONFIG["encoder_out"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    bce = nn.BCEWithLogitsLoss()

    loss_hist=[]; top1_hist=[]; top5_hist=[]; null_hist=[]
    best=-1
    grad_acc=CONFIG["grad_accum_steps"]

    # initial NULL threshold (will be calibrated)
    NULL_PROB_TRAIN = 0.5

    for ep in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0

        all_pos=[]; all_null=[]
        all_null_probs=[]; all_null_gt=[]

        opt.zero_grad()

        for i in tqdm(range(len(ds)), desc=f"Epoch {ep+1}/{CONFIG['epochs']}"):

            A,B,m = ds[i]
            A=A.to(device); B=B.to(device); m=m.to(device)

            hA,hB,zA,zB,logA,logB = model(A,B)

            # ---------- NULL probability ----------
            p_null_A = torch.sigmoid(logA)

            

            # ---------- InfoNCE ----------
            pos = m[(m[:,0]!=-1)&(m[:,1]!=-1)]
            l1 = info_nce_weighted(zA, zB, pos, CONFIG["temperature"])
            l2 = info_nce_weighted(zB, zA, pos[:, [1,0]], CONFIG["temperature"])
            info_loss = 0.5 * (l1 + l2)

            # --- detect unmodified faces ---
            identical_A = torch.zeros(hA.size(0), device=device)

            for a, b in pos.tolist():
                sim = (zA[a] * zB[b]).sum()
                if sim > 0.995:
                    identical_A[a] = 1

            identical_set = set(torch.where(identical_A == 1)[0].tolist())


            identical_B = torch.zeros(hB.size(0), device=device)

            if zA.shape[0] > 0:
                sims_BA = zB @ zA.t()  # [NB, NA]

                top2 = sims_BA.topk(k=min(2, sims_BA.size(1)), dim=1).values
                max_sim = top2[:, 0]
                second_sim = top2[:, 1] if top2.size(1) > 1 else torch.zeros_like(max_sim)

                identical_B = (
                    (max_sim > 0.98) &
                    ((max_sim - second_sim) > 0.1)
                ).float()



            # ---------- NULL classification ----------
            labels_A = torch.zeros(hA.size(0), device=device)
            labels_B = torch.zeros(hB.size(0), device=device)
            for a,b in m.tolist():
                if a!=-1 and b==-1: labels_A[a]=1
                if a==-1 and b!=-1: labels_B[b]=1

            null_weights_A = torch.ones_like(labels_A)
            null_weights_A[identical_A == 1] = 0.05  # anchors

            null_loss_A = (
                F.binary_cross_entropy_with_logits(
                    logA, labels_A, reduction="none"
                ) * null_weights_A
            ).mean()

            null_weights_B = torch.ones_like(labels_B)
            null_weights_B[identical_B == 1] = 0.05   # anchors

            null_loss_B = (
                F.binary_cross_entropy_with_logits(
                    logB, labels_B, reduction="none"
                ) * null_weights_B
            ).mean()


            null_loss = null_loss_A + null_loss_B

            # ---------- STRUCTURAL LOSS (classifier-aware) ----------
            pred_map = {}
            sims = zA @ zB.t() if zB.shape[0] > 0 else None
            conf_map = {}  # confidence weight for structure

            for a, _ in m.tolist():
                if a == -1:
                    continue

                null_prob = p_null_A[a].item()

                # ---- SAFETY: no B nodes exist ----
                if sims is None:
                    pred_map[a] = -1
                    continue

                if null_prob > NULL_PROB_TRAIN:
                    pred_map[a] = -1
                    continue

                sim_row = sims[a]

                top2 = sim_row.topk(min(2, sim_row.numel())).values
                gap = top2[0] - (top2[1] if top2.numel() > 1 else 0.0)

                if gap < 0.05:
                    pred_map[a] = -1
                    continue

                pred_map[a] = int(sim_row.argmax())
                conf_map[a] = (1.0 - null_prob) * gap
           

            edgeA = A.edge_index.t().tolist()
            edgeB_set = set(tuple(x) for x in B.edge_index.t().tolist())

            struct_pairs=[]
            for ai,aj in edgeA:
                if ai in identical_set or aj in identical_set:
                    continue
                bi = pred_map.get(ai,-1)
                bj = pred_map.get(aj,-1)
                if bi!=-1 and bj!=-1 and (bi,bj) in edgeB_set:
                    w = conf_map.get(ai, 0.0) * conf_map.get(aj, 0.0)
                    if w > 0:
                        struct_pairs.append((aj, bj, w))


            if struct_pairs:
                idxA = torch.tensor([p[0] for p in struct_pairs], device=device)
                idxB = torch.tensor([p[1] for p in struct_pairs], device=device)
                weights = torch.tensor([p[2] for p in struct_pairs], device=device)

                sims_struct = F.cosine_similarity(hA[idxA], hB[idxB])
                struct_loss = 1 - (weights * sims_struct).sum() / (weights.sum() + 1e-6)

            else:
                struct_loss = torch.tensor(0., device=device)

            struct_w = CONFIG["struct_weight"] if ep >= 10 else 0.0

            total = (
                info_loss +
                CONFIG["null_weight"] * null_loss +
                struct_w * struct_loss
            ) / grad_acc


            total.backward()
            total_loss += total.item()*grad_acc

            # ---------- stats ----------
            t1,t5,ps,ns = compute_top1_top5(zA,zB,m)
            all_pos+=ps; all_null+=ns

            for a,b in m.tolist():
                if a!=-1:
                    all_null_probs.append(p_null_A[a].item())
                    all_null_gt.append(1 if b==-1 else 0)

            if (i+1)%grad_acc==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad()

        # ---------- CALIBRATE NULL THRESHOLD ----------
        NULL_PROB_TRAIN, null_f1 = calibrate_null_threshold(
            all_null_probs, all_null_gt
        )

        print(f"  🔧 Calibrated NULL threshold: {NULL_PROB_TRAIN:.3f} (F1={null_f1:.3f})")


        # ---------- validation ----------
        model.eval()
        E1,E5,EN=[],[],[]

        mp = np.mean(all_pos) if all_pos else 0.5
        mn = np.mean(all_null) if all_null else -0.5
        thr = float(np.clip((mp+mn)/2, -0.9, 0.9))

        with torch.no_grad():
            for i in range(len(ds)):
                A,B,m = ds[i]
                A=A.to(device); B=B.to(device); m=m.to(device)
                _,_,zA,zB,_,_ = model(A,B)

                t1,t5,_,_ = compute_top1_top5(zA,zB,m)
                if t1 is not None:
                    E1.append(100*t1)
                    E5.append(100*t5)

                logitsA = model.null_head(
                    torch.cat([
                        model.encoder(A.x, A.edge_index),
                        zA,
                        torch.zeros(zA.size(0), 1, device=device)  # sim_gap not needed here
                    ], dim=1)
                ).squeeze(1)

                p_null = torch.sigmoid(logitsA)

                preds = torch.full((zA.size(0),), -1, device=device)

                if zB.shape[0] > 0:
                    sims = zA @ zB.t()
                    best_match = sims.argmax(dim=1)
                    for a in range(zA.size(0)):
                        if p_null[a] <= NULL_PROB_TRAIN:
                            preds[a] = best_match[a]

                TP=FP=FN=0
                for a,b in m.tolist():
                    if a==-1: continue
                    pr = preds[a].item()
                    if b==-1:
                        TP += int(pr==-1)
                        FN += int(pr!=-1)
                    else:
                        FP += int(pr==-1)

                prec = TP/(TP+FP+1e-12)
                rec  = TP/(TP+FN+1e-12)
                EN.append(100*(2*prec*rec/(prec+rec+1e-12)))

        avg1, avg5, avgN = np.mean(E1), np.mean(E5), np.mean(EN)
        ep_loss = total_loss/len(ds)

        loss_hist.append(ep_loss)
        top1_hist.append(avg1)
        top5_hist.append(avg5)
        null_hist.append(avgN)

        print(
            f"Epoch {ep+1:03d} "
            f"Loss={ep_loss:.4f} "
            f"Top1={avg1:.2f}% "
            f"Top5={avg5:.2f}% "
            f"NullF1={avgN:.2f}%"
        )


        if avg1 > best:
            best = avg1
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": CONFIG,
                "feat_mean": ds.feat_mean.cpu().tolist(),
                "feat_std": ds.feat_std.cpu().tolist(),
                "null_prob_threshold": float(NULL_PROB_TRAIN)
            }, CONFIG["save_path"])

    print("\nTraining Completed — Best Top1:", best)




    # plots
    plt.figure(figsize=(12,5))
    plt.plot(loss_hist, color="red", label="Loss")
    plt.legend(); plt.grid(); plt.title("Training Loss"); plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(top1_hist, label="Top1")
    plt.plot(top5_hist, label="Top5")
    plt.plot(null_hist, label="Null F1")
    plt.legend(); plt.grid(); plt.title("Metrics"); plt.show()


if __name__ == "__main__":
    train()
