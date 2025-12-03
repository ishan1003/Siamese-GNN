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
    "xt_path": r"C:\Users\Z0054udc\Downloads\Siamese GNN\XT_merged_Synthetic.json",

    "use_features": [0,1,2,3,15],  # 16 features available, using selected 5

    "proj_dim": 64,
    "encoder_hidden": 64,
    "encoder_out": 32,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    "epochs": 1500,
    "grad_accum_steps": 8,

    "temperature": 0.1,
    "null_margin": 0.2,
    "null_weight": 1.0,

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
            for a,b in edges:
                if int(a) in id_to_idx and int(b) in id_to_idx:
                    ai = id_to_idx[int(a)]
                    bi = id_to_idx[int(b)]
                    out.append([ai,bi])
                    out.append([bi,ai])
            if len(out)==0:
                return torch.empty((2,0), dtype=torch.long)
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
            nn.Linear(out_dim + proj_dim, 64),
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

        # classifier features
        null_feat_A = torch.cat([hA, zA], dim=1)
        null_feat_B = torch.cat([hB, zB], dim=1)

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
#                  PREDICT FUNCTION
# ---------------------------------------------------
def predict(z1, z2, thr):
    """
    Predict for each A face the best B face based on similarity threshold.
    If max similarity < thr → predict NULL.
    """
    if z2.shape[0] == 0:
        # if model-B has 0 faces
        return torch.full((z1.shape[0],), -1, dtype=torch.long), None

    sims = z1 @ z2.t()            # [N1, N2] cosine similarities
    max_s, idx = sims.max(dim=1)  # best similarity & index for each A

    # If similarity is below threshold → NULL
    pred = torch.where(max_s > thr, idx, torch.full_like(idx, -1))

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

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    bce = nn.BCEWithLogitsLoss()

    loss_hist=[]; top1_hist=[]; top5_hist=[]; null_hist=[]

    best=-1
    grad_acc=CONFIG["grad_accum_steps"]

    for ep in range(CONFIG["epochs"]):
        model.train()
        total_loss=0
        all_pos=[]; all_null=[]

        opt.zero_grad()

        for i in tqdm(range(len(ds)), desc=f"Epoch {ep+1}/{CONFIG['epochs']}"):

            A,B,m = ds[i]
            A=A.to(device); B=B.to(device); m=m.to(device)

            hA,hB,zA,zB, nullA_logits, nullB_logits = model(A,B)

            pos = m[(m[:,0]!=-1)&(m[:,1]!=-1)]

            # InfoNCE
            l1 = info_nce(zA,zB,pos, CONFIG["temperature"])
            l2 = info_nce(zB,zA,pos[:,[1,0]], CONFIG["temperature"]) if pos.numel()>0 else 0
            info = 0.5*(l1+l2)

            # NULL labels
            labels_A = torch.zeros(hA.shape[0], device=device)
            labels_B = torch.zeros(hB.shape[0], device=device)

            for a,b in m.tolist():
                if a!=-1 and b==-1:
                    labels_A[a] = 1
                if a==-1 and b!=-1:
                    labels_B[b] = 1

            null_cls_loss = (
                bce(nullA_logits, labels_A) +
                bce(nullB_logits, labels_B)
            ) * CONFIG["null_weight"]

            loss = (info + null_cls_loss)/grad_acc
            loss.backward()

            total_loss += loss.item()*grad_acc

            t1,t5,ps,ns = compute_top1_top5(zA,zB,m)
            all_pos += ps
            all_null += ns

            if (i+1)%grad_acc==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
                opt.step()
                opt.zero_grad()

        # threshold estimation
        mp = np.mean(all_pos) if len(all_pos)>0 else 0.5
        mn = np.mean(all_null) if len(all_null)>0 else -0.5
        thr = float(np.clip((mp+mn)/2, -0.9,0.9))

        # evaluate
        model.eval()
        E1=[]; E5=[]; EN=[]

        with torch.no_grad():
            for i in range(len(ds)):
                A,B,m = ds[i]
                A=A.to(device); B=B.to(device); m=m.to(device)
                _,_,zA,zB,_,_ = model(A,B)

                t1,t5,ps,ns = compute_top1_top5(zA,zB,m)

                if t1 is not None:
                    E1.append(100*t1)
                    E5.append(100*t5)

                # NULL metrics
                preds,_ = predict(zA,zB,thr)
                TP=FP=FN=0
                for a,b in m.tolist():
                    pr = int(preds[int(a)]) if a!=-1 else 0
                    if b==-1:
                        if pr==-1: TP+=1
                        else: FN+=1
                    else:
                        if pr==-1: FP+=1

                prec=TP/(TP+FP+1e-12)
                rec =TP/(TP+FN+1e-12)
                F1 =2*prec*rec/(prec+rec+1e-12)

                EN.append(100*F1)

        avg1=np.mean(E1)
        avg5=np.mean(E5)
        avgN=np.mean(EN)
        ep_loss = total_loss/len(ds)

        loss_hist.append(ep_loss)
        top1_hist.append(avg1)
        top5_hist.append(avg5)
        null_hist.append(avgN)

        print(f"Epoch {ep+1:03d}  Loss={ep_loss:.4f}  Top1={avg1:.2f}%  Top5={avg5:.2f}%  NullF1={avgN:.2f}%")

        if avg1>best:
            best=avg1
            torch.save({
                "model_state":model.state_dict(),
                "optimizer_state":opt.state_dict(),
                "config":CONFIG,
                "feat_mean": ds.feat_mean.cpu().numpy().tolist(),
                "feat_std":  ds.feat_std.cpu().numpy().tolist(),
            }, CONFIG["save_path"])

    print("Training completed. Best Top1:", best)

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
