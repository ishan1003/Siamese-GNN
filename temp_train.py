# ================================
# train_full_with_infonce_null.py
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
    "xt_path": r"C:\Users\Z0054udc\Downloads\Siamese GNN\XT_merged_new1_doubled_fixed.json",

    # YOU CONFIRMED: 16 features exist, feature index 15 = FACE TYPE
    "use_features": [0,1,2,3,15],

    "proj_dim": 64,
    "encoder_hidden": 64,
    "encoder_out": 32,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    "epochs": 500,
    "grad_accum_steps": 8,

    "temperature": 0.1,
    "null_margin": 0.2,
    "null_weight": 0.5,

    "device": "cpu",
    "seed": 42,

    "save_path": "siamese_infonce_null.pt",
}
# -------------------------

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ---------------------------------------------------
#                DATASET
# ---------------------------------------------------
class SingleFileEmbeddingPairDataset(Dataset):
    def __init__(self, json_path, feature_idx=None):
        super().__init__(os.path.dirname(json_path))

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.keys = sorted(self.data.keys(), key=lambda x: int(x))
        self.feature_idx = feature_idx if feature_idx is not None else list(range(5))

        # gather all embeddings for global normalization
        all_feats = []
        for key in self.keys:
            pair = self.data[key]
            for _, v in pair.get("A_embeddings", {}).items():
                all_feats.append(v)
            for _, v in pair.get("B_embeddings", {}).items():
                all_feats.append(v)

        arr = np.array(all_feats, dtype=np.float32)[:, self.feature_idx]
        self.feat_mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        self.feat_std = torch.tensor(arr.std(axis=0) + 1e-9, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.data[key]

        # A embeddings
        A_ids = sorted(pair["A_embeddings"].keys(), key=lambda x: int(x))
        xA = np.array([pair["A_embeddings"][aid] for aid in A_ids], dtype=np.float32)
        xA = torch.tensor(xA[:, self.feature_idx], dtype=torch.float32)
        xA = (xA - self.feat_mean) / self.feat_std

        # B embeddings
        B_ids = sorted(pair["B_embeddings"].keys(), key=lambda x: int(x))
        xB = np.array([pair["B_embeddings"][bid] for bid in B_ids], dtype=np.float32)
        xB = torch.tensor(xB[:, self.feature_idx], dtype=torch.float32)
        xB = (xB - self.feat_mean) / self.feat_std

        # Edges: convert ID → indices
        def convert_edges(edges, id_list):
            id_to_idx = {int(id_): i for i, id_ in enumerate(id_list)}
            out = []
            for a,b in edges:
                if int(a) in id_to_idx and int(b) in id_to_idx:
                    ai = id_to_idx[int(a)]
                    bi = id_to_idx[int(b)]
                    out.append([ai, bi])
                    out.append([bi, ai])
            if len(out)==0:
                return torch.empty((2,0), dtype=torch.long)
            return torch.tensor(out, dtype=torch.long).t().contiguous()

        A_edges = convert_edges(pair.get("A_edges", []), A_ids)
        B_edges = convert_edges(pair.get("B_edges", []), B_ids)

        # Build matches tensor
        A_id_to_idx = {int(a): i for i, a in enumerate(A_ids)}
        B_id_to_idx = {int(b): i for i, b in enumerate(B_ids)}

        mappings = []
        for a,b in pair["mappings"]:
            ia = -1 if a=="NULL" else A_id_to_idx.get(int(a), -1)
            ib = -1 if b=="NULL" else B_id_to_idx.get(int(b), -1)
            mappings.append([ia, ib])
        matches = torch.tensor(mappings, dtype=torch.long)

        return (
            Data(x=xA, edge_index=A_edges, xt_entity_ids=A_ids),
            Data(x=xB, edge_index=B_edges, xt_entity_ids=B_ids),
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
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, A, B):
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)

        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)

        return F.normalize(hA,dim=1), F.normalize(hB,dim=1), zA, zB

# ---------------------------------------------------
#              LOSSES
# ---------------------------------------------------
def info_nce(z1, z2, matches_pos, tau=0.1):
    if matches_pos.numel()==0:
        return z1.sum()*0.0
    sims = torch.matmul(z1, z2.t()) / tau
    anchors = matches_pos[:,0].long()
    targets = matches_pos[:,1].long()
    return F.cross_entropy(sims[anchors], targets)

def null_loss(z_null, z_other, margin=0.2):
    if z_null.numel()==0:
        return z_null.sum()*0.0
    sims = (z_null @ z_other.t())
    max_s = sims.max(dim=1).values
    return F.softplus(max_s - margin).mean()

# ---------------------------------------------------
#         METRICS + THRESHOLD
# ---------------------------------------------------
def predict(z1,z2,thr):
    if z2.shape[0]==0:
        return torch.full((z1.shape[0],), -1, dtype=torch.long)
    sims = z1 @ z2.t()
    max_s, idx = sims.max(dim=1)
    return torch.where(max_s > thr, idx, torch.full_like(idx,-1)), max_s

def compute_top1_top5(z1,z2,matches,k=5):
    d = torch.cdist(z1,z2)
    validA=[]; validB=[]
    for a,b in matches.tolist():
        if b!=-1:
            validA.append(a)
            validB.append(b)

    if len(validA)==0 or z2.shape[0]==0:
        return None, None,[],[]

    validA = torch.tensor(validA)
    validB = torch.tensor(validB)

    minidx = d.argmin(dim=1)[validA]
    top1 = (minidx.cpu()==validB).float().mean().item()

    actual_k = min(k, z2.shape[0])
    topk = torch.topk(-d[validA], actual_k, dim=1).indices
    top5 = torch.any(topk==validB.unsqueeze(1),dim=1).float().mean().item()

    pos_sims = (z1[validA] * z2[validB]).sum(dim=1).cpu().tolist()
    null_sims=[]
    nullA=[a for a,b in matches.tolist() if b==-1]
    if len(nullA)>0:
        sims = z1[nullA] @ z2.t()
        null_sims = sims.max(dim=1).values.cpu().tolist()

    return top1, top5, pos_sims, null_sims

# ---------------------------------------------------
#                TRAIN LOOP
# ---------------------------------------------------
def train():

    device = torch.device(CONFIG["device"])
    ds = SingleFileEmbeddingPairDataset(CONFIG["xt_path"], CONFIG["use_features"])
    print("Found", len(ds), "pairs")

    model = SiameseGNN(
        in_dim=len(CONFIG["use_features"]),
        hid=CONFIG["encoder_hidden"],
        out_dim=CONFIG["encoder_out"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best = -1
    grad_acc = CONFIG["grad_accum_steps"]

    for ep in range(CONFIG["epochs"]):
        model.train()
        total_loss=0
        all_pos=[]
        all_null=[]
        opt.zero_grad()

        for i in tqdm(range(len(ds)), desc=f"Epoch {ep+1}/{CONFIG['epochs']}"):

            A,B,m = ds[i]
            A=A.to(device); B=B.to(device); m=m.to(device)

            hA,hB,zA,zB = model(A,B)

            pos = m[(m[:,0]!=-1)&(m[:,1]!=-1)]

            # InfoNCE (both directions)
            l1 = info_nce(zA,zB,pos, CONFIG["temperature"])
            l2 = info_nce(zB,zA,pos[:,[1,0]], CONFIG["temperature"]) if pos.numel()>0 else l1*0
            info = 0.5*(l1+l2)

            # NULL penalty
            nullA_idx = m[m[:,1]==-1, 0]
            nullB_idx = m[m[:,0]==-1, 1]
            l_nullA = null_loss(zA[nullA_idx], zB, CONFIG["null_margin"]) if nullA_idx.numel()>0 else 0
            l_nullB = null_loss(zB[nullB_idx], zA, CONFIG["null_margin"]) if nullB_idx.numel()>0 else 0
            nulltot = CONFIG["null_weight"]*(l_nullA+l_nullB)

            loss = (info + nulltot)/grad_acc
            loss.backward()

            total_loss += loss.item()*grad_acc

            # collect sims for threshold
            t1,t5,pos_s,null_s = compute_top1_top5(zA,zB,m)
            all_pos+=pos_s
            all_null+=null_s

            # step
            if (i+1)%grad_acc==0 or (i+1)==len(ds):
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
                opt.step()
                opt.zero_grad()

        # dynamic threshold
        mp = np.mean(all_pos) if len(all_pos)>0 else 0.5
        mn = np.mean(all_null) if len(all_null)>0 else -0.5
        thr = (mp+mn)/2
        thr = float(np.clip(thr, -0.9,0.9))

        # ---- Evaluate ----
        model.eval()
        E_top1=[]; E_top5=[]; E_null_f1=[]
        with torch.no_grad():
            for i in range(len(ds)):
                A,B,m = ds[i]
                A=A.to(device); B=B.to(device); m=m.to(device)
                _,_,zA,zB = model(A,B)

                top1,top5,pos_s,null_s = compute_top1_top5(zA,zB,m)
                if top1 is not None: E_top1.append(100*top1)
                if top5 is not None: E_top5.append(100*top5)

                preds,_ = predict(zA,zB,thr)

                # NULL metrics
                TP=FP=FN=0
                for a,b in m.tolist():
                    pred = int(preds[int(a)].item())
                    if b==-1:
                        if pred==-1: TP+=1
                        else: FN+=1
                    else:
                        if pred==-1: FP+=1

                prec = TP/(TP+FP+1e-12)
                rec  = TP/(TP+FN+1e-12)
                f1   = 2*prec*rec/(prec+rec+1e-12)
                E_null_f1.append(100*f1)

        avg_top1 = np.mean(E_top1)
        avg_top5 = np.mean(E_top5)
        avg_null = np.mean(E_null_f1)
        epoch_loss = total_loss/len(ds)

        print(f"Epoch {ep+1:03d}  Loss={epoch_loss:.4f}  Top1={avg_top1:.2f}%  Top5={avg_top5:.2f}%  NullF1={avg_null:.2f}%  thr={thr:.3f}")

        # save best model
        if avg_top1 > best:
            best = avg_top1
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": CONFIG
            }, CONFIG["save_path"])

    print("Training completed. Best Top1:", best)

if __name__=="__main__":
    train()
