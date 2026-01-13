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

    "use_features": [0,1,2,3,4,5,6,20,21,22],  # indices of features to use {0-Moment0 , (1-3)- Moment1 , 4-Moment2 , 5-Loops , 6-FaceType , (20-22)- FaceNormal}

    "proj_dim": 64,
    "encoder_hidden": 64,
    "encoder_out": 32,

    "lr": 1e-3, # learning rate
    "weight_decay": 1e-4, # weight decay used for AdamW optimizer [helps against overfitting]

    "epochs": 155, # number of training epochs
    "grad_accum_steps": 8, # gradient accumulation steps [to simulate larger batch size]

    "temperature": 0.1, # temperature for InfoNCE loss[lower = harder negatives are penalized more]
    "null_margin": 0.2, # margin for NULL classification
    "null_weight": 1.2, # weight for NULL classification loss
    "struct_weight": 20,   # weight for structural consistency loss (after 10 epochs)


    "device": "cpu",
    "seed": 42, # random seed for reproducibility

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
        
        # Load entire dataset JSON into memory
        # Structure:
        # {
        #   "0": {A_embeddings, B_embeddings, A_edges, B_edges, mappings},
        #   "1": {...},
        #   ...
        # }

        with open(json_path, "r") as f:   # Load entire JSON file
            self.data = json.load(f)

        self.keys = sorted(self.data.keys(), key=lambda x: int(x))   # Sort keys numerically for deterministic order 
        self.feature_idx = feature_idx # indices of features to use

        # ---------------------------------------------------
        # Compute GLOBAL mean and std over ALL nodes (A + B)
        # This ensures consistent feature scaling across graphs
        # ---------------------------------------------------
        all_feats = []
        for key in self.keys:
            pair = self.data[key]
            # Collect features from graph A
            for _, v in pair.get("A_embeddings", {}).items():
                all_feats.append(v)
            # Collect features from graph B
            for _, v in pair.get("B_embeddings", {}).items():
                all_feats.append(v)

         # Select only desired features and compute stats
        arr = np.array(all_feats, dtype=np.float32)[:, feature_idx]
        self.feat_mean = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        self.feat_std =  torch.tensor(arr.std(axis=0) + 1e-9, dtype=torch.float32)

    def __len__(self):
        # Return the number of pairs in the dataset
        return len(self.keys)

    def __getitem__(self, idx):
        # Fetch a single pair by index
        key = self.keys[idx]
        pair = self.data[key]

        # ---------------------------------------------------
        # Build Graph A
        # ---------------------------------------------------
        # Sort node IDs to ensure stable indexing
        A_ids = sorted(pair["A_embeddings"].keys(), key=lambda x: int(x))
        # Stack node feature vectors into a tensor
        xA = torch.tensor([pair["A_embeddings"][aid] for aid in A_ids], dtype=torch.float32)
        # Select only required features and normalize
        xA = xA[:, self.feature_idx]
        xA = (xA - self.feat_mean) / self.feat_std

        # ---------------------------------------------------
        # Build Graph B (same process as A)
        # ---------------------------------------------------
        B_ids = sorted(pair["B_embeddings"].keys(), key=lambda x: int(x))
        xB = torch.tensor([pair["B_embeddings"][bid] for bid in B_ids], dtype=torch.float32)
        xB = xB[:, self.feature_idx]
        xB = (xB - self.feat_mean) / self.feat_std

        # ---------------------------------------------------
        # Edge conversion function
        # Converts CAD node IDs → contiguous indices
        # Ensures:
        #   - No malformed edges
        #   - No missing nodes
        #   - Undirected graph (bi-directional edges)
        # ---------------------------------------------------
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


        # Convert raw edges to PyG edge_index format
        A_edges = convert_edges(pair.get("A_edges", []), A_ids)
        B_edges = convert_edges(pair.get("B_edges", []), B_ids)

        # ---------------------------------------------------
        # Build ground-truth node mappings
        # Format: [A_index, B_index]
        # -1 indicates NULL (unmatched node)
        # ---------------------------------------------------
        A_map = {int(a): i for i, a in enumerate(A_ids)}
        B_map = {int(b): i for i, b in enumerate(B_ids)}

        matches = []
        for a,b in pair["mappings"]:
            ia = -1 if a=="NULL" else A_map.get(int(a), -1)
            ib = -1 if b=="NULL" else B_map.get(int(b), -1)
            matches.append([ia,ib])
        matches = torch.tensor(matches, dtype=torch.long)

         # Return PyG Data objects + ground truth mapping
        return (
            Data(x=xA, edge_index=A_edges, xt=A_ids),
            Data(x=xB, edge_index=B_edges, xt=B_ids),
            matches
        )


# ---------------------------------------------------
#                MODEL
# ---------------------------------------------------
# ---------------------------------------------------
# GraphEncoder: learns structural node embeddings
# ---------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()

        # Three-layer GraphSAGE network
        self.g1 = SAGEConv(in_dim, hid)
        self.g2 = SAGEConv(hid, hid)
        self.g3 = SAGEConv(hid, out_dim)

    def forward(self, x, edge_index):
        # Message passing + non-linearity
        x = F.relu(self.g1(x,edge_index))
        x = F.relu(self.g2(x,edge_index))
        x = self.g3(x,edge_index)
        return x


# ---------------------------------------------------
# SiameseGNN: shared encoder for Graph A and Graph B
# ---------------------------------------------------

class SiameseGNN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        # Shared graph encoder (Siamese)
        self.encoder = GraphEncoder(in_dim, hid, out_dim)

        
        # Projection head for contrastive learning (InfoNCE)
        # Separates "matching identity" from structural embedding
        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # NULL classifier head
        # Input = [structural emb, projected emb, similarity gap]
        self.null_head = nn.Sequential(
            nn.Linear(out_dim + proj_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, A, B):
        # ---------------------------------------------------
        # Encode graphs A and B using shared GNN
        # ---------------------------------------------------
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)

        # ---------------------------------------------------
        # Project embeddings into contrastive space
        # Normalize so dot product = cosine similarity
        # ---------------------------------------------------
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)

        # ---------------------------------------------------
        # Compute similarity gap (confidence signal)
        # sim_gap = max similarity - mean similarity
        # Large gap → confident match
        # Small gap → ambiguous / NULL-like
        # ---------------------------------------------------

        if zB.shape[0] > 0:
            sims_AB = zA @ zB.t()              
            maxA = sims_AB.max(dim=1).values
            meanA = sims_AB.mean(dim=1)
            sim_gap_A = maxA - meanA
        else:
            sim_gap_A = torch.zeros(zA.size(0), device=zA.device)

        if zA.shape[0] > 0:
            sims_BA = zB @ zA.t()               
            maxB = sims_BA.max(dim=1).values
            meanB = sims_BA.mean(dim=1)
            sim_gap_B = maxB - meanB
        else:
            sim_gap_B = torch.zeros(zB.size(0), device=zB.device)



        # ---------------------------------------------------
        # Build NULL classifier input
        # ---------------------------------------------------
        null_feat_A = torch.cat([hA, zA, sim_gap_A.unsqueeze(1)], dim=1)
        null_feat_B = torch.cat([hB, zB, sim_gap_B.unsqueeze(1)], dim=1)


        # Predict NULL logits (raw scores)
        null_logits_A = self.null_head(null_feat_A).squeeze(1)
        null_logits_B = self.null_head(null_feat_B).squeeze(1)

        # Return:
        # - normalized structural embeddings (for structure loss)
        # - projected embeddings (for matching)
        # - NULL logits

        return (
            F.normalize(hA, dim=1),
            F.normalize(hB, dim=1),
            zA, zB,
            null_logits_A, null_logits_B
        )


# ---------------------------------------------------
#      LOSSES
# ---------------------------------------------------
# ---------------------------------------------------
# InfoNCE Loss (basic contrastive matching)
# Teaches: "this node in A should match that node in B"
# ---------------------------------------------------

def info_nce(z1, z2, matches_pos, tau=0.1):
    """
    z1, z2       : projected (L2-normalized) embeddings of graph A and B
    matches_pos  : tensor of shape [K, 2] containing GT matched pairs
                   (only non-NULL matches)
    tau          : temperature parameter controlling softmax sharpness
    """
    # If no valid matches exist (all nodes are NULL),
    # return a zero loss while keeping gradient flow intact
    if matches_pos.numel()==0:
        return z1.sum()*0.0
    # Pairwise similarity matrix (cosine similarity / temperature)
    # Shape: [num_nodes_A, num_nodes_B]
    sims = (z1 @ z2.t()) / tau
    # Anchors: node indices from graph A
    # Targets: corresponding correct nodes in graph B
    anchors = matches_pos[:,0].long()
    targets = matches_pos[:,1].long()

    # Cross-entropy treats each anchor as a classification task:
    # "Which B-node is the correct match?"
    return F.cross_entropy(sims[anchors], targets)


# ---------------------------------------------------
# Weighted InfoNCE Loss
# Downweights trivial (identical) matches
# Focuses learning on hard / ambiguous correspondences
# ---------------------------------------------------
def info_nce_weighted(z1, z2, matches_pos, tau=0.1, sim_eps=0.995):
    """
    z1, z2       : projected embeddings from graphs A and B
    matches_pos  : GT matched node pairs (non-NULL)
    tau          : temperature parameter
    sim_eps      : similarity threshold to detect trivial matches
    """
    # Safety: no valid matches
    if matches_pos.numel() == 0:
        return z1.sum() * 0.0

    # Extract anchor (A) and target (B) indices
    anchors = matches_pos[:, 0].long()
    targets = matches_pos[:, 1].long()

    # Compute cosine similarity for GT matched pairs
    # Because z is normalized, dot product = cosine similarity
    pos_sim = (z1[anchors] * z2[targets]).sum(dim=1)

    # Initialize all matches with full weight
    weights = torch.ones_like(pos_sim)

    # Very high similarity indicates an unmodified / trivial match
    # These should act as anchors, not dominate learning
    weights[pos_sim > sim_eps] = 0.1   

    # Compute similarity matrix for InfoNCE
    sims = (z1 @ z2.t()) / tau

    # Compute per-sample cross-entropy loss
    # (no reduction yet — we apply weights manually)
    loss = F.cross_entropy(
        sims[anchors],
        targets,
        reduction="none"
    )

    # Weighted average loss:
    # - Hard matches dominate gradients
    # - Easy matches contribute minimally
    return (loss * weights).sum() / weights.sum()



# ---------------------------------------------------
# Structural Consistency Loss
# Enforces graph-level structural alignment
# ---------------------------------------------------

def structural_consistency_loss(hA, hB, edgeA, edgeB, pred_map):
    """
    hA, hB     : structural embeddings from GraphEncoder
    edgeA      : edge_index of graph A
    edgeB      : edge_index of graph B
    pred_map   : predicted A -> B node mapping (confidence-filtered)
    """
    # Convert edge indices to Python lists for iteration
    edgeA_list = edgeA.t().tolist()

    # Use set for O(1) edge existence lookup in graph B
    edgeB_set = set(map(tuple, edgeB.t().tolist()))

    # Collect structurally consistent node pairs
    pairs = []
    for ai, aj in edgeA_list:
        # Map nodes from A to predicted nodes in B
        bi = pred_map.get(int(ai), -1)
        bj = pred_map.get(int(aj), -1)
        # Keep only confident matches where:
        # - both nodes are matched
        # - corresponding edge exists in graph B
        if bi != -1 and bj != -1 and (bi, bj) in edgeB_set:
            pairs.append((aj, bj))

    # If no valid structural correspondences exist,
    # return zero loss (safe for early training)
    if not pairs:
        return hA.sum() * 0.0   # zero but keeps graph connected

    
    # Convert matched node indices to tensors
    idxA = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=hA.device)
    idxB = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=hB.device)

    # Structural loss:
    # Encourage matched nodes to have similar structural embeddings
    # Uses cosine similarity to be scale-invariant
    return (1 - F.cosine_similarity(hA[idxA], hB[idxB]).mean())

# ---------------------------------------------------
# NULL Threshold Calibration
# Learns the optimal probability cutoff for NULL
# ---------------------------------------------------

def calibrate_null_threshold(probs, labels):
    """
    probs  : list of predicted NULL probabilities (sigmoid outputs)
    labels : list of ground-truth NULL labels
             1 = NULL node
             0 = valid (non-NULL) node

    Goal:
    -----
    Find the probability threshold that maximizes NULL F1-score.
    """
    # Default threshold (fallback)
    best_thr = 0.5
    best_f1 = -1

    # Sweep over a range of candidate thresholds
    # (Avoid extremes where everything becomes NULL or non-NULL)
    for t in np.linspace(0.05, 0.95, 37):

        # Confusion matrix counters (NULL as positive class)
        TP = FP = FN = 0
        for p, y in zip(probs, labels):
            # Predict NULL if probability exceeds threshold
            pred_null = (p > t)
            # True Positive: correctly predicted NULL
            if y == 1 and pred_null: TP += 1
            # False Negative: missed NULL
            if y == 1 and not pred_null: FN += 1
            # False Positive: incorrectly predicted NULL    
            if y == 0 and pred_null: FP += 1

        prec = TP / (TP + FP + 1e-12) 
        rec  = TP / (TP + FN + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    return best_thr, best_f1




# ---------------------------------------------------
# Prediction Function
# Converts similarity scores into final matches
# ---------------------------------------------------
def predict(z1, z2, thr, margin=0.05):
    """
    z1     : projected embeddings of graph A (normalized)
    z2     : projected embeddings of graph B (normalized)
    thr    : similarity threshold (data-driven)
    margin : safety margin to reject ambiguous matches

    Returns:
    --------
    pred   : predicted B-index for each A-node (-1 = NULL)
    max_s  : maximum similarity score per A-node
    """
    # If graph B has no nodes, all A-nodes are NULL
    if z2.shape[0] == 0:
        return torch.full((z1.shape[0],), -1), None

    # Compute cosine similarity matrix
    # Shape: [num_A, num_B]
    sims = z1 @ z2.t()
    # For each node in A, find the best matching node in B
    max_s, idx = sims.max(dim=1)

    # Accept match only if similarity is confidently above threshold
    # Otherwise mark as NULL (-1)
    pred = torch.where(
        max_s > (thr + margin),
        idx,
        torch.full_like(idx, -1)
    )

    return pred, max_s



# ---------------------------------------------------
# Top-1 / Top-5 Matching Metrics
# Also collects similarity statistics
# ---------------------------------------------------
def compute_top1_top5(z1, z2, matches):
    """
    z1, z2   : projected embeddings from graphs A and B
    matches  : ground-truth node mapping tensor
               [A_index, B_index] with -1 indicating NULL

    Returns:
    --------
    top1     : Top-1 matching accuracy (ignoring NULLs)
    top5     : Top-5 matching accuracy (ignoring NULLs)
    pos_s    : cosine similarities of true matched pairs
    null_s   : max similarity of NULL nodes (for threshold estimation)
    """

    # Compute pairwise Euclidean distance matrix
    # (Used only for ranking, not for loss)
    d = torch.cdist(z1, z2)

    # Collect valid (non-NULL) ground-truth pairs
    validA = []
    validB = []
    for a,b in matches.tolist():
        if a!=-1 and b!=-1:
            validA.append(a); validB.append(b)

    # If no valid matches exist, return empty metrics
    if len(validA)==0:
        return None, None, [], []

    validA = torch.tensor(validA)
    validB = torch.tensor(validB)

    # ---------------------------------------------------
    # Top-1 Accuracy
    # ---------------------------------------------------
    # Predict closest node in B for each A-node
    minidx = d.argmin(dim=1)[validA]
    # Percentage of correct matches
    top1 = (minidx==validB).float().mean().item()

    # ---------------------------------------------------
    # Top-5 Accuracy
    # ---------------------------------------------------
    k = min(5, z2.shape[0])
    # Get indices of top-5 closest nodes in B
    topk = torch.topk(-d[validA], k, dim=1).indices
    # Check if GT match appears in top-5
    top5 = torch.any(topk==validB.unsqueeze(1), dim=1).float().mean().item()

    # ---------------------------------------------------
    # Similarity statistics (used for thresholding)
    # ---------------------------------------------------

    # Similarities of true matched pairs
    pos_s = (z1[validA] * z2[validB]).sum(dim=1).detach().cpu().tolist()

    # For NULL nodes in A, record their best similarity to B
    nullA = [a for a,b in matches.tolist() if b==-1]
    null_s = []
    if len(nullA)>0:
        sims = z1[nullA] @ z2.t()
        null_s = sims.max(dim=1).values.detach().cpu().tolist()

    return top1, top5, pos_s, null_s


# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
def train():
    # ---------------------------------------------------
    # Device & Dataset
    # ---------------------------------------------------
    device = torch.device(CONFIG["device"])
    # Dataset returns (Graph A, Graph B, GT mappings)
    ds = SingleFileEmbeddingPairDataset(CONFIG["xt_path"], CONFIG["use_features"])

    print(f"Found {len(ds)} pairs")

    # ---------------------------------------------------
    # Model initialization
    # ---------------------------------------------------
    model = SiameseGNN(
        in_dim=len(CONFIG["use_features"]),
        hid=CONFIG["encoder_hidden"],
        out_dim=CONFIG["encoder_out"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device)

    # Optimizer (AdamW is more stable than Adam for GNNs)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    bce = nn.BCEWithLogitsLoss()
     
    # ---------------------------------------------------
    # Tracking variables
    # ---------------------------------------------------

    loss_hist=[]; top1_hist=[]; top5_hist=[]; null_hist=[] ; unified_hist=[]
    best=-1
    grad_acc=CONFIG["grad_accum_steps"]

    # initial NULL threshold (will be calibrated)
    NULL_PROB_TRAIN = 0.5

    # ===================================================
    # Epoch Loop
    # ===================================================
    for ep in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0

        # ---------------------------------------------------
        # Buffers for threshold calibration
        # ---------------------------------------------------
        all_pos = []          # similarities of true matches
        all_null = []         # max similarities of NULL nodes
        all_null_probs = []   # predicted NULL probabilities
        all_null_gt = []      # ground truth NULL labels

        opt.zero_grad()

        # ===================================================
        # Iterate over graph pairs
        # ===================================================
        for i in tqdm(range(len(ds)), desc=f"Epoch {ep+1}/{CONFIG['epochs']}"):
            # ---------------------------------------------------
            # Load one (A, B, mapping) sample
            # ---------------------------------------------------
            A,B,m = ds[i]
            A=A.to(device); B=B.to(device); m=m.to(device)

            # ---------------------------------------------------
            # Forward pass
            # ---------------------------------------------------
            hA,hB,zA,zB,logA,logB = model(A,B)

            # Convert NULL logits to probabilities
            p_null_A = torch.sigmoid(logA)

            

            # ---------------------------------------------------
            # InfoNCE (matching loss)
            # ---------------------------------------------------
            # Use only non-NULL GT matches

            pos = m[(m[:,0]!=-1)&(m[:,1]!=-1)]
            l1 = info_nce_weighted(zA, zB, pos, CONFIG["temperature"])
            l2 = info_nce_weighted(zB, zA, pos[:, [1,0]], CONFIG["temperature"])
            info_loss = 0.5 * (l1 + l2)

            # ---------------------------------------------------
            # Detect identical / unmodified nodes (A side)
            # ---------------------------------------------------
            identical_A = torch.zeros(hA.size(0), device=device)

            for a, b in pos.tolist():
                sim = (zA[a] * zB[b]).sum()
                if sim > 0.995:
                    identical_A[a] = 1

            identical_set = set(torch.where(identical_A == 1)[0].tolist())

            # ---------------------------------------------------
            # Detect identical / unmodified nodes (B side)
            # ---------------------------------------------------
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

            # ---------------------------------------------------
            # NULL classification loss
            # ---------------------------------------------------
            # Goal:
            # -----
            # Learn to decide whether a node should be:
            #   - matched to some node in the other graph (label = 0)
            #   - or declared NULL (label = 1)
            #
            # This is a binary classification problem solved using BCE loss.
            # ---------------------------------------------------
            

            # ---------------------------------------------------
            # Step 1: Build per-node weights (Graph A)
            # ---------------------------------------------------
            # Default weight = 1.0 (full learning signal)
            null_weights_A = torch.ones_like(labels_A)
            # Nodes marked as "identical / unmodified" are extremely easy cases:
            #   - They already match with very high confidence
            #   - They are not informative for learning NULL boundaries
            null_weights_A[identical_A == 1] = 0.05  # anchors

            # ---------------------------------------------------
            # Step 2: Compute weighted NULL loss for Graph A
            # ---------------------------------------------------
            # BCEWithLogitsLoss is used because:
            #   - The model outputs raw logits (logA)
            #   - It is numerically stable
            #
            # reduction="none" gives per-node loss values:
            #   loss_A[i] = loss contribution of node i

            null_loss_A = (
                F.binary_cross_entropy_with_logits(
                    logA, labels_A, reduction="none" # predicted NULL logits for A nodes and GT labels
                ) * null_weights_A  # Apply per-node weights
            ).mean() # Average over all nodes

            # ---------------------------------------------------
            # Step 3: Build per-node weights (Graph B)
            # ---------------------------------------------------

            null_weights_B = torch.ones_like(labels_B)
            # Same logic as Graph A:
            #   - Identical nodes are easy anchors
            #   - Downweight to avoid dominating training
            null_weights_B[identical_B == 1] = 0.05   # anchors

            null_loss_B = (
                F.binary_cross_entropy_with_logits(
                    logB, labels_B, reduction="none"
                ) * null_weights_B
            ).mean()


            null_loss = null_loss_A + null_loss_B

            # ---------------------------------------------------
            # STRUCTURAL CONSISTENCY LOSS (classifier-aware)
            # ---------------------------------------------------
            # Goal:
            # -----
            # Encourage matched nodes to preserve graph topology:
            #   If (A_i — A_j) is an edge in graph A,
            #   then their predicted matches (B_i — B_j)
            #   should also form an edge in graph B.

            
            # Mapping: A_index → predicted B_index (or -1 if NULL)
            pred_map = {} 
            # Pairwise similarity matrix between A and B
            # (used to choose best predicted matches)
            sims = zA @ zB.t() if zB.shape[0] > 0 else None
            # Confidence score per predicted match
            # Used to weight structural constraints
            conf_map = {}  # confidence weight for structure

            # ---------------------------------------------------
            # STEP 1: Build predicted A → B mapping
            # ---------------------------------------------------

            for a, _ in m.tolist():    # iterate over all A nodes
                if a == -1:
                    continue

                null_prob = p_null_A[a].item()   # predicted NULL probability .item()

                # ---- SAFETY: no B nodes exist ----
                if sims is None:
                    pred_map[a] = -1
                    continue

                if null_prob > NULL_PROB_TRAIN:
                    pred_map[a] = -1
                    continue
                
                # Similarity scores between A[a] and all B nodes
                sim_row = sims[a]

                # Compute top-2 similarities to measure ambiguity
                top2 = sim_row.topk(min(2, sim_row.numel())).values
                gap = top2[0] - (top2[1] if top2.numel() > 1 else 0.0)

                # Small gap ⇒ ambiguous match ⇒ unreliable structure
                if gap < 0.05:
                    pred_map[a] = -1
                    continue
                
                # Assign best match in B
                pred_map[a] = int(sim_row.argmax())
                # Confidence combines:
                #   - low NULL probability
                #   - high separation from next best match
                conf_map[a] = (1.0 - null_prob) * gap
           
            # ---------------------------------------------------
            # STEP 2: Identify structurally consistent edge pairs
            # ---------------------------------------------------
            edgeA = A.edge_index.t().tolist()
            edgeB_set = set(tuple(x) for x in B.edge_index.t().tolist())

            struct_pairs=[]
            for ai,aj in edgeA:
                # Skip trivial / identical nodes
                # They already match perfectly and add no structural signal

                if ai in identical_set or aj in identical_set:
                    continue
                bi = pred_map.get(ai,-1)
                bj = pred_map.get(aj,-1)
                 # Structural consistency condition:
                    #   (A_ai — A_aj) exists
                    #   AND
                    #   (B_bi — B_bj) also exists
                if bi!=-1 and bj!=-1 and (bi,bj) in edgeB_set:
                    # Weight is product of both nodes' confidences
                    w = conf_map.get(ai, 0.0) * conf_map.get(aj, 0.0)
                    if w > 0:
                        struct_pairs.append((aj, bj, w))

            # ---------------------------------------------------
            # STEP 3: Compute structural loss
            # ---------------------------------------------------


            if struct_pairs:
                # Extract indices and weights
                idxA = torch.tensor([p[0] for p in struct_pairs], device=device)
                idxB = torch.tensor([p[1] for p in struct_pairs], device=device)
                weights = torch.tensor([p[2] for p in struct_pairs], device=device)

                # Cosine similarity between structural embeddings
                sims_struct = F.cosine_similarity(hA[idxA], hB[idxB])
                # Weighted structural loss
                # Encourage embeddings to align when structure is preserved
                struct_loss = 1 - (weights * sims_struct).sum() / (weights.sum() + 1e-6)

            else:
                struct_loss = torch.tensor(0., device=device)

            # ---------------------------------------------------
            # STEP 4: Curriculum learning for structure
            # ---------------------------------------------------
            # Structural loss is dangerous early when predictions are noisy.
            # Enable it only after embeddings stabilize.
            struct_w = CONFIG["struct_weight"] if ep >= 10 else 0.0
            
            # ---------------------------------------------------
            # STEP 5: Total loss
            # ---------------------------------------------------
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
        E1,E5,EN,EU=[],[],[],[]

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

                TP = FP = FN = 0
                correct = 0
                total = 0

                # which B faces were claimed by any A
                predicted_B = set(preds.tolist())
                predicted_B.discard(-1)

                for a, b in m.tolist():

                    # CREATED FACE (exists only in B)
                    if a == -1 and b != -1:
                        is_null_pred = (b not in predicted_B)

                        TP += int(is_null_pred)
                        FN += int(not is_null_pred)

                        correct += int(is_null_pred)
                        total += 1
                        continue

                    # From here: A exists
                    if a == -1:
                        continue

                    is_null_pred = (preds[a].item() == -1)

                    # DELETED FACE
                    if b == -1:
                        TP += int(is_null_pred)
                        FN += int(not is_null_pred)

                        correct += int(is_null_pred)

                    # MODIFIED / UNMODIFIED FACE
                    else:
                        FP += int(is_null_pred)
                        correct += int((not is_null_pred) and (preds[a].item() == b))

                    total += 1

                # NULL F1 
                prec = TP / (TP + FP + 1e-12)
                rec  = TP / (TP + FN + 1e-12)
                EN.append(100 * (2 * prec * rec / (prec + rec + 1e-12)))

                # Unified Accuracy 
                EU.append(100 * correct / (total + 1e-12))

        avg1 = np.mean(E1)
        avg5 = np.mean(E5)
        avgN = np.mean(EN)
        avgU = np.mean(EU)   

        ep_loss = total_loss/len(ds)

        loss_hist.append(ep_loss)
        top1_hist.append(avg1)
        top5_hist.append(avg5)
        null_hist.append(avgN)
        unified_hist.append(np.mean(EU))

        print(
            f"Epoch {ep+1:03d} "
            f"Loss={ep_loss:.4f} "
            f"Top1={avg1:.2f}% "
            f"Top5={avg5:.2f}% "
            f"NullF1={avgN:.2f}% "
            f"Unified={avgU:.2f}% "
        )



        if avgU > best:
            best = avgU
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": CONFIG,
                "feat_mean": ds.feat_mean.cpu().tolist(),
                "feat_std": ds.feat_std.cpu().tolist(),
                "null_prob_threshold": float(NULL_PROB_TRAIN),
                "best_metric": "unified_accuracy",
                "best_value": float(avgU)
            }, CONFIG["save_path"])

    print("\nTraining Completed — Best Unified Accuracy:", best)




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
