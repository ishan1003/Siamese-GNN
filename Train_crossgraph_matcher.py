"""
train_full_pipeline.py

Cross-Graph Transformer + Sinkhorn with:
 - Positional encodings (SPD, degree, centrality)
 - Spectral features (top-k Laplacian eigenvectors)
 - Multi-head cross-attention
 - Structural-consistency loss (adjacency alignment)
 - Contrastive InfoNCE loss
 - NULL/no-match prediction (explicit NULL column in affinity)
 - Works with XT_merged.json-style dataset

Run: python train_full_pipeline.py
"""

import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv  # used as local encoder
from tqdm import tqdm

# -------------------------
# Hyperparameters / config
# -------------------------
DATA_PATH = "XT_merged_Synthetic.json"
DEVICE = torch.device("cpu")

BATCH_SIZE = 4             # small because matching is O(n^2)
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 300

IN_FEAT_DIM = None          # inferred automatically
HIDDEN_DIM = 128
NUM_GNN_LAYERS = 2

NUM_CROSS_LAYERS = 3
NUM_HEADS = 4               # multi-head cross-attention
SINKHORN_ITERS = 20
SINKHORN_EPS = 1e-3

SPEC_FEAT_K = 6             # top-k Laplacian eigenvectors
SPD_MAX = 8                 # cap for shortest-path distances (positional encoding)
TEMP_CONTRAST = 0.1         # InfoNCE temperature

LOSS_W_SUP = 1.0
LOSS_W_STRUCT = 1.0
LOSS_W_CONTRAST = 0.5

SAVE_PATH = "full_crossgraph_best.pt"

# -------------------------
# Utilities - graph ops
# -------------------------
def build_nx_graph(node_ids: List[int], edge_list: List[List[int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    # only add edges where both endpoints are present
    for u, v in edge_list:
        if u in node_ids and v in node_ids:
            G.add_edge(u, v)
    return G

def compute_shortest_path_matrix(G: nx.Graph, node_ids: List[int], max_hop=SPD_MAX) -> np.ndarray:
    n = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    sp_mat = np.ones((n, n), dtype=np.float32) * (max_hop + 1)
    for i, u in enumerate(node_ids):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_hop)
        for v, d in lengths.items():
            sp_mat[i, id2idx[v]] = float(min(d, max_hop))
    # optionally one-hot encode distances later
    return sp_mat  # integer distances in [0..max_hop] or max_hop+1

def compute_degree_centrality(G: nx.Graph, node_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    deg = np.array([G.degree(n) for n in node_ids], dtype=np.float32)
    cent = np.array([nx.degree_centrality(G)[n] for n in node_ids], dtype=np.float32)
    return deg.reshape(-1, 1), cent.reshape(-1, 1)

def compute_spectral_features(G: nx.Graph, node_ids: List[int], k=SPEC_FEAT_K) -> np.ndarray:
    # compute normalized Laplacian eigenvectors (small k). fallback to zeros if fail
    n = len(node_ids)
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)
    # adjacency
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    rows, cols = [], []
    for u, v in G.edges():
        if u in id2idx and v in id2idx:
            rows.append(id2idx[u]); cols.append(id2idx[v])
            rows.append(id2idx[v]); cols.append(id2idx[u])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    try:
        # normalized Laplacian
        degs = np.array(A.sum(axis=1)).flatten()
        with np.errstate(divide='ignore'):
            inv_sqrt = np.where(degs > 0, 1.0 / np.sqrt(degs), 0.0)
        D_sqrt_inv = sp.diags(inv_sqrt)
        L = sp.eye(n) - D_sqrt_inv @ A @ D_sqrt_inv
        k_eff = min(k, n - 1) if n > 1 else 1
        if k_eff <= 0:
            return np.zeros((n, k), dtype=np.float32)
        vals, vecs = eigsh(L, k=k_eff, which='SM')  # smallest eigenvalues
        # pad to k if needed
        vecs_full = np.zeros((n, k), dtype=np.float32)
        vecs_full[:, :k_eff] = vecs
        return vecs_full
    except Exception as e:
        # fallback: degree features
        return np.zeros((n, k), dtype=np.float32)

# -------------------------
# Dataset loader
# -------------------------
class JSONGraphMatchDataset(Dataset):
    def __init__(self, path: str):
        assert os.path.exists(path), f"{path} not found"
        with open(path, 'r') as f:
            data = json.load(f)
        # data may be dict-of-examples keyed by index strings
        if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
            self.examples = [data[k] for k in sorted(data.keys(), key=int)]
        elif isinstance(data, list):
            self.examples = data
        else:
            self.examples = list(data.values())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        A_map = ex['A_embeddings']
        B_map = ex['B_embeddings']
        A_nodes = [int(k) for k in A_map.keys()]
        B_nodes = [int(k) for k in B_map.keys()]

        # preserve consistent order
        A_nodes_sorted = A_nodes
        B_nodes_sorted = B_nodes

        A_feats = np.array([A_map[str(n)] if str(n) in A_map else A_map[n] for n in A_nodes_sorted], dtype=np.float32)
        B_feats = np.array([B_map[str(n)] if str(n) in B_map else B_map[n] for n in B_nodes_sorted], dtype=np.float32)

        # graphs via networkx for positional / spectral
        A_edges = ex.get('A_edges', [])
        B_edges = ex.get('B_edges', [])

        G_A = build_nx_graph(A_nodes_sorted, A_edges)
        G_B = build_nx_graph(B_nodes_sorted, B_edges)

        # compute pos enc / spectral / degree / centrality
        spd_A = compute_shortest_path_matrix(G_A, A_nodes_sorted)  # [n_a, n_a]
        spd_B = compute_shortest_path_matrix(G_B, B_nodes_sorted)  # [n_b, n_b]

        deg_A, cent_A = compute_degree_centrality(G_A, A_nodes_sorted)
        deg_B, cent_B = compute_degree_centrality(G_B, B_nodes_sorted)

        spec_A = compute_spectral_features(G_A, A_nodes_sorted, k=SPEC_FEAT_K)
        spec_B = compute_spectral_features(G_B, B_nodes_sorted, k=SPEC_FEAT_K)

        # normalize original features
        # if numeric dims differ, we'll project later in model
        sample = {
            'A_nodes': A_nodes_sorted,
            'B_nodes': B_nodes_sorted,
            'A_feats': A_feats,           # [n_a, f]
            'B_feats': B_feats,           # [n_b, f]
            'A_edges': A_edges,
            'B_edges': B_edges,
            'A_spd': spd_A,               # [n_a, n_a]
            'B_spd': spd_B,
            'A_deg': deg_A,               # [n_a,1]
            'B_deg': deg_B,
            'A_cent': cent_A,             # [n_a,1]
            'B_cent': cent_B,
            'A_spec': spec_A,             # [n_a, SPEC_FEAT_K]
            'B_spec': spec_B,
            'mappings': ex.get('mappings', [])
        }
        return sample

def collate_fn(batch):
    # small batch list -> return as-is
    return batch

# -------------------------
# Model components
# -------------------------
class LocalGNNEncoder(nn.Module):
    """Simple GCN-based local encoder to mix neighborhood info - projects to hidden_dim"""
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            if edge_index is None or edge_index.numel() == 0:
                # no edges: linear mapping
                h = conv.lin(h)
            else:
                h = conv(h, edge_index)
            h = ln(h)
            h = self.act(h)
        return h

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: A attends to B and B attends to A (symmetric)"""
    def __init__(self, emb_dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_lin = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_lin = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_lin = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, A: torch.Tensor, B: torch.Tensor, maskA=None, maskB=None):
        # A: [n_a, d], B: [n_b, d]
        Bsz = 1
        Qa = self.q_lin(A)  # [n_a,d]
        Kb = self.k_lin(B)  # [n_b,d]
        Vb = self.v_lin(B)  # [n_b,d]

        # reshape for multi-head
        def split_heads(x):
            # x: [n, d] -> [n, h, head_dim] -> [h, n, head_dim]
            n = x.size(0)
            x = x.view(n, self.num_heads, self.head_dim).permute(1, 0, 2)
            return x

        Qh = split_heads(Qa)
        Kh = split_heads(Kb)
        Vh = split_heads(Vb)
        # attention scores per head: [h, n_a, n_b]
        scores = torch.einsum('hnd,hmd->hnm', Qh, Kh) / self.scale
        attn = F.softmax(scores, dim=-1)  # softmax over B nodes
        attn = self.dropout(attn)
        # A update: [h, n_a, head_dim]
        Ah = torch.einsum('hnm,hmd->hnd', attn, Vh)
        # combine heads
        Ah = Ah.permute(1, 0, 2).contiguous().view(A.size(0), self.emb_dim)
        A_out = self.out(Ah) + A

        # symmetric B <- A
        Qb = self.q_lin(B)
        Ka = self.k_lin(A)
        Va = self.v_lin(A)
        Qh_b = split_heads(Qb)
        Kh_a = split_heads(Ka)
        Vh_a = split_heads(Va)
        scores_b = torch.einsum('hnd,hmd->hnm', Qh_b, Kh_a) / self.scale
        attn_b = F.softmax(scores_b, dim=-1)
        attn_b = self.dropout(attn_b)
        Bh = torch.einsum('hnm,hmd->hnd', attn_b, Vh_a)
        Bh = Bh.permute(1, 0, 2).contiguous().view(B.size(0), self.emb_dim)
        B_out = self.out(Bh) + B

        return A_out, B_out

def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    # log_alpha: [n_a, n_b] (per-sample logits) OR batched [B, n_a, n_b]
    # We'll operate in log-space
    if log_alpha.dim() == 2:
        log_p = log_alpha.unsqueeze(0)
    else:
        log_p = log_alpha
    for _ in range(n_iters):
        # row norm
        log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
        # col norm
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    P = torch.exp(log_p)
    if log_alpha.dim() == 2:
        return P[0]
    return P

# -------------------------
# Full model
# -------------------------
class FullCrossGraphMatcher(nn.Module):
    def __init__(self, raw_feat_dim, aux_feat_dim, hidden_dim=HIDDEN_DIM,
                 num_gnn_layers=NUM_GNN_LAYERS, num_cross_layers=NUM_CROSS_LAYERS,
                 num_heads=NUM_HEADS, sinkhorn_iters=SINKHORN_ITERS):
        """
        raw_feat_dim: dimension of original node features
        aux_feat_dim: dimension of appended aux features (spectral + deg + cent + spd-onehot or pooled)
        We'll project (raw + aux) to hidden_dim and run GNN + cross-attention
        """
        super().__init__()
        self.in_proj = nn.Linear(raw_feat_dim + aux_feat_dim, hidden_dim)
        self.local = LocalGNNEncoder(hidden_dim, hidden_dim, num_layers=num_gnn_layers)
        self.cross_layers = nn.ModuleList([MultiHeadCrossAttention(hidden_dim, num_heads=num_heads)
                                           for _ in range(num_cross_layers)])
        self.post_ln = nn.LayerNorm(hidden_dim)
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.sinkhorn_iters = sinkhorn_iters

        # add NULL column bias: we will append one learnable NULL vector to B projections
        self.null_vec = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # projection for contrastive embeddings
        self.contrast_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, A_x, A_ei, B_x, B_ei):
        # A_x: [n_a, feat], B_x: [n_b, feat]
        A_in = self.in_proj(A_x)
        B_in = self.in_proj(B_x)
        # local GNN (edge_index expects torch.LongTensor 2xE in 0..n-1)
        A_h = self.local(A_in, A_ei)
        B_h = self.local(B_in, B_ei)
        for layer in self.cross_layers:
            A_h, B_h = layer(A_h, B_h)
        A_h = self.post_ln(A_h)
        B_h = self.post_ln(B_h)

        A_p = self.project(A_h)   # [n_a, H]
        B_p = self.project(B_h)   # [n_b, H]

        # affinity logits: dot product scaled
        logits = (A_p @ B_p.t()) / math.sqrt(A_p.size(-1))  # [n_a, n_b]
        # add NULL column: logits_null = A_p @ null_vec
        null_col = (A_p @ self.null_vec) / math.sqrt(A_p.size(-1))  # [n_a]
        logits_with_null = torch.cat([logits, null_col.unsqueeze(1)], dim=1)  # [n_a, n_b+1]

        # Sinkhorn expects log probs; we pass logits_with_null
        P = sinkhorn(logits_with_null, n_iters=self.sinkhorn_iters)  # [n_a, n_b+1], rows sum to 1

        # contrastive vectors for InfoNCE
        zA = self.contrast_proj(A_p)  # [n_a, H]
        zB = self.contrast_proj(B_p)  # [n_b, H]
        return logits_with_null, P, zA, zB

# -------------------------
# Losses
# -------------------------
def supervised_row_crossentropy(logits_with_null: torch.Tensor, target_indices: torch.Tensor):
    """
    logits_with_null: [n_a, n_b+1] (raw logits)
    target_indices: [n_a] in 0..n_b (index of matched B) OR n_b (index for NULL)
    Returns mean cross-entropy over all rows (including NULL rows)
    """
    if logits_with_null.numel() == 0:
        return torch.tensor(0., device=logits_with_null.device, requires_grad=True)
    # use standard cross-entropy per-row
    logp = F.log_softmax(logits_with_null, dim=1)
    # gather
    nll = -logp[torch.arange(logits_with_null.size(0), device=logits_with_null.device), target_indices.to(logits_with_null.device)]
    return nll.mean()

def structural_consistency_loss(P_soft: torch.Tensor, A_adj: torch.Tensor, B_adj: torch.Tensor):
    """
    P_soft: [n_a, n_b+1] - last column is NULL and should not be used for structural consistency.
    A_adj: [n_a, n_a] adjacency (0/1)
    B_adj: [n_b, n_b] adjacency
    Compute soft adjacency alignment: A_adj ~ P[:, :n_b] @ B_adj @ P[:, :n_b]^T
    Use Frobenius norm.
    """
    if A_adj.numel() == 0 or B_adj.numel() == 0:
        return torch.tensor(0., device=P_soft.device)
    P = P_soft[:, :-1]  # drop NULL column -> [n_a, n_b]
    # compute soft transported adjacency
    # S = P * B_adj * P^T  -> [n_a, n_a]
    S = P @ B_adj @ P.t()
    loss = F.mse_loss(S, A_adj)
    return loss

def contrastive_infonce(zA: torch.Tensor, zB: torch.Tensor, pos_pairs: List[Tuple[int,int]], temperature=TEMP_CONTRAST):
    """
    zA: [n_a, d], zB: [n_b, d]
    pos_pairs: list of (i_a, j_b) indices that are positive matches
    We'll compute InfoNCE where for each positive pair, negatives are other zB in same sample.
    Return averaged contrastive loss across pos_pairs.
    """
    if len(pos_pairs) == 0:
        return torch.tensor(0., device=zA.device)
    zA_norm = F.normalize(zA, dim=1)
    zB_norm = F.normalize(zB, dim=1)
    losses = []
    for (ia, jb) in pos_pairs:
        if ia < 0 or jb < 0 or ia >= zA_norm.size(0) or jb >= zB_norm.size(0):
            continue
        anchor = zA_norm[ia:ia+1]  # [1,d]
        pos = zB_norm[jb:jb+1]     # [1,d]
        # logits: anchor dot all positives
        logits = (anchor @ zB_norm.t()).squeeze(0) / temperature  # [n_b]
        # target index = jb
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([jb], device=logits.device))
        losses.append(loss)
    if len(losses) == 0:
        return torch.tensor(0., device=zA.device)
    return torch.stack(losses).mean()

# -------------------------
# Helpers to build adjacency / edge_index tensor
# -------------------------
def build_edge_index_and_adj(node_ids: List[int], edge_list: List[List[int]]):
    # map ids to indices
    idx = {n: i for i, n in enumerate(node_ids)}
    edges = []
    for u, v in edge_list:
        if u in idx and v in idx:
            edges.append((idx[u], idx[v]))
            edges.append((idx[v], idx[u]))
    if len(edges) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        adj = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.float32)
        return edge_index, adj
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    n = len(node_ids)
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    data = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.zeros((n, n), dtype=torch.float32)
    adj[rows, cols] = 1.0
    return edge_index, adj

# -------------------------
# Training & eval loops
# -------------------------
def prepare_input_tensors(sample, device):
    # sample from dataset: produce torch tensors for raw features and appended aux features
    A_feats = torch.tensor(sample['A_feats'], dtype=torch.float32, device=device)  # [n_a, f]
    B_feats = torch.tensor(sample['B_feats'], dtype=torch.float32, device=device)

    # build adjacency & edge_index for GCN
    A_ei, A_adj = build_edge_index_and_adj(sample['A_nodes'], sample['A_edges'])
    B_ei, B_adj = build_edge_index_and_adj(sample['B_nodes'], sample['B_edges'])
    A_ei = A_ei.to(device)
    B_ei = B_ei.to(device)
    A_adj = A_adj.to(device)
    B_adj = B_adj.to(device)

    # Positional encodings: we'll convert SPD to per-node one-hot histogram features:
    # For each node i, compute histogram of shortest path distances to all nodes (caps at SPD_MAX)
    def spd_to_hist(spd_mat):
        if spd_mat.size == 0:
            return np.zeros((0, SPD_MAX+2), dtype=np.float32)
        # distances are integers up to SPD_MAX+1
        n = spd_mat.shape[0]
        hist = np.zeros((n, SPD_MAX+2), dtype=np.float32)
        for i in range(n):
            row = spd_mat[i]
            # clip
            row_clipped = np.minimum(row, SPD_MAX+1).astype(np.int32)
            for d in row_clipped:
                hist[i, d] += 1.0
            # normalize histogram
            if hist[i].sum() > 0:
                hist[i] = hist[i] / hist[i].sum()
        return hist

    A_spd_hist = spd_to_hist(sample['A_spd'])
    B_spd_hist = spd_to_hist(sample['B_spd'])

    A_deg = sample['A_deg']
    B_deg = sample['B_deg']
    A_cent = sample['A_cent']
    B_cent = sample['B_cent']
    A_spec = sample['A_spec']
    B_spec = sample['B_spec']

    # concatenate aux features: [spd_hist | deg | cent | spec]
    A_aux = np.concatenate([A_spd_hist, A_deg, A_cent, A_spec], axis=1).astype(np.float32)
    B_aux = np.concatenate([B_spd_hist, B_deg, B_cent, B_spec], axis=1).astype(np.float32)

    A_aux_t = torch.tensor(A_aux, dtype=torch.float32, device=device)
    B_aux_t = torch.tensor(B_aux, dtype=torch.float32, device=device)

    # If raw features dims differ, pad / truncate to align (model will accept raw_dim same across samples)
    return A_feats, A_aux_t, A_ei, A_adj, B_feats, B_aux_t, B_ei, B_adj

def build_targets(sample):
    # target indices per A node: 0..n_b-1, or n_b for NULL
    n_b = len(sample['B_nodes'])
    default_null = n_b
    target = [default_null] * len(sample['A_nodes'])
    for pair in sample['mappings']:
        a_id, b_id = pair
        if a_id == "NULL" or b_id == "NULL":
            continue
        try:
            ia = sample['A_nodes'].index(int(a_id))
            jb = sample['B_nodes'].index(int(b_id))
            target[ia] = jb
        except ValueError:
            continue
    return torch.tensor(target, dtype=torch.long)

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_sup = 0.0
    total_struct = 0.0
    total_contrast = 0.0
    total_examples = 0
    total_correct = 0
    pbar = tqdm(dataloader, desc=f"Train E{epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        batch_loss = 0.0
        # process each sample (batch size small)
        for sample in batch:
            A_feats, A_aux, A_ei, A_adj, B_feats, B_aux, B_ei, B_adj = prepare_input_tensors(sample, device)
            # build unified raw+aux features
            A_in = torch.cat([A_feats, A_aux], dim=1)
            B_in = torch.cat([B_feats, B_aux], dim=1)
            # ensure dims consistent
            logits_with_null, P_soft, zA, zB = model(A_in, A_ei, B_in, B_ei)
            target = build_targets(sample).to(device)

            # supervised cross-entropy (including NULL column)
            sup_loss = supervised_row_crossentropy(logits_with_null, target)
            struct_loss = structural_consistency_loss(P_soft, A_adj, B_adj)
            # gather positive pairs for InfoNCE
            pos_pairs = []
            for pair in sample['mappings']:
                a_id, b_id = pair
                if a_id == "NULL" or b_id == "NULL":
                    continue
                try:
                    ia = sample['A_nodes'].index(int(a_id))
                    jb = sample['B_nodes'].index(int(b_id))
                    pos_pairs.append((ia, jb))
                except ValueError:
                    pass
            contrast_loss = contrastive_infonce(zA, zB, pos_pairs)

            loss = LOSS_W_SUP * sup_loss + LOSS_W_STRUCT * struct_loss + LOSS_W_CONTRAST * contrast_loss
            batch_loss = batch_loss + loss

            # stats
            # predicted index per A row (including NULL index)
            preds = logits_with_null.argmax(dim=1).detach().cpu().numpy()
            tgt = target.detach().cpu().numpy()
            total_correct += (preds == tgt).sum()
            total_examples += tgt.size

            total_sup += float(sup_loss.detach().cpu().item())
            total_struct += float(struct_loss.detach().cpu().item())
            total_contrast += float(contrast_loss.detach().cpu().item())

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()
        total_loss += float(batch_loss.detach().cpu().item())
        pbar.set_postfix(train_loss=f"{total_loss/(total_examples+1e-12):.5f}", acc=f"{total_correct/total_examples:.3f}")

    avg_loss = total_loss / max(1, len(dataloader))
    avg_acc = total_correct / (total_examples + 1e-12)
    return avg_loss, avg_acc

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            for sample in batch:
                A_feats, A_aux, A_ei, A_adj, B_feats, B_aux, B_ei, B_adj = prepare_input_tensors(sample, device)
                A_in = torch.cat([A_feats, A_aux], dim=1)
                B_in = torch.cat([B_feats, B_aux], dim=1)
                logits_with_null, P_soft, zA, zB = model(A_in, A_ei, B_in, B_ei)
                target = build_targets(sample).to(device)
                sup_loss = supervised_row_crossentropy(logits_with_null, target)
                total_loss += float(sup_loss.cpu().item())
                preds = logits_with_null.argmax(dim=1).cpu().numpy()
                tgt = target.cpu().numpy()
                total_correct += (preds == tgt).sum()
                total_examples += tgt.size
    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)
    return avg_loss, avg_acc

# -------------------------
# Main runner
# -------------------------
def main():
    ds = JSONGraphMatchDataset(DATA_PATH)
    n = len(ds)
    print(f"Found {n} examples.")
    n_train = int(0.8 * n)
    train_ds = JSONGraphMatchDataset(DATA_PATH)
    train_ds.examples = train_ds.examples[:n_train]
    val_ds = JSONGraphMatchDataset(DATA_PATH)
    val_ds.examples = val_ds.examples[n_train:]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # infer raw_feat and aux dims from first item
    sample0 = train_ds[0]
    raw_dim = sample0['A_feats'].shape[1]
    # aux dim = SPD histogram width + deg + cent + spec
    spd_hist_dim = SPD_MAX + 2
    aux_dim = spd_hist_dim + 1 + 1 + SPEC_FEAT_K

    print("raw_dim:", raw_dim, "aux_dim:", aux_dim)
    model = FullCrossGraphMatcher(raw_dim, aux_dim, hidden_dim=HIDDEN_DIM,
                                  num_gnn_layers=NUM_GNN_LAYERS, num_cross_layers=NUM_CROSS_LAYERS,
                                  num_heads=NUM_HEADS, sinkhorn_iters=SINKHORN_ITERS).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, DEVICE, epoch)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, SAVE_PATH)
            print(f"Saved best model at epoch {epoch} val_acc {val_acc:.4f}")

if __name__ == "__main__":
    main()
