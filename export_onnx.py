# ===============================
# export_onnx.py (FINAL PATCHED)
# ===============================

import torch
import torch.nn as nn
import torch.nn.functional as F

# We only need torch_geometric to REDEFINE the original model so we can load the checkpoint.
# The ONNX model does NOT depend on torch_geometric.
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
except ImportError:
    SAGEConv = None


# -------------------------------------------------------------------
# 1. ORIGINAL TRAINING MODEL (must match your training architecture)
# -------------------------------------------------------------------
if SAGEConv is not None:
    class GraphEncoder(nn.Module):
        def __init__(self, in_dim, hid, out_dim):
            super().__init__()
            self.g1 = SAGEConv(in_dim, hid)
            self.g2 = SAGEConv(hid, hid)
            self.g3 = SAGEConv(hid, out_dim)

        def forward(self, x, edge_index):
            x = F.relu(self.g1(x, edge_index))
            x = F.relu(self.g2(x, edge_index))
            x = self.g3(x, edge_index)
            return x
else:
    # Fallback for environments without torch_geometric
    class DummyConv(nn.Module):
        def __init__(self):
            super().__init__()
    class GraphEncoder(nn.Module):
        def __init__(self, in_dim, hid, out_dim):
            super().__init__()
            self.g1 = DummyConv()
            self.g2 = DummyConv()
            self.g3 = DummyConv()
        def forward(self, x, edge_index):
            raise RuntimeError("torch_geometric not installed.")


class SiameseGNN(nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hid, out_dim)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        self.null_head = nn.Sequential(
            nn.Linear(out_dim + proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, A, B):
        hA = self.encoder(A.x, A.edge_index)
        hB = self.encoder(B.x, B.edge_index)
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)
        nullA = self.null_head(torch.cat([hA, zA], dim=1))
        nullB = self.null_head(torch.cat([hB, zB], dim=1))
        return hA, hB, zA, zB, nullA, nullB


# -------------------------------------------------------------------
# 2. SAFE LOAD WRAPPER (avoids torch.load warnings)
# -------------------------------------------------------------------
def safe_torch_load(path, map_location="cpu"):
    """
    Prefer weights_only=True (safer). Fallback to default if unsupported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except RuntimeError:
        print("Warning: weights_only=True failed; falling back to standard torch.load().")
        return torch.load(path, map_location=map_location)


# -------------------------------------------------------------------
# 3. ONNX-FRIENDLY GRAPH SAGE IMPLEMENTATION
# -------------------------------------------------------------------
class SAGEConvONNX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        neigh = adj @ x
        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphEncoderONNX(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.g1 = SAGEConvONNX(in_dim, hid)
        self.g2 = SAGEConvONNX(hid, hid)
        self.g3 = SAGEConvONNX(hid, out_dim)

    def forward(self, x, adj):
        x = F.relu(self.g1(x, adj))
        x = F.relu(self.g2(x, adj))
        x = self.g3(x, adj)
        return x


class SiameseGNN_ONNX(nn.Module):
    def __init__(self, in_dim, hid, out_dim, proj_dim):
        super().__init__()
        self.encoder = GraphEncoderONNX(in_dim, hid, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.null_head = nn.Sequential(
            nn.Linear(out_dim + proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, xA, adjA, xB, adjB):
        hA = self.encoder(xA, adjA)
        hB = self.encoder(xB, adjB)
        zA = F.normalize(self.proj(hA), dim=1)
        zB = F.normalize(self.proj(hB), dim=1)
        nullA = self.null_head(torch.cat([hA, zA], dim=1))
        nullB = self.null_head(torch.cat([hB, zB], dim=1))
        return hA, hB, zA, zB, nullA, nullB


# -------------------------------------------------------------------
# 4. ADJ BUILDER
# -------------------------------------------------------------------
def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes))
    for i, j in edge_index.t().tolist():
        adj[i, j] = 1.0
    deg = adj.sum(1, keepdim=True).clamp(min=1.0)
    return adj / deg


# -------------------------------------------------------------------
# 5. SAFE WEIGHT COPY FOR ALL SAGEConv VARIANTS
# -------------------------------------------------------------------
def copy_sage_weights(src_conv, dst_conv):
    """
    Handles:
    - src.lin_l + optional src.lin_r
    - src.lin_l.bias = None
    - root_weight=False -> no src.lin_r
    - fallback: src.lin_l copied to dst.lin_neigh
    """

    def safe_copy(src_lin, dst_lin):
        dst_lin.weight.data.copy_(src_lin.weight.data)
        if hasattr(src_lin, "bias") and src_lin.bias is not None:
            dst_lin.bias.data.copy_(src_lin.bias.data)
        else:
            dst_lin.bias.data.zero_()

    # Standard SAGEConv: lin_l always exists
    if hasattr(src_conv, "lin_l") and src_conv.lin_l is not None:
        safe_copy(src_conv.lin_l, dst_conv.lin_self)

        # root_weight=True -> lin_r exists
        if hasattr(src_conv, "lin_r") and src_conv.lin_r is not None:
            safe_copy(src_conv.lin_r, dst_conv.lin_neigh)
        else:
            # fallback if lin_r doesn’t exist or is None
            dst_conv.lin_neigh.weight.data.copy_(src_conv.lin_l.weight.data)
            if src_conv.lin_l.bias is not None:
                dst_conv.lin_neigh.bias.data.copy_(src_conv.lin_l.bias.data)
            else:
                dst_conv.lin_neigh.bias.data.zero_()
        return

    # Older variant with single .lin
    if hasattr(src_conv, "lin") and src_conv.lin is not None:
        safe_copy(src_conv.lin, dst_conv.lin_self)
        safe_copy(src_conv.lin, dst_conv.lin_neigh)
        return

    raise RuntimeError("Unsupported SAGEConv format; cannot copy weights.")


# -------------------------------------------------------------------
# 6. MAIN EXPORT PIPELINE
# -------------------------------------------------------------------
if __name__ == "__main__":
    ckpt = safe_torch_load("siamese_infonce_null_classifier.pt")

    CONFIG = ckpt["config"]

    # Build & load original model (with PyG SAGEConv)
    orig_model = SiameseGNN(
        in_dim=len(CONFIG["use_features"]),
        hid=CONFIG["encoder_hidden"],
        out_dim=CONFIG["encoder_out"],
        proj_dim=CONFIG["proj_dim"]
    )
    orig_model.load_state_dict(ckpt["model_state"])
    orig_model.eval()

    # Build ONNX model
    onnx_model = SiameseGNN_ONNX(
        in_dim=len(CONFIG["use_features"]),
        hid=CONFIG["encoder_hidden"],
        out_dim=CONFIG["encoder_out"],
        proj_dim=CONFIG["proj_dim"]
    )

    # Copy GraphEncoder (SAGEConv) weights
    copy_sage_weights(orig_model.encoder.g1, onnx_model.encoder.g1)
    copy_sage_weights(orig_model.encoder.g2, onnx_model.encoder.g2)
    copy_sage_weights(orig_model.encoder.g3, onnx_model.encoder.g3)

    # Copy projection & classifier weights
    onnx_model.proj.load_state_dict(orig_model.proj.state_dict())
    onnx_model.null_head.load_state_dict(orig_model.null_head.state_dict())

    onnx_model.eval()

    # Dummy inputs for export
    numA, numB = 8, 10
    xA = torch.randn(numA, len(CONFIG["use_features"]))
    xB = torch.randn(numB, len(CONFIG["use_features"]))
    edgeA = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    edgeB = torch.tensor([[0,1,3],[1,3,4]], dtype=torch.long)
    adjA = edge_index_to_adj(edgeA, numA)
    adjB = edge_index_to_adj(edgeB, numB)

    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        (xA, adjA, xB, adjB),
        "siamese_gnn.onnx",
        opset_version=17,
        input_names=["xA", "adjA", "xB", "adjB"],
        output_names=["hA", "hB", "zA", "zB", "nullA", "nullB"],
        dynamic_axes={
            "xA": {0: "nodesA"},
            "adjA": {0: "nodesA", 1: "nodesA"},
            "xB": {0: "nodesB"},
            "adjB": {0: "nodesB", 1: "nodesB"},
        }
    )

    print("\n✔ Export completed successfully → siamese_gnn.onnx")
