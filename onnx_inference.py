# ==============================
# onnx_inference.py (FINAL)
# ==============================

import numpy as np
import onnxruntime as ort
import torch

# ------------------------------------------------------
# Helper: edge_index -> adjacency matrix
# ------------------------------------------------------
def edge_index_to_adj(edge_index, num_nodes):
    """
    edge_index: LongTensor shape [2, E]
    returns (num_nodes, num_nodes) adjacency normalized by out-degree.
    """
    edge_index = edge_index.cpu().numpy()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edge_index.T:
        adj[i, j] = 1.0

    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    adj = adj / deg
    return adj.astype(np.float32)


# ------------------------------------------------------
# Load ONNX runtime session
# ------------------------------------------------------
onnx_path = "siamese_gnn.onnx"
print(f"Loading ONNX model: {onnx_path}")

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Get input/output names (for safety)
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]
print("Inputs:", input_names)
print("Outputs:", output_names)


# ------------------------------------------------------
# RUN INFERENCE ON SAMPLE GRAPH
# Replace xA, edgeA, xB, edgeB with your real data.
# ------------------------------------------------------

# Example dummy graph A
numA = 5
xA = torch.randn(numA, 8)           # 8 = feature_size example
edgeA = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
adjA = edge_index_to_adj(edgeA, numA)

# Example dummy graph B
numB = 6
xB = torch.randn(numB, 8)
edgeB = torch.tensor([[0,1,3],[1,3,4]], dtype=torch.long)
adjB = edge_index_to_adj(edgeB, numB)

# Convert to numpy
inputs = {
    "xA": xA.numpy().astype(np.float32),
    "adjA": adjA,
    "xB": xB.numpy().astype(np.float32),
    "adjB": adjB,
}

# ------------------------------------------------------
# Execute ONNX forward pass
# ------------------------------------------------------
print("\nRunning ONNX inference...")
outputs = session.run(output_names, inputs)

hA, hB, zA, zB, nullA, nullB = outputs

print("\n--- ONNX Outputs ---")
print("hA:", hA.shape)
print("hB:", hB.shape)
print("zA:", zA.shape)
print("zB:", zB.shape)
print("nullA:", nullA.shape)
print("nullB:", nullB.shape)

# ------------------------------------------------------
# Example: Compute similarity & best match for each A-node
# ------------------------------------------------------
print("\n--- Matching Example (Top-1) ---")
# zA: [numA, proj_dim],  zB: [numB, proj_dim]

sims = zA @ zB.T  # cosine similarity matrix
best_idx = sims.argmax(axis=1)
best_sim = sims.max(axis=1)

for i in range(numA):
    print(f"A[{i}] → B[{best_idx[i]}]  (sim={best_sim[i]:.4f})")
