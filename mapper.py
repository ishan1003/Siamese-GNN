import json
import os

# Input file (change if needed)
INPUT_JSON = "XT_complete.json"

# Output file
OUTPUT_JSON = "XT_complete_face_mappings.json"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} xt-entries.")

# ---------------------------------------
# Build A/B pairs per model id
# ---------------------------------------
pairs = {}

for entry in data:
    path = entry["xt_file"]  # example: "151_block_A.x_b"
    fname = os.path.basename(path)

    base = fname.replace(".x_b", "")       # 151_block_A
    parts = base.split("_")                # ["151","block","A"]

    model_id = parts[0]                    # "151"
    side = parts[-1]                       # "A" or "B"

    if model_id not in pairs:
        pairs[model_id] = {}

    pairs[model_id][side] = entry["xt_entity_ids"]

print(f"Found {len(pairs)} model pairs.")

# ---------------------------------------
# Build combined mapping structure
# ---------------------------------------
final_output = {}

for model_id, ab in pairs.items():
    A_ids = sorted(ab.get("A", []))
    B_ids = sorted(ab.get("B", []))

    A_set = set(A_ids)
    B_set = set(B_ids)

    mappings = []

    # First handle A → B / NULL
    for a in A_ids:
        if a in B_set:
            mappings.append([a, a])
        else:
            mappings.append([a, "NULL"])

    # Then handle B-only faces
    for b in B_ids:
        if b not in A_set:
            mappings.append(["NULL", b])

    final_output[model_id] = mappings

# ---------------------------------------
# Save final combined mappings file
# ---------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\nSaved mappings to {OUTPUT_JSON}")
