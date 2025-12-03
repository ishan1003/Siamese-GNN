import json

# 🔥 CHANGE THIS to your dataset file
json_path = r"C:\Users\Z0054udc\Downloads\Siamese GNN\XT_merged_new1_doubled_fixed.json"

with open(json_path, "r") as f:
    data = json.load(f)

zero_match_models = []

for model_num, pair in data.items():
    mappings = pair.get("mappings", [])

    # Check if there exists ANY valid mapping (a,b both not NULL)
    has_valid = False
    for a, b in mappings:
        if a != "NULL" and b != "NULL":
            has_valid = True
            break

    if not has_valid:
        zero_match_models.append(model_num)

print("\n=== MODELS WITH ZERO VALID MATCHES ===")
for m in zero_match_models:
    print(m)

print("\nTotal such models:", len(zero_match_models))
