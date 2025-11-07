import os
import json

def extract_model_num(emb_json):
    # Assumes the model number is the numeric prefix in file names, e.g. "123_..."
    for k in ['A', 'B']:
        filename = emb_json.get('files', {}).get(k)
        if filename:
            base = os.path.basename(filename)
            digits = ''
            for ch in base:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return digits
    return None

def merge_embeddings_to_xt(xt_path, embedding_folder, out_path):
    # Load main XT.json (with ground truth mappings)
    with open(xt_path, 'r') as f_xt:
        xt_data = json.load(f_xt)

    # List all embedding jsons
    emb_files = [os.path.join(embedding_folder, fname) for fname in os.listdir(embedding_folder) if fname.endswith('.json')]
    print(f"Found {len(emb_files)} embedding files.")

    model_to_embeddings = {}
    for emb_file in emb_files:
        print(f"Attempting to load: {emb_file}")
        try:
            with open(emb_file, 'r') as f_emb:
                emb_json = json.load(f_emb)
        except Exception as e:
            print(f"❌ Skipping {emb_file} due to JSON error:\n{e}\n")
            continue

        model_num = extract_model_num(emb_json)
        if not model_num:
            print(f"⚠️ Could not extract model number from '{emb_file}', skipping!\n")
            continue

        # Get only embeddings
        A_emb = emb_json.get("A_embeddings", {})
        B_emb = emb_json.get("B_embeddings", {})
        model_to_embeddings[model_num] = {"A_embeddings": A_emb, "B_embeddings": B_emb}

    # Merge everything into the new structure with embeddings first
    updated_xt = {}
    for model_num, mapping in xt_data.items():
        embeds = model_to_embeddings.get(model_num)
        if embeds:
            # Order: embeddings first, then mappings
            merged = {**embeds, "mappings": mapping}
        else:
            print(f"⚠️ No embeddings found for model {model_num}, keeping mappings only.")
            merged = {"mappings": mapping}
        updated_xt[model_num] = merged

    # Save to output
    with open(out_path, 'w') as f_out:
        json.dump(updated_xt, f_out, indent=2)

    print(f"\n✅ Done! XT.json written to '{out_path}' with embeddings before mappings.")

if __name__ == "__main__":
    xt_path = "matched_xt_ids.json"                           # path to XT.json (ground truth mappings)
    embedding_folder = "C:\\Users\\Z0054udc\\Downloads\\jsons" # path to embedding *.json files folder
    out_path = "XT_merged.json"                               # output file

    merge_embeddings_to_xt(xt_path, embedding_folder, out_path)