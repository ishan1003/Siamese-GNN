import os
import json

def extract_model_num(emb_json):
    """
    Extracts the numeric model number prefix from the filenames in the 'files' dict.
    Example: '106_block_A.x_b' → returns '106'
    """
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
    """
    Merges ground-truth mappings (from XT JSON) with embeddings + edge lists
    (A_embeddings, B_embeddings, A_edges, B_edges) from individual model JSONs.
    """
    # 1️⃣ Load the main XT.json (contains mappings)
    with open(xt_path, 'r') as f_xt:
        xt_data = json.load(f_xt)

    # 2️⃣ List all embedding JSONs
    emb_files = [
        os.path.join(embedding_folder, fname)
        for fname in os.listdir(embedding_folder)
        if fname.endswith('.json')
    ]
    print(f"Found {len(emb_files)} embedding files in '{embedding_folder}'.")

    # 3️⃣ Load all embedding files into a dictionary
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

        # Extract embeddings and edges if present
        A_emb = emb_json.get("A_embeddings", {})
        B_emb = emb_json.get("B_embeddings", {})
        A_edges = emb_json.get("A_edges", [])
        B_edges = emb_json.get("B_edges", [])

        # Store everything together
        model_to_embeddings[model_num] = {
            "A_embeddings": A_emb,
            "B_embeddings": B_emb,
            "A_edges": A_edges,
            "B_edges": B_edges
        }

    # 4️⃣ Merge embeddings + edges with mappings
    updated_xt = {}
    for model_num, mapping in xt_data.items():
        embeds = model_to_embeddings.get(model_num)
        if embeds:
            # Merge structure: embeddings + edges + mappings
            merged = {
                "A_embeddings": embeds["A_embeddings"],
                "B_embeddings": embeds["B_embeddings"],
                "A_edges": embeds.get("A_edges", []),
                "B_edges": embeds.get("B_edges", []),
                "mappings": mapping
            }
        else:
            print(f"⚠️ No embeddings found for model {model_num}, keeping mappings only.")
            merged = {"mappings": mapping}
        updated_xt[model_num] = merged

    # 5️⃣ Save final merged JSON
    with open(out_path, 'w') as f_out:
        json.dump(updated_xt, f_out, indent=2)

    print(f"\n✅ Done! Merged XT file written to '{out_path}' with embeddings and edges included.")


if __name__ == "__main__":
    xt_path = "XT_complete_face_mappings.json"  # Ground truth mappings file
    embedding_folder = "C:\\Users\\Z0054udc\\Downloads\\jsons_complete"  # Folder with embedding JSONs
    out_path = "XT_merged_complete.json"  # Output file

    merge_embeddings_to_xt(xt_path, embedding_folder, out_path)
