import os

embedding_folder = r"C:\\Users\\Z0054udc\\Downloads\\jsons"
emb_files = [os.path.join(embedding_folder, f) for f in os.listdir(embedding_folder) if f.endswith('.json')]

for fname in emb_files:
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Only fix if enough lines
    if len(lines) > 5:
        # Remove previous accidental double-commas!
        if lines[4].strip().endswith(','):
            # Already has comma, nothing to do
            continue
        # Insert comma at the end of line 5 (index 4)
        lines[4] = lines[4].rstrip('\n').rstrip('\r').rstrip() + ',\n'
        # Write the fixed lines back
        with open(fname, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Fixed (comma added): {fname}")
    else:
        print(f"⛔ Too short (skipped): {fname}")

print("All done!")