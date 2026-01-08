import os

folder = "C:\\Users\\Z0054udc\\Downloads\\temp"  # <-- change this

for filename in os.listdir(folder):
    if filename.endswith("_pair.json"):
        number = int(filename.split("_")[0])
        new_number = number - 3000
        new_name = f"{new_number}_pair.json"

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        print(f"Renaming {filename} → {new_name}")
        os.rename(old_path, new_path)
