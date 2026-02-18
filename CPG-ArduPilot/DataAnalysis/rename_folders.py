import os

TARGET_DIR = "./data"  # change this to your actual path
PREFIX = "2--"

for folder in os.listdir(TARGET_DIR):
    folder_path = os.path.join(TARGET_DIR, folder)
    if os.path.isdir(folder_path) and not folder.startswith(PREFIX) and not folder.startswith("3--"):
        new_name = PREFIX + folder
        new_path = os.path.join(TARGET_DIR, new_name)
        os.rename(folder_path, new_path)
        print(f"Renamed: {folder} -> {new_name}")
