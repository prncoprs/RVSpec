import json
from pathlib import Path

# Root directory where model outputs are stored


# cz42: Adjust paths for different RVs
ROOT_DIR = Path("mtl/cpg_mtl")
ROOT_DIR = Path("mtl/cpg_mtl-openpilot")
ROOT_DIR = Path("mtl/cpg_mtl-cFS")
# MODELS = ["gpt-4o", "gpt-4.1"]
MODELS = ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]

def merge_json_files_to_jsonl(model_name):
    model_dir = ROOT_DIR / model_name
    output_file = ROOT_DIR / f"combined_{model_name}.jsonl"

    if not model_dir.exists():
        print(f"[!] Directory not found: {model_dir}")
        return

    json_files = list(model_dir.glob("*.json"))
    print(f" Merging {len(json_files)} files from {model_name}...")

    with open(output_file, "w", encoding="utf-8") as fout:
        for jf in sorted(json_files):
            try:
                with open(jf, "r", encoding="utf-8") as fin:
                    obj = json.load(fin)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[!] Failed to read {jf}: {e}")
    
    print(f" Combined JSONL saved to: {output_file.resolve()}")

def main():
    for model in MODELS:
        merge_json_files_to_jsonl(model)

if __name__ == "__main__":
    main()
