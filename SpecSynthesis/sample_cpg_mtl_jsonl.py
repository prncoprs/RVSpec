import json
import random
import re
from pathlib import Path
from collections import OrderedDict

# Configuration
ROOT_DIR = Path("mtl/cpg_mtl")
ROOT_DIR = Path("mtl/cpg_mtl-openpilot")
ROOT_DIR = Path("mtl/cpg_mtl-cFS")

MODELS = ["gpt-4o", "gpt-4.1"]
MODELS = ["claude-sonnet-4-20250514"]  # Currently only claude-sonnet-4 is processed
SAMPLE_SIZE = 44
RANDOM_SEED = 42
PATH_PREFIX = "<RVSPEC_ROOT>/SpecSynthesis/"

def merge_section_ids(section_ids):
    if not section_ids:
        return section_ids
    pattern = r"(.+)__(\d+)"
    base = None
    nums = []
    for sid in section_ids:
        match = re.match(pattern, sid)
        if match:
            b, n = match.groups()
            if base is None:
                base = b
            nums.append(int(n))
        else:
            return section_ids  # Leave unmerged if format is unexpected
    nums = sorted(nums)
    return f"{base}__{nums[0]}-{nums[-1]}" if base else section_ids

def sample_jsonl(model_name):
    input_file = ROOT_DIR / f"combined_{model_name}.jsonl"
    output_file = ROOT_DIR / f"sampled_{model_name}.jsonl"

    if not input_file.exists():
        print(f"[!] Input file does not exist: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    if len(entries) <= SAMPLE_SIZE:
        print(f"[!] Only {len(entries)} records found. Copying all.")
        sampled = entries
    else:
        random.seed(RANDOM_SEED)
        sampled = random.sample(entries, SAMPLE_SIZE)

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(sampled, start=1):
            if entry.get("path", "").startswith(PATH_PREFIX):
                entry["path"] = entry["path"].replace(PATH_PREFIX, "", 1)
            if isinstance(entry.get("section_ids"), list):
                entry["section_ids"] = merge_section_ids(entry["section_ids"])

            # Create OrderedDict to enforce key order
            ordered = OrderedDict()
            ordered["id"] = idx
            for key in entry:
                if key != "id":
                    ordered[key] = entry[key]

            fout.write(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n\n")

    print(f" Sampled {len(sampled)} entries to {output_file.resolve()}")

def main():
    for model in MODELS:
        sample_jsonl(model)

if __name__ == "__main__":
    main()
