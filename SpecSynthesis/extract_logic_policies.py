import json
import re
from pathlib import Path

MODELS = ["gpt-4o", "gpt-4.1"]  # Add more models here if needed
MODELS = ["gpt-4o"]  # Currently only gpt-4o is processed

def extract_valid_logic_policies(raw):
    try:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        policies = data.get("logic_policies", [])
        if isinstance(policies, list) and policies:
            return policies
    except Exception:
        return None
    return None

def clean_policy_text(policy):
    """Remove leading numbering like '1. ', '2. ' etc."""
    return re.sub(r"^\s*\d+\.\s*", "", policy).strip()

def process_model(model_name):
    model_dir = Path(f"./logic_policies/{model_name}")
    input_path = model_dir / f"combined_{model_name}.jsonl"
    output_path = Path("./logic_policies") / f"extracted_logic_policies_{model_name}.jsonl"

    if not input_path.exists():
        print(f"[!] Skipping {model_name}: {input_path} does not exist.")
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        count_valid = 0
        count_total = 0

        for line in fin:
            count_total += 1
            try:
                entry = json.loads(line)
                raw = entry.get("raw_content")
                if raw:
                    policies = extract_valid_logic_policies(raw)
                    if policies:
                        for idx, p in enumerate(policies, start=1):
                            fout.write(json.dumps({
                                "path": entry["path"],
                                "section_ids": entry["section_ids"],
                                "policy_id": idx,
                                "policy": clean_policy_text(p)
                            }, ensure_ascii=False) + "\n")
                        count_valid += 1
            except json.JSONDecodeError:
                continue

    print(f" [{model_name}] Extracted policies from {count_valid}/{count_total} entries.")
    print(f" Output saved to: {output_path.resolve()}")

def main():
    for model in MODELS:
        process_model(model)

if __name__ == "__main__":
    main()
