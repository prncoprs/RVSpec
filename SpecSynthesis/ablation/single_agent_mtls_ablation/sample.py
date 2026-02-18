import json

# Input and output paths
input_path = "extracted_single_agent_mtls.jsonl"
output_path = "split_mtls.jsonl"


next_id = 1

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        base_fields = {
            "path": data["path"],
            "section_ids": data["section_ids"],
            "documentation": data.get("documentation", None)
        }
        mtls = data.get("mtl", [])
        for mtl_entry in mtls:
            entry = {
                "id": next_id,
                **base_fields,
                "policy": mtl_entry["policy"],
                "mtl": mtl_entry["mtl"]
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            next_id += 1

print(f"Done. Output written to {output_path}")