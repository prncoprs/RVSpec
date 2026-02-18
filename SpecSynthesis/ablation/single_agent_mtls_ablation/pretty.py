import json
from collections import OrderedDict

input_path = "split_mtls.jsonl"
output_path = "split_mtls_pretty.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin, start=1):
        original_data = json.loads(line)
        
        # Insert "id" as the first field
        new_data = OrderedDict()
        new_data["id"] = idx
        for k, v in original_data.items():
            new_data[k] = v

        # Pretty-print with indent=2
        pretty_line = json.dumps(new_data, indent=2, ensure_ascii=False)
        fout.write(pretty_line + "\n")
