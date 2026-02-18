import os
import re
from pathlib import Path
import json

# cFS documentation directories
cfs_md_paths = [
    Path("<RVSPEC_ROOT>/UAV/nos3/docs/wiki").resolve(),
    Path("<RVSPEC_ROOT>/UAV/nos3/fsw/cfe/docs").resolve(),
    Path("<RVSPEC_ROOT>/UAV/nos3/fsw/osal/docs").resolve()
]

# cFS output base directory
cfs_output_base = Path("doc/cFS_doc").resolve()

# Base path for calculating relative paths
base_uav_path = Path("<RVSPEC_ROOT>/UAV").resolve()

def read_md_content(content):
    return content.strip()  # Keep raw .md content

# Process cFS files
count_md = 0
for md_path in cfs_md_paths:
    if not md_path.exists():
        print(f"[!] Warning: {md_path} does not exist, skipping...")
        continue
        
    print(f"Processing: {md_path}")
    
    for root, _, files in os.walk(md_path):
        for filename in files:
            if filename.endswith(".md"):
                input_path = Path(root) / filename
                
                # Calculate relative path from the base UAV directory
                relative_from_uav = input_path.relative_to(base_uav_path)
                output_path = cfs_output_base / relative_from_uav.with_suffix(".txt")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    with open(input_path, "r", encoding="utf-8") as infile:
                        raw_content = infile.read()
                    content = read_md_content(raw_content)
                    with open(output_path, "w", encoding="utf-8") as outfile:
                        outfile.write(content)
                    count_md += 1
                except Exception as e:
                    print(f"[!] Error processing {input_path}: {e}")

print(f" Processed {count_md} cFS .md files.")

# Input directory
cfs_txt_dir = Path("doc/cFS_doc").resolve()

# Output directory and file
output_dir = Path("split_sections").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
output_jsonl_path = output_dir / "split_sections-cFS.jsonl"

def split_md_sections(content):
    matches = list(re.finditer(r'^(#{1,6})\s+(.+)', content, flags=re.MULTILINE))
    if not matches:
        return [content.strip()]
    sections = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end].strip()
        sections.append(section)
    return sections

def get_sections_from_file(file_path, base_path):
    relative_path = file_path.relative_to(base_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Use markdown splitting for cFS files
    sections = split_md_sections(content)
    return relative_path.as_posix(), sections

def collect_all_sections(doc_dir):
    section_data = []
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = Path(root) / filename
                try:
                    relative_path, sections = get_sections_from_file(file_path, doc_dir)
                    for idx, section in enumerate(sections):
                        section_data.append({
                            "path": str(file_path),
                            "section_id": f"{filename[:-4]}__{idx}",
                            "text": section
                        })
                except Exception as e:
                    print(f"[!] Error processing {file_path}: {e}")
    return section_data

# Collect sections from cFS
sections = collect_all_sections(cfs_txt_dir)

# Sort by path and section index
def sort_key(entry):
    section_match = re.search(r'__(\d+)$', entry['section_id'])
    section_idx = int(section_match.group(1)) if section_match else 0
    return (entry['path'], section_idx)

sections_sorted = sorted(sections, key=sort_key)

# Write to JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as out:
    for entry in sections_sorted:
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f" Sorted {len(sections_sorted)} cFS sections written to: {output_jsonl_path}")