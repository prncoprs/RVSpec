import os
import re
from pathlib import Path
import json
import hashlib
from collections import defaultdict
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import multiprocessing


# ArduPilot .rst documentation directory
ardupilot_rst_path = Path("<RVSPEC_ROOT>/SpecSynthesis/raw_doc_repo/ardupilot_wiki").resolve()
ardupilot_output_path = Path("doc/ardupilot_doc").resolve()

# PX4 English .md documentation directory
px4_md_path = Path("<RVSPEC_ROOT>/SpecSynthesis/raw_doc_repo/PX4-user_guide/en").resolve()
px4_output_path = Path("doc/px4_doc").resolve()



def read_rst_content(content):
    return content.strip()  # Keep raw .rst content

def read_md_content(content):
    return content.strip()  # Keep raw .md content

count_rst = 0
for root, _, files in os.walk(ardupilot_rst_path):
    for filename in files:
        if filename.endswith(".rst"):
            input_path = Path(root) / filename
            relative_path = input_path.relative_to(ardupilot_rst_path)
            output_path = ardupilot_output_path / relative_path.with_suffix(".txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(input_path, "r", encoding="utf-8") as infile:
                    raw_content = infile.read()
                content = read_rst_content(raw_content)
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(content)
                count_rst += 1
            except Exception as e:
                print(f"[!] Error processing {input_path}: {e}")

print(f" Processed {count_rst} ArduPilot .rst files.")


count_md = 0
for root, _, files in os.walk(px4_md_path):
    for filename in files:
        if filename.endswith(".md"):
            input_path = Path(root) / filename
            relative_path = input_path.relative_to(px4_md_path)
            output_path = px4_output_path / relative_path.with_suffix(".txt")
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

print(f" Processed {count_md} PX4 .md files.")



# Input directories
ardupilot_txt_dir = Path("doc/ardupilot_doc").resolve()
px4_txt_dir = Path("doc/px4_doc").resolve()

# Output directory and file
output_dir = Path("split_sections").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
output_jsonl_path = output_dir / "split_sections.jsonl"

def split_rst_sections(content):
    section_regex = r'(?:\n|^)([^\n]+)\n[=~\-`^]{3,}\n'
    parts = re.split(section_regex, content)
    if len(parts) < 2:
        return [content.strip()]
    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip()
        section_text = f"{title}\n{body}"
        sections.append(section_text)
    return sections

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
    
    if file_path.suffix == ".txt" and "px4_doc" in str(base_path):
        sections = split_md_sections(content)
    else:
        sections = split_rst_sections(content)
    
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

# Collect sections from both sources
sections = collect_all_sections(ardupilot_txt_dir) + collect_all_sections(px4_txt_dir)

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

print(f" Sorted {len(sections_sorted)} sections written to: {output_jsonl_path}")

