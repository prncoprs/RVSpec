import os
import re
from pathlib import Path
import json
from bs4 import BeautifulSoup

# OpenPilot HTML documentation directory
openpilot_html_path = Path("<RVSPEC_ROOT>/SpecSynthesis/raw_doc_repo/openpilot-docs").resolve()
openpilot_output_path = Path("doc/openpilot_doc").resolve()

def clean_html_content(content):
    """Clean HTML content using BeautifulSoup"""
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside", "meta", "link"]):
        element.decompose()
    
    # Get text with line breaks
    text = soup.get_text(separator='\n', strip=True)
    
    # Clean up extra whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def read_html_content(content):
    return clean_html_content(content)

# Process OpenPilot HTML files
count_html = 0
for root, _, files in os.walk(openpilot_html_path):
    for filename in files:
        if filename.endswith(".html") or filename.endswith(".htm"):
            input_path = Path(root) / filename
            relative_path = input_path.relative_to(openpilot_html_path)
            output_path = openpilot_output_path / relative_path.with_suffix(".txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(input_path, "r", encoding="utf-8") as infile:
                    raw_content = infile.read()
                content = read_html_content(raw_content)
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(content)
                count_html += 1
            except Exception as e:
                print(f"[!] Error processing {input_path}: {e}")

print(f" Processed {count_html} OpenPilot .html files.")

# Input directory
openpilot_txt_dir = Path("doc/openpilot_doc").resolve()

# Output directory and file
output_dir = Path("split_sections").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
openpilot_jsonl_path = output_dir / "split_sections-openpilot.jsonl"

def split_html_sections(content):
    """Split HTML-converted content by paragraphs or logical breaks"""
    # Split by double line breaks (paragraph breaks)
    sections = [section.strip() for section in content.split('\n\n') if section.strip()]
    
    # If no clear paragraph breaks, split by single line breaks but keep longer chunks
    if len(sections) == 1:
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if line:
                current_section.append(line)
            else:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
        
        if current_section:
            sections.append('\n'.join(current_section))
    
    return sections if sections else [content.strip()]

def get_sections_from_file(file_path, base_path):
    relative_path = file_path.relative_to(base_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = split_html_sections(content)
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

# Collect sections from OpenPilot
openpilot_sections = collect_all_sections(openpilot_txt_dir)

# Sort by path and section index
def sort_key(entry):
    section_match = re.search(r'__(\d+)$', entry['section_id'])
    section_idx = int(section_match.group(1)) if section_match else 0
    return (entry['path'], section_idx)

openpilot_sections_sorted = sorted(openpilot_sections, key=sort_key)

# Write to OpenPilot-specific JSONL
with open(openpilot_jsonl_path, "w", encoding="utf-8") as out:
    for entry in openpilot_sections_sorted:
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f" Sorted {len(openpilot_sections_sorted)} OpenPilot sections written to: {openpilot_jsonl_path}")