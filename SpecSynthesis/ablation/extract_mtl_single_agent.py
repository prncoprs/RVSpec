import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def extract_and_merge_mtl_data(input_file: Path, output_file: Path):
    """
    Extract MTL data from raw_content and merge with documentation field.
    
    Args:
        input_file: Path to input JSONL file containing entries with raw_content
        output_file: Path to output JSONL file with properly formatted MTL data
    """
    
    processed_count = 0
    error_count = 0
    total_mtls = 0
    
    print(f" Processing file: {input_file}")
    print(f" Output file: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the input line
                entry = json.loads(line.strip())
                
                # Skip entries that don't have raw_content or already have proper mtl field
                if 'raw_content' not in entry:
                    if 'mtl' in entry:
                        # Already properly formatted, just write it out
                        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        processed_count += 1
                        if isinstance(entry.get('mtl'), list):
                            total_mtls += len(entry['mtl'])
                    continue
                
                # Parse the raw_content JSON
                try:
                    raw_data = json.loads(entry['raw_content'])
                except json.JSONDecodeError as e:
                    print(f"  Line {line_num}: Failed to parse raw_content JSON: {e}")
                    error_count += 1
                    continue
                
                # Create the new entry structure
                new_entry = {
                    "path": entry.get("path", raw_data.get("path", "")),
                    "section_ids": entry.get("section_ids", raw_data.get("section_ids", [])),
                    "documentation": entry.get("documentation", "")
                }
                
                # Extract MTL data from raw_content
                if "results" in raw_data:
                    # Format: {"results": [{"policy": "...", "mtl": "..."}, ...]}
                    mtl_list = []
                    for result in raw_data["results"]:
                        if isinstance(result, dict) and "mtl" in result:
                            mtl_item = {
                                "mtl": result["mtl"]
                            }
                            # Include policy if available
                            if "policy" in result:
                                mtl_item["policy"] = result["policy"]
                            mtl_list.append(mtl_item)
                    
                    new_entry["mtl"] = mtl_list
                    total_mtls += len(mtl_list)
                    
                elif "mtl" in raw_data:
                    # Format: {"mtl": [...]}
                    new_entry["mtl"] = raw_data["mtl"]
                    if isinstance(raw_data["mtl"], list):
                        total_mtls += len(raw_data["mtl"])
                    else:
                        total_mtls += 1
                        
                else:
                    # No MTL found, mark as error
                    new_entry["error"] = "No MTL data found in raw_content"
                    error_count += 1
                
                # Write the processed entry
                outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: Failed to parse line JSON: {e}")
                error_count += 1
            except Exception as e:
                print(f"  Line {line_num}: Unexpected error: {e}")
                error_count += 1
    
    # Print summary
    print(f"\n Processing completed!")
    print(f" Total lines processed: {processed_count}")
    print(f" Total MTLs extracted: {total_mtls}")
    print(f" Errors encountered: {error_count}")
    print(f" Output saved to: {output_file.resolve()}")

def process_directory(input_dir: Path, output_dir: Path):
    """
    Process all JSONL files in a directory.
    
    Args:
        input_dir: Directory containing JSONL files to process
        output_dir: Directory to save processed files
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f" No JSONL files found in {input_dir}")
        return
    
    print(f" Found {len(jsonl_files)} JSONL files to process")
    
    for jsonl_file in jsonl_files:
        output_file = output_dir / f"processed_{jsonl_file.name}"
        print(f"\n Processing: {jsonl_file.name}")
        extract_and_merge_mtl_data(jsonl_file, output_file)

def process_json_files_to_jsonl(input_dir: Path, output_jsonl: Path):
    """
    Process all JSON files in a directory and combine into a single JSONL file.
    
    Args:
        input_dir: Directory containing JSON files to process
        output_jsonl: Output JSONL file path
    """
    
    input_dir = Path(input_dir)
    output_jsonl = Path(output_jsonl)
    
    # Create output directory if it doesn't exist
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f" No JSON files found in {input_dir}")
        return
    
    print(f" Found {json_files.__len__()} JSON files to process")
    
    processed_count = 0
    error_count = 0
    total_mtls = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            try:
                print(f" Processing: {json_file.name}")
                
                with open(json_file, 'r', encoding='utf-8') as infile:
                    entry = json.load(infile)
                
                # Skip entries that don't have raw_content or already have proper mtl field
                if 'raw_content' not in entry:
                    if 'mtl' in entry:
                        # Already properly formatted, just write it out
                        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        processed_count += 1
                        if isinstance(entry.get('mtl'), list):
                            total_mtls += len(entry['mtl'])
                    continue
                
                # Parse the raw_content JSON
                try:
                    raw_data = json.loads(entry['raw_content'])
                except json.JSONDecodeError as e:
                    print(f"  {json_file.name}: Failed to parse raw_content JSON: {e}")
                    error_count += 1
                    continue
                
                # Create the new entry structure
                new_entry = {
                    "path": entry.get("path", raw_data.get("path", "")),
                    "section_ids": entry.get("section_ids", raw_data.get("section_ids", [])),
                    "documentation": entry.get("documentation", "")
                }
                
                # Extract MTL data from raw_content
                if "results" in raw_data:
                    # Format: {"results": [{"policy": "...", "mtl": "..."}, ...]}
                    mtl_list = []
                    for result in raw_data["results"]:
                        if isinstance(result, dict) and "mtl" in result:
                            mtl_item = {
                                "mtl": result["mtl"]
                            }
                            # Include policy if available
                            if "policy" in result:
                                mtl_item["policy"] = result["policy"]
                            mtl_list.append(mtl_item)
                    
                    new_entry["mtl"] = mtl_list
                    total_mtls += len(mtl_list)
                    
                elif "mtl" in raw_data:
                    # Format: {"mtl": [...]}
                    new_entry["mtl"] = raw_data["mtl"]
                    if isinstance(raw_data["mtl"], list):
                        total_mtls += len(raw_data["mtl"])
                    else:
                        total_mtls += 1
                        
                else:
                    # No MTL found, mark as error
                    new_entry["error"] = "No MTL data found in raw_content"
                    error_count += 1
                
                # Write the processed entry
                outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  {json_file.name}: Failed to parse JSON file: {e}")
                error_count += 1
            except Exception as e:
                print(f"  {json_file.name}: Unexpected error: {e}")
                error_count += 1
    
    # Print summary
    print(f"\n Processing completed!")
    print(f" Total files processed: {processed_count}")
    print(f" Total MTLs extracted: {total_mtls}")
    print(f" Errors encountered: {error_count}")
    print(f" Output saved to: {output_jsonl.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Extract MTL data from raw_content and merge with documentation")
    parser.add_argument("input", help="Input JSONL file or directory")
    parser.add_argument("output", help="Output JSONL file or directory")
    parser.add_argument("--directory", "-d", action="store_true", 
                       help="Process all JSONL files in input directory")
    parser.add_argument("--json-to-jsonl", action="store_true",
                       help="Process JSON files from directory to single JSONL file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.json_to_jsonl:
        if not input_path.is_dir():
            print(f" Input path is not a directory: {input_path}")
            return
        process_json_files_to_jsonl(input_path, output_path)
    elif args.directory:
        if not input_path.is_dir():
            print(f" Input path is not a directory: {input_path}")
            return
        process_directory(input_path, output_path)
    else:
        if not input_path.is_file():
            print(f" Input file not found: {input_path}")
            return
        extract_and_merge_mtl_data(input_path, output_path)

if __name__ == "__main__":
    # Example usage if run without arguments
    import sys
    
    if len(sys.argv) == 1:
        print(" Example usage:")
        print("  python extract_merge.py input.jsonl output.jsonl")
        print("  python extract_merge.py input_dir/ output_dir/ --directory")
        print("  python extract_merge.py input_dir/ output.jsonl --json-to-jsonl")
        print("\n Running with your specific paths...")
        
        # Your specific case
        input_dir = Path("./single_agent_mtls_ablation/gpt-4o")
        output_file = Path("./single_agent_mtls_ablation/extracted_single_agent_mtls.jsonl")
        
        if input_dir.exists():
            print(f" Processing JSON files from: {input_dir}")
            print(f" Output will be saved to: {output_file}")
            process_json_files_to_jsonl(input_dir, output_file)
        else:
            print(f" Input directory not found: {input_dir}")
            print(" Creating test file instead...")
            
            # Create a test file if directory doesn't exist
            test_input = Path("test_input.jsonl")
            test_output = Path("processed_output.jsonl")
            
            test_data = {
                "path": "<RVSPEC_ROOT>/SpecSynthesis/doc/ardupilot_doc/common/source/docs/common-arkflow.txt",
                "section_ids": ["common-arkflow__0", "common-arkflow__1"],
                "documentation": "ARK Flow Open Source Optical Flow and Distance Sensor...",
                "error": "Invalid JSON format: Missing or malformed 'mtl'.",
                "raw_content": '{"path": "<RVSPEC_ROOT>/SpecSynthesis/doc/ardupilot_doc/common/source/docs/common-arkflow.txt", "section_ids": ["common-arkflow__0"], "results": [{"policy": "1. If the ARK Flow sensor is connected...", "mtl": "{ (Sensor_connected_to_CAN_t = TRUE) → (FLOW_TYPE_t = 6) }"}]}'
            }
            with open(test_input, 'w', encoding='utf-8') as f:
                f.write(json.dumps(test_data, ensure_ascii=False) + '\n')
            print(f" Created test file: {test_input}")
            
            extract_and_merge_mtl_data(test_input, test_output)
    else:
        main()