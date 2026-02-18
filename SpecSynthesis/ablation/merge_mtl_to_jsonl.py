import json
import os
import argparse
from pathlib import Path

def merge_json_to_jsonl(input_folder, output_file, recursive=False, add_filename=False):
    """
    Merge all JSON files in a folder into a single JSONL file.
    
    Args:
        input_folder (str): Path to folder containing JSON files
        output_file (str): Path for output JSONL file
        recursive (bool): Search subdirectories recursively
        add_filename (bool): Add filename as a field in each JSON object
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Get JSON files (recursive or not)
    if recursive:
        json_files = list(input_path.rglob("*.json"))
    else:
        json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_folder}'")
        return
    
    print(f"Found {len(json_files)} JSON files to merge...")
    
    processed_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                
                # Optionally add filename to the data
                if add_filename:
                    if isinstance(data, dict):
                        data['_source_file'] = json_file.name
                    elif isinstance(data, list):
                        # Handle arrays by wrapping in an object
                        data = {'_source_file': json_file.name, 'data': data}
                
                # Write as single line to JSONL file
                json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))
                outfile.write('\n')
                
                processed_count += 1
                print(f"Processed: {json_file.name}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing {json_file.name}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                error_count += 1
    
    print(f"\nMerge complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge JSON files into a JSONL file')
    parser.add_argument('input_folder', help='Input folder containing JSON files')
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Search subdirectories recursively')
    parser.add_argument('-f', '--add-filename', action='store_true',
                       help='Add source filename to each JSON object')
    
    args = parser.parse_args()
    
    merge_json_to_jsonl(args.input_folder, args.output_file, 
                       args.recursive, args.add_filename)

if __name__ == "__main__":
    main()