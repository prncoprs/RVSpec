#!/usr/bin/env python3
"""
Format JSONL file with proper indentation and blank lines between records.
"""

import json
import sys

def format_jsonl(input_file, output_file):
    """
    Read a JSONL file and reformat it with:
    - JSON indentation (4 spaces)
    - Blank lines between each JSON object
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output formatted file
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()
            total_lines = len(lines)
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse JSON and reformat with indentation
                    json_obj = json.loads(line)
                    formatted_json = json.dumps(json_obj, indent=4, ensure_ascii=False)
                    
                    # Write formatted JSON
                    outfile.write(formatted_json)
                    
                    # Add blank line between objects (except after the last one)
                    if i < total_lines - 1:
                        outfile.write('\n\n')
                    else:
                        outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i+1}: {e}")
                    print(f"Problematic line: {line[:100]}...")
                    continue
            
            print(f"Successfully formatted {input_file} -> {output_file}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    # Define input and output file names
    input_file = "combined_claude-sonnet-4-20250514.jsonl"
    output_file = "formatted_combined_claude-sonnet-4-20250514.jsonl"  # 'formatted' is correct
    
    print(f"Formatting JSONL file...")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    format_jsonl(input_file, output_file)

if __name__ == "__main__":
    main()