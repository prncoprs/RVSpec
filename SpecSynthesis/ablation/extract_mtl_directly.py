import json
import os
from pathlib import Path
from collections import defaultdict

def extract_mtls_from_json_files(input_dir, output_dir, model_name):
    """
    Extract MTLs from individual JSON files and combine them into a single JSONL file.
    
    Args:
        input_dir (Path): Directory containing individual JSON files
        output_dir (Path): Directory to save the combined JSONL file
        model_name (str): Name of the model (e.g., 'gpt-4o')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all MTLs
    all_mtls = []
    total_files = 0
    files_with_mtls = 0
    total_mtl_count = 0
    
    print(f" Scanning {input_path} for JSON files...")
    
    # Process all JSON files in the directory
    for json_file in input_path.glob("*.json"):
        # Skip metadata files
        if json_file.name.startswith("ablation_metadata"):
            continue
            
        total_files += 1
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if the file contains MTLs
            if "mtl" in data and isinstance(data["mtl"], list) and len(data["mtl"]) > 0:
                files_with_mtls += 1
                
                # Extract each MTL and add metadata
                for mtl in data["mtl"]:
                    mtl_entry = {
                        "path": data.get("path", ""),
                        "model": model_name,
                        "section_ids": data.get("section_ids", []),
                        "documentation": data.get("documentation", ""),
                        "mtl": mtl
                    }
                    all_mtls.append(mtl_entry)
                    total_mtl_count += 1
                
                print(f" Extracted {len(data['mtl'])} MTLs from {json_file.name}")
            
            elif "error" in data:
                print(f"  Skipped {json_file.name}: {data['error']}")
            else:
                print(f"  Skipped {json_file.name}: No MTLs found")
                
        except Exception as e:
            print(f" Error processing {json_file.name}: {e}")
    
    # Sort MTLs by source path for consistency
    all_mtls.sort(key=lambda x: (x["path"], x["section_ids"]))
    
    # Save to JSONL file
    output_file = output_path / f"extracted_mtls_{model_name}.jsonl"
    
    print(f"\n Saving {total_mtl_count} MTLs to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for mtl_entry in all_mtls:
            f.write(json.dumps(mtl_entry, ensure_ascii=False) + "\n")
    
    # Print summary
    print(f"\n Extraction Complete!")
    print(f" Summary:")
    print(f"   - Total JSON files processed: {total_files}")
    print(f"   - Files with MTLs: {files_with_mtls}")
    print(f"   - Total MTLs extracted: {total_mtl_count}")
    print(f"   - Output file: {output_file.resolve()}")
    
    return total_mtl_count, output_file

def extract_mtls_for_all_models(base_dir="./direct_mtls_ablation"):
    """
    Extract MTLs for all models found in the base directory.
    
    Args:
        base_dir (str): Base directory containing model subdirectories
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f" Base directory {base_path} does not exist!")
        return
    
    # Find all model directories
    model_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not model_dirs:
        print(f" No model directories found in {base_path}")
        return
    
    print(f" Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"   - {model_dir.name}")
    
    total_mtls_all_models = 0
    
    # Process each model directory
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n Processing model: {model_name}")
        
        try:
            mtl_count, output_file = extract_mtls_from_json_files(
                input_dir=model_dir,
                output_dir=base_path,
                model_name=model_name
            )
            total_mtls_all_models += mtl_count
            
        except Exception as e:
            print(f" Error processing {model_name}: {e}")
    
    print(f"\n Overall Summary:")
    print(f"   - Total MTLs across all models: {total_mtls_all_models}")
    print(f"   - Output directory: {base_path.resolve()}")

def main():
    """
    Main function to extract MTLs from all model directories.
    """
    print(" Starting MTL extraction...")
    extract_mtls_for_all_models()

if __name__ == "__main__":
    main()