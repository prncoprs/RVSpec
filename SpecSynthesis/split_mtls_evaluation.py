import pandas as pd
import json
from pathlib import Path

# Configuration
ROOT_DIR = Path("mtl/cpg_mtl")
CSV_DIR = Path("mtl/mtl_evaluation")  # All CSV files will be in this directory
MODELS = ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]

def load_id_distribution(model_name):
    """Load ID distribution from the distribution file"""
    distribution_file = ROOT_DIR / f"id_distribution_{model_name}.txt"
    px4_ids = []
    ardupilot_ids = []
    
    if not distribution_file.exists():
        print(f"[!] Distribution file not found: {distribution_file}")
        return px4_ids, ardupilot_ids
    
    with open(distribution_file, "r", encoding="utf-8") as f:
        content = f.read()
        
        # Parse the detailed breakdown section
        in_breakdown = False
        for line in content.split('\n'):
            if "Detailed breakdown:" in line:
                in_breakdown = True
                continue
            if in_breakdown and line.strip():
                if line.startswith("ID"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            id_num = int(parts[0].replace("ID", "").strip())
                            system = parts[1].strip()
                            if system == "PX4":
                                px4_ids.append(id_num)
                            elif system == "ArduPilot":
                                ardupilot_ids.append(id_num)
                        except ValueError:
                            continue
    
    return sorted(px4_ids), sorted(ardupilot_ids)

def load_enhanced_jsonl(model_name, system_type):
    """Load the enhanced sampled JSONL file for a specific system"""
    if system_type.lower() == "px4":
        jsonl_file = ROOT_DIR / f"enhanced_px4_sampled_{model_name}.jsonl"
    elif system_type.lower() == "ardupilot":
        jsonl_file = ROOT_DIR / f"enhanced_ardupilot_sampled_{model_name}.jsonl"
    else:
        return []
    
    if not jsonl_file.exists():
        print(f"[!] Enhanced JSONL file not found: {jsonl_file}")
        return []
    
    entries = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            json_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
            for block in json_blocks:
                try:
                    entry = json.loads(block)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return entries

def process_model_evaluation_improved(model_name):
    """Improved version that matches entries by content rather than position"""
    # Map model names to CSV files
    csv_mapping = {
        "claude-sonnet-4-20250514": "cpg_mtl_evaluation(claude-sonnet-4-20250514).csv",
        "gpt-4o": "cpg_mtl_evaluation(gpt-4o).csv", 
        "gpt-4.1": "cpg_mtl_evaluation(gpt-41).csv"
    }
    
    csv_filename = csv_mapping.get(model_name)
    if not csv_filename:
        print(f"[!] No CSV mapping found for model: {model_name}")
        return
    
    csv_file = CSV_DIR / csv_filename
    
    print(f"\nProcessing {model_name}...")
    print(f"Looking for CSV: {csv_file}")
    print(f"CSV_DIR exists: {CSV_DIR.exists()}")
    
    if CSV_DIR.exists():
        print(f"Files in CSV_DIR:")
        for file in CSV_DIR.iterdir():
            print(f"  - {file.name}")
    
    if not csv_file.exists():
        print(f"[!] CSV file not found: {csv_file.resolve()}")
        return
    
    print(f"Loading CSV: {csv_file}")
    
    # Load the evaluation CSV
    df = pd.read_csv(csv_file)
    
    # Load ID distribution
    px4_ids, ardupilot_ids = load_id_distribution(model_name)
    print(f"  Original PX4 IDs: {len(px4_ids)}")
    print(f"  Original ArduPilot IDs: {len(ardupilot_ids)}")
    
    # Load enhanced JSONL entries
    px4_entries = load_enhanced_jsonl(model_name, "px4")
    ardupilot_entries = load_enhanced_jsonl(model_name, "ardupilot")
    print(f"  Enhanced PX4 entries: {len(px4_entries)}")
    print(f"  Enhanced ArduPilot entries: {len(ardupilot_entries)}")
    
    # Load original sampled data to match content
    original_file = ROOT_DIR / f"sampled_{model_name}.jsonl"
    original_entries = {}
    
    if original_file.exists():
        with open(original_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                json_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
                for block in json_blocks:
                    try:
                        entry = json.loads(block)
                        original_entries[entry['id']] = entry
                    except json.JSONDecodeError:
                        continue
    
    # Create mapping from original ID to evaluation data
    evaluation_map = {}
    for _, row in df.iterrows():
        mtl_id = int(row['MTLs'])
        evaluation_map[mtl_id] = {
            'Syntactic Validity': row['Syntactic Validity'],
            'Semantic Accuracy': row['Semantic Accuracy '],
            'Cyber-physical Consistency': row['Cyber-physical Consistency'],
            'Overall Correctness': row['Overall Correctness ']
        }
    
    def create_system_data(entries, original_ids):
        """Create data for a specific system"""
        data = []
        for entry in entries:
            new_id = entry['id']
            evaluated = False
            eval_data = {'Syntactic Validity': '', 'Semantic Accuracy': '', 
                        'Cyber-physical Consistency': '', 'Overall Correctness': ''}
            
            # Try to match with original entries by content (MTL, policy, etc.)
            for orig_id in original_ids:
                if orig_id in original_entries and orig_id in evaluation_map:
                    orig_entry = original_entries[orig_id]
                    # Match by MTL content or policy content
                    if (entry.get('mtl') == orig_entry.get('mtl') or 
                        entry.get('policy') == orig_entry.get('policy')):
                        evaluated = True
                        eval_data = evaluation_map[orig_id]
                        break
            
            row_data = {
                'MTLs': new_id,
                'Models': model_name,
                'Syntactic Validity': eval_data.get('Syntactic Validity', ''),
                'Semantic Accuracy': eval_data.get('Semantic Accuracy', ''),
                'Cyber-physical Consistency': eval_data.get('Cyber-physical Consistency', ''),
                'Overall Correctness': eval_data.get('Overall Correctness', ''),
                'Previously_Evaluated': evaluated
            }
            data.append(row_data)
        
        return data
    
    # Process both systems
    px4_data = create_system_data(px4_entries, px4_ids)
    ardupilot_data = create_system_data(ardupilot_entries, ardupilot_ids)
    
    # Create DataFrames
    px4_df = pd.DataFrame(px4_data)
    ardupilot_df = pd.DataFrame(ardupilot_data)
    
    # Save to CSV files
    px4_output = CSV_DIR / f"cpg_mtl_evaluation_px4_{model_name.replace('-', '_')}.csv"
    ardupilot_output = CSV_DIR / f"cpg_mtl_evaluation_ardupilot_{model_name.replace('-', '_')}.csv"
    
    # Ensure output directory exists
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    
    px4_df.to_csv(px4_output, index=False)
    ardupilot_df.to_csv(ardupilot_output, index=False)
    
    # Print statistics
    px4_evaluated = px4_df['Previously_Evaluated'].sum()
    px4_unevaluated = len(px4_df) - px4_evaluated
    ardupilot_evaluated = ardupilot_df['Previously_Evaluated'].sum()
    ardupilot_unevaluated = len(ardupilot_df) - ardupilot_evaluated
    
    print(f"   PX4 CSV saved: {px4_output}")
    print(f"     - Previously evaluated: {px4_evaluated}")
    print(f"     - Need evaluation: {px4_unevaluated}")
    print(f"   ArduPilot CSV saved: {ardupilot_output}")
    print(f"     - Previously evaluated: {ardupilot_evaluated}")
    print(f"     - Need evaluation: {ardupilot_unevaluated}")

def main():
    print("=" * 80)
    print("Splitting Evaluation CSV by System Type")
    print("=" * 80)
    
    for model in MODELS:
        process_model_evaluation_improved(model)
    
    print(f"\n{'='*80}")
    print(" ALL EVALUATION CSVS PROCESSED!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()