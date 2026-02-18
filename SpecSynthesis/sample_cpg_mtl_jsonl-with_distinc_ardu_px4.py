import json
import random
import re
from pathlib import Path
from collections import OrderedDict

# Configuration
ROOT_DIR = Path("mtl/cpg_mtl")
MODELS = ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4.1"]
TARGET_SIZE_PER_SYSTEM = 200
RANDOM_SEED = 42
PATH_PREFIX = "<RVSPEC_ROOT>/SpecSynthesis/"

def merge_section_ids(section_ids):
    if not section_ids:
        return section_ids
    pattern = r"(.+)__(\d+)"
    base = None
    nums = []
    for sid in section_ids:
        match = re.match(pattern, sid)
        if match:
            b, n = match.groups()
            if base is None:
                base = b
            nums.append(int(n))
        else:
            return section_ids  # Leave unmerged if format is unexpected
    nums = sorted(nums)
    return f"{base}__{nums[0]}-{nums[-1]}" if base else section_ids

def get_system_type(path):
    """Determine system type based on path"""
    if "px4_doc" in path:
        return "px4"
    elif "ardupilot_doc" in path:
        return "ardupilot"
    else:
        return "unknown"

def load_and_categorize_existing(sampled_file):
    """Load existing sampled data and categorize by system type"""
    existing_entries = []
    px4_entries = []
    ardupilot_entries = []
    px4_ids = []
    ardupilot_ids = []
    
    if sampled_file.exists():
        with open(sampled_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                # Split by double newlines to handle the formatting
                json_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
                for block in json_blocks:
                    try:
                        entry = json.loads(block)
                        existing_entries.append(entry)
                        system_type = get_system_type(entry.get("path", ""))
                        entry_id = entry.get("id", "unknown")
                        
                        if system_type == "px4":
                            px4_entries.append(entry)
                            px4_ids.append(entry_id)
                        elif system_type == "ardupilot":
                            ardupilot_entries.append(entry)
                            ardupilot_ids.append(entry_id)
                    except json.JSONDecodeError:
                        continue
    
    return existing_entries, px4_entries, ardupilot_entries, px4_ids, ardupilot_ids

def load_all_entries(input_file):
    """Load all entries from combined file and categorize by system type"""
    all_px4 = []
    all_ardupilot = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                system_type = get_system_type(entry.get("path", ""))
                if system_type == "px4":
                    all_px4.append(entry)
                elif system_type == "ardupilot":
                    all_ardupilot.append(entry)
            except json.JSONDecodeError:
                continue
    
    return all_px4, all_ardupilot

def sample_additional_entries(existing_entries, all_entries, target_size, seed_offset=0):
    """Sample additional entries to reach target size"""
    if len(existing_entries) >= target_size:
        return existing_entries[:target_size]
    
    needed = target_size - len(existing_entries)
    
    # Get existing paths to avoid duplicates
    existing_paths = set(entry.get("path", "") for entry in existing_entries)
    
    # Filter out existing entries
    available_entries = [entry for entry in all_entries 
                        if entry.get("path", "") not in existing_paths]
    
    if len(available_entries) < needed:
        print(f"[!] Warning: Only {len(available_entries)} new entries available, but {needed} needed")
        additional = available_entries
    else:
        random.seed(RANDOM_SEED + seed_offset)
        additional = random.sample(available_entries, needed)
    
    return existing_entries + additional

def create_ordered_entry(entry, entry_id):
    """Create an OrderedDict entry with proper formatting"""
    # Clean path
    if entry.get("path", "").startswith(PATH_PREFIX):
        entry["path"] = entry["path"].replace(PATH_PREFIX, "", 1)
    
    # Merge section_ids if needed
    if isinstance(entry.get("section_ids"), list):
        entry["section_ids"] = merge_section_ids(entry["section_ids"])
    
    # Create OrderedDict with ID first
    ordered = OrderedDict()
    ordered["id"] = entry_id
    for key in entry:
        if key != "id":
            ordered[key] = entry[key]
    
    return ordered

def save_id_distribution(model_name, px4_ids, ardupilot_ids):
    """Save ID distribution to a file"""
    distribution_file = ROOT_DIR / f"id_distribution_{model_name}.txt"
    
    with open(distribution_file, "w", encoding="utf-8") as f:
        f.write(f"ID Distribution for Model: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"PX4 System - {len(px4_ids)} entries:\n")
        f.write(f"IDs: {sorted(px4_ids)}\n\n")
        
        f.write(f"ArduPilot System - {len(ardupilot_ids)} entries:\n")
        f.write(f"IDs: {sorted(ardupilot_ids)}\n\n")
        
        # Also write a more readable format
        f.write("Detailed breakdown:\n")
        f.write("-" * 30 + "\n")
        
        all_ids = sorted(set(px4_ids + ardupilot_ids))
        for entry_id in all_ids:
            if entry_id in px4_ids:
                system = "PX4"
            elif entry_id in ardupilot_ids:
                system = "ArduPilot"
            else:
                system = "Unknown"
            f.write(f"ID {entry_id:3d}: {system}\n")
    
    print(f" ID distribution saved to: {distribution_file.resolve()}")

def write_system_file(entries, output_file, system_name):
    """Write entries to a system-specific file"""
    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(entries, start=1):
            ordered = create_ordered_entry(entry, idx)
            fout.write(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n\n")
    
    print(f" {system_name} sampling complete! {len(entries)} entries written to {output_file.resolve()}")

def enhanced_sample_jsonl(model_name):
    input_file = ROOT_DIR / f"combined_{model_name}.jsonl"
    sampled_file = ROOT_DIR / f"sampled_{model_name}.jsonl"
    
    # Define output files for each system
    px4_output_file = ROOT_DIR / f"enhanced_px4_sampled_{model_name}.jsonl"
    ardupilot_output_file = ROOT_DIR / f"enhanced_ardupilot_sampled_{model_name}.jsonl"

    if not input_file.exists():
        print(f"[!] Input file does not exist: {input_file}")
        return

    print(f"Loading existing sampled data from {sampled_file}")
    existing_entries, existing_px4, existing_ardupilot, px4_ids, ardupilot_ids = load_and_categorize_existing(sampled_file)
    
    print(f"Found {len(existing_entries)} existing entries:")
    print(f"  - PX4: {len(existing_px4)} entries")
    print(f"    PX4 IDs: {sorted(px4_ids)}")
    print(f"  - ArduPilot: {len(existing_ardupilot)} entries") 
    print(f"    ArduPilot IDs: {sorted(ardupilot_ids)}")
    
    # Save ID distribution to file
    save_id_distribution(model_name, px4_ids, ardupilot_ids)

    print(f"Loading all entries from {input_file}")
    all_px4, all_ardupilot = load_all_entries(input_file)
    
    print(f"Total available entries:")
    print(f"  - PX4: {len(all_px4)}")
    print(f"  - ArduPilot: {len(all_ardupilot)}")

    # Sample to reach target size for each system
    print(f"\nSampling to reach {TARGET_SIZE_PER_SYSTEM} entries per system...")
    
    final_px4 = sample_additional_entries(existing_px4, all_px4, TARGET_SIZE_PER_SYSTEM, seed_offset=0)
    final_ardupilot = sample_additional_entries(existing_ardupilot, all_ardupilot, TARGET_SIZE_PER_SYSTEM, seed_offset=1000)
    
    print(f"Final counts:")
    print(f"  - PX4: {len(final_px4)}")
    print(f"  - ArduPilot: {len(final_ardupilot)}")

    # Write separate files for each system
    write_system_file(final_px4, px4_output_file, "PX4")
    write_system_file(final_ardupilot, ardupilot_output_file, "ArduPilot")
    
    print(f"\n All files generated successfully!")
    print(f"   - PX4 file: {px4_output_file.resolve()}")
    print(f"   - ArduPilot file: {ardupilot_output_file.resolve()}")

def main():
    print("=" * 80)
    print("Enhanced Sampling for Multiple Models")
    print("=" * 80)
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n{'='*20} Processing Model {i}/{len(MODELS)}: {model} {'='*20}")
        enhanced_sample_jsonl(model)
        
    print(f"\n{'='*80}")
    print(" ALL MODELS PROCESSED SUCCESSFULLY!")
    print(f"{'='*80}")
    
    # Summary
    print("\nFiles generated:")
    for model in MODELS:
        px4_file = ROOT_DIR / f"enhanced_px4_sampled_{model}.jsonl"
        ardupilot_file = ROOT_DIR / f"enhanced_ardupilot_sampled_{model}.jsonl"
        distribution_file = ROOT_DIR / f"id_distribution_{model}.txt"
        print(f"  {model}:")
        print(f"    - PX4: {px4_file.name}")
        print(f"    - ArduPilot: {ardupilot_file.name}")
        print(f"    - ID Distribution: {distribution_file.name}")

if __name__ == "__main__":
    main()