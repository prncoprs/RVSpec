import json
import re
from pathlib import Path

MODELS = ["gpt-4o", "gpt-4.1", "claude-sonnet-4-20250514"]  # Support both GPT and Claude models

def extract_valid_logic_policies_from_raw(raw):
    """Extract logic policies from raw_content field (typically from GPT models)"""
    try:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        elif raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        
        data = json.loads(raw)
        policies = data.get("logic_policies", [])
        if isinstance(policies, list) and policies:
            return policies
    except Exception:
        return None
    return None

def extract_valid_logic_policies_from_structured(entry):
    """Extract logic policies from structured response (typically from Claude models)"""
    try:
        policies = entry.get("logic_policies", [])
        if isinstance(policies, list) and policies:
            return policies
    except Exception:
        return None
    return None

def extract_logic_policies(entry):
    """Universal function to extract logic policies from either GPT or Claude responses"""
    # First, try to get policies from structured format (Claude style)
    policies = extract_valid_logic_policies_from_structured(entry)
    if policies:
        return policies, "structured"
    
    # If not found, try to extract from raw_content (GPT style)
    raw = entry.get("raw_content")
    if raw:
        policies = extract_valid_logic_policies_from_raw(raw)
        if policies:
            return policies, "raw"
    
    return None, None

def clean_policy_text(policy):
    """Remove leading numbering like '1. ', '2. ' etc. and clean whitespace"""
    # Remove leading numbering
    cleaned = re.sub(r"^\s*\d+\.\s*", "", policy).strip()
    # Remove any trailing periods if they seem to be artifacts
    # cleaned = re.sub(r"\.$", "", cleaned).strip()
    return cleaned

def is_claude_model(model_name):
    """Check if the model is a Claude model"""
    return model_name.startswith("claude-")

def is_gpt_model(model_name):
    """Check if the model is a GPT model"""
    return model_name.startswith(("gpt-", "o1-"))

def process_model(model_name):
    ### cz42: Adjust paths for cFS
    model_dir = Path(f"./logic_policies-cFS/{model_name}")
    input_path = model_dir / f"combined_{model_name}.jsonl"
    output_path = Path("./logic_policies-cFS") / f"extracted_logic_policies_{model_name}.jsonl"

    if not input_path.exists():
        print(f"[!] Skipping {model_name}: {input_path} does not exist.")
        return

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        count_valid = 0
        count_total = 0
        count_structured = 0
        count_raw = 0
        count_errors = 0

        for line in fin:
            count_total += 1
            try:
                entry = json.loads(line)
                
                # Check for error entries first
                if "error" in entry:
                    count_errors += 1
                    continue
                
                # Extract policies using universal method
                policies, source_type = extract_logic_policies(entry)
                
                if policies:
                    # Track extraction method
                    if source_type == "structured":
                        count_structured += 1
                    elif source_type == "raw":
                        count_raw += 1
                    
                    # Write each policy as a separate entry
                    for idx, policy in enumerate(policies, start=1):
                        cleaned_policy = clean_policy_text(policy)
                        if cleaned_policy:  # Only write non-empty policies
                            output_entry = {
                                "path": entry["path"],
                                "section_ids": entry["section_ids"],
                                "policy_id": idx,
                                "policy": cleaned_policy,
                                "model": model_name,
                                "extraction_method": source_type
                            }
                            fout.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                    
                    count_valid += 1
                    
            except json.JSONDecodeError as e:
                print(f"[!] JSON decode error in {model_name}: {e}")
                continue
            except Exception as e:
                print(f"[!] Unexpected error processing entry in {model_name}: {e}")
                continue

    # Print detailed statistics
    print(f" [{model_name}] Processing completed:")
    print(f"    Total entries: {count_total}")
    print(f"    Valid extractions: {count_valid}")
    print(f"    Error entries: {count_errors}")
    if is_claude_model(model_name):
        print(f"    Structured extractions: {count_structured}")
        print(f"    Raw content extractions: {count_raw}")
    elif is_gpt_model(model_name):
        print(f"    Raw content extractions: {count_raw}")
        print(f"    Structured extractions: {count_structured}")
    print(f"    Output saved to: {output_path.resolve()}")
    print()

def main():
    print(" Extracting logic policies from model outputs...")
    print(f" Processing models: {MODELS}")
    print()
    
    total_models_processed = 0
    
    for model in MODELS:
        print(f" Processing {model}...")
        process_model(model)
        total_models_processed += 1
    
    print(f" Completed processing {total_models_processed} model(s)!")

if __name__ == "__main__":
    main()