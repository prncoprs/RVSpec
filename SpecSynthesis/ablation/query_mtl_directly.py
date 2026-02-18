import os
import re
import json
import time
import signal
import hashlib
import threading
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from openai import OpenAI, OpenAIError
import tiktoken

from tqdm import tqdm


# ========== Environment Setup ==========
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
# Ensure the OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")


    
    
# ========== Configuration ==========
MODELS = ["gpt-4o", "gpt-4.1"]
MODELS = ["gpt-4o"]
MAX_WORKERS = max(1, os.cpu_count() // 2)
RETRY_LIMIT = 3
RETRY_DELAY = 2
RATE_LIMIT_PAUSE = 60  # seconds
MAX_TOKEN_INPUT = 25000

# Ablation study specific settings
TARGET_MTL_COUNT = 250  # Stop after getting 250 MTLs
RANDOM_SEED = 42        # For reproducible random sampling

STRIP_PREFIX = "<RVSPEC_ROOT>/SpecSynthesis/doc/"
INPUT_JSONL = Path("../split_sections/split_sections.jsonl")
PROMPT_PATH = Path("prompts/prompt_for_direct_mtl.txt")
OUTPUT_BASE_DIR = Path("direct_mtls_ablation")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

stop_event = threading.Event()
pause_event = threading.Event()
client = OpenAI()

# Global counter for successful MTLs
successful_mtl_count = 0
mtl_count_lock = threading.Lock()

# ========== Helper Functions ==========
def estimate_tokens(text: str, model="gpt-4o"):
    model_tokenizer_map = {
        "gpt-4.1": "gpt-4o",  # gpt-4.1 uses the same tokenizer as gpt-4o
    }
    tokenizer_name = model_tokenizer_map.get(model, model)
    try:
        encoding = tiktoken.encoding_for_model(tokenizer_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def safe_filename(path: str) -> str:
    h = hashlib.md5(path.encode()).hexdigest()[:8]
    name = re.sub(r"[\\/]", "__", path.replace(STRIP_PREFIX, ""))
    return f"{name}_{h}_mtl.json"

def already_processed(path, output_dir):
    filename = safe_filename(path)
    full_path = output_dir / filename
    if full_path.exists():
        print(f"[] Already processed and saved: {path}")
        return True
    return False

def load_and_sample_sections(jsonl_path: Path, target_count: int, random_seed: int):
    """Load all sections and randomly sample target_count entries"""
    print(f" Loading and sampling {target_count} entries with random seed {random_seed}...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Load all sections
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            path = entry["path"]
            groups[path].append(entry)
    
    # Sort sections within each path
    for path in groups:
        groups[path] = sorted(groups[path], key=lambda e: int(e["section_id"].split("__")[-1]))
    
    # Convert to list of (path, sections) tuples
    all_entries = list(groups.items())
    total_available = len(all_entries)
    
    print(f" Total available documents: {total_available}")
    
    if target_count >= total_available:
        print(f"  Target count ({target_count}) >= available documents ({total_available}). Using all documents.")
        sampled_entries = all_entries
    else:
        # Randomly sample target_count entries
        sampled_entries = random.sample(all_entries, target_count)
        print(f" Randomly sampled {len(sampled_entries)} documents")
    
    # Convert back to dictionary format
    return dict(sampled_entries)

def read_prompt_template():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def build_messages(prompt_template, path, section_ids, text):
    user_input = {
        "path": path,
        "section_ids": section_ids,
        "text": text
    }
    return [
        {"role": "system", "content": prompt_template.strip()},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False, indent=2)}
    ]

def call_model_until_success(model_name, messages):
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            if pause_event.is_set():
                print("[] Global pause triggered. Waiting...")
                pause_event.wait()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            err_msg = str(e)
            if "rate limit" in err_msg or "429" in err_msg:
                print(f"[Rate Limit] {err_msg}. Pausing all threads for {RATE_LIMIT_PAUSE}s...")
                pause_event.set()
                time.sleep(RATE_LIMIT_PAUSE)
                pause_event.clear()
            else:
                print(f"[!] Attempt {attempt}: {err_msg}")
                if attempt >= RETRY_LIMIT:
                    return None
                time.sleep(RETRY_DELAY * attempt)

def parse_response(content, path, section_ids, documentation_text):
    """Parse GPT response and check for MTL content"""
    try:
        content = content.strip()
        if content.startswith("```json"):
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        elif content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Top-level response is not a JSON object.")
        if "mtl" not in data or not isinstance(data["mtl"], list):
            raise ValueError("Missing or malformed 'mtl'.")
        data["path"] = path
        data["section_ids"] = section_ids
        data["documentation"] = documentation_text  # Add this line
        return data, len(data["mtl"])  # Return data and MTL count
    except Exception as e:
        return {
            "path": path,
            "section_ids": section_ids,
            "documentation": documentation_text,  # Add this line
            "error": f"Invalid JSON format: {e}",
            "raw_content": content
        }, 0

def check_target_reached():
    """Check if we've reached the target MTL count"""
    with mtl_count_lock:
        return successful_mtl_count >= TARGET_MTL_COUNT

def update_mtl_count(count):
    """Thread-safe update of MTL count"""
    global successful_mtl_count
    with mtl_count_lock:
        successful_mtl_count += count
        return successful_mtl_count

def process_entry(path, sections, prompt_template, model_name):
    """Process a single entry and return result with MTL count"""
    if check_target_reached():
        return None, 0  # Stop processing if target reached
    
    section_ids = []
    texts = []
    total_tokens = 0

    for section in sections:
        estimated = estimate_tokens(section["text"], model=model_name)
        if total_tokens + estimated > MAX_TOKEN_INPUT:
            print(f"[] Token limit exceeded. Truncated input for {path} to fit within {MAX_TOKEN_INPUT} tokens.")
            break
        texts.append(section["text"])
        section_ids.append(section["section_id"])
        total_tokens += estimated

    if not texts:
        return {
            "path": path,
            "section_ids": [s["section_id"] for s in sections],
            "documentation": "",  # Add this line
            "error": f"Skipped: Estimated tokens exceed {MAX_TOKEN_INPUT} limit"
        }, 0

    text = "\n\n".join(texts)
    messages = build_messages(prompt_template, path, section_ids, text)
    content = call_model_until_success(model_name, messages)
    
    if content is None:
        return {
            "path": path,
            "section_ids": section_ids,
            "documentation": text,  # Add this line
            "error": "Failed to get response after retries"
        }, 0
    
    return parse_response(content, path, section_ids, text)  # Pass text as documentation

def save_individual_result(result, output_dir):
    stripped_path = result["path"].replace(STRIP_PREFIX, "")
    filename = safe_filename(stripped_path)
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def write_combined_jsonl_from_dir(output_dir, output_path):
    all_results = []
    for file in output_dir.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Only keep entries that have 'mtl' field or error
                if "mtl" in data or "error" in data:
                    all_results.append(data)
                else:
                    print(f"[!] Skipped {file}: missing 'mtl' key")
        except Exception as e:
            print(f"[!] Failed to load {file}: {e}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(all_results, key=lambda x: x["path"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    return len(all_results)

def signal_handler(sig, frame):
    print("\n Gracefully stopping after current tasks...")
    stop_event.set()

# ========== Main Pipeline with Queue ==========
def worker(model_name, prompt_template, queue, results, output_dir, pbar):
    while not stop_event.is_set() and not check_target_reached():
        try:
            path, sections = queue.get(timeout=1)
        except Empty:
            return
        try:
            result, mtl_count = process_entry(path, sections, prompt_template, model_name)
            if result is None:  # Target reached
                queue.task_done()
                pbar.update(1)
                continue
                
            results.append(result)
            save_individual_result(result, output_dir)
            
            # Update global MTL count
            current_total = update_mtl_count(mtl_count)
            
            if mtl_count > 0:
                print(f"[] Generated {mtl_count} MTLs from {path}. Total: {current_total}/{TARGET_MTL_COUNT}")
            
            # Check if we've reached the target
            if current_total >= TARGET_MTL_COUNT:
                print(f" Target reached! Generated {current_total} MTLs.")
                stop_event.set()
                
        except Exception as e:
            print(f"[!] Failed on {path}: {e}")
        finally:
            queue.task_done()
            pbar.update(1)

def run_ablation_study():
    global successful_mtl_count
    
    signal.signal(signal.SIGINT, signal_handler)
    prompt_template = read_prompt_template()
    
    # Sample entries randomly
    grouped = load_and_sample_sections(INPUT_JSONL, TARGET_MTL_COUNT * 2, RANDOM_SEED)  # Sample more than needed as buffer
    
    model = MODELS[0]  # Only using gpt-4o
    output_dir = OUTPUT_BASE_DIR / model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create queue with sampled entries
    queue = Queue()
    for path, sections in grouped.items():
        if not already_processed(path, output_dir):
            queue.put((path, sections))
    
    total_to_process = queue.qsize()
    print(f" Starting ablation study with {model}")
    print(f" Sampled entries to process: {total_to_process}")
    print(f" Target MTL count: {TARGET_MTL_COUNT}")
    print(f" Random seed: {RANDOM_SEED}")

    # Start worker threads
    threads = []
    results = []
    pbar = tqdm(total=total_to_process, desc=f" Generating MTLs ({model})", dynamic_ncols=True)

    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker, args=(model, prompt_template, queue, results, output_dir, pbar))
        t.start()
        threads.append(t)

    # Monitor progress
    while any(t.is_alive() for t in threads) and not check_target_reached():
        time.sleep(1)
        # Update progress bar description with current MTL count
        with mtl_count_lock:
            pbar.set_description(f" MTLs: {successful_mtl_count}/{TARGET_MTL_COUNT}")

    # Wait for all threads to complete
    for t in threads:
        t.join()
    pbar.close()

    # Generate final report
    combined_jsonl = output_dir / f"combined_{model}_ablation.jsonl"
    total_saved = write_combined_jsonl_from_dir(output_dir, combined_jsonl)
    
    print(f"\n Ablation Study Completed!")
    print(f" Total MTLs generated: {successful_mtl_count}")
    print(f" Files processed: {len(results)}")
    print(f" Results saved to: {combined_jsonl.resolve()}")
    print(f" Random seed used: {RANDOM_SEED}")
    
    # Save metadata about the ablation study
    metadata = {
        "target_mtl_count": TARGET_MTL_COUNT,
        "actual_mtl_count": successful_mtl_count,
        "random_seed": RANDOM_SEED,
        "model_used": model,
        "files_processed": len(results),
        "total_sampled": len(grouped),
        "completion_status": "complete" if successful_mtl_count >= TARGET_MTL_COUNT else "partial"
    }
    
    metadata_file = output_dir / "ablation_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f" Metadata saved to: {metadata_file.resolve()}")

def main():
    run_ablation_study()

if __name__ == "__main__":
    main()