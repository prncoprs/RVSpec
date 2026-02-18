import os
import re
import json
import time
import signal
import hashlib
import threading
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
MODELS = ["gpt-4.1"]
MAX_WORKERS = max(1, os.cpu_count() // 2)
RETRY_LIMIT = 3
RETRY_DELAY = 2
RATE_LIMIT_PAUSE = 60  # seconds
MAX_TOKEN_INPUT = 25000

STRIP_PREFIX = "<RVSPEC_ROOT>/SpecSynthesis/doc/"
INPUT_JSONL = Path("split_sections/split_sections.jsonl")
PROMPT_PATH = Path("prompts/prompt_for_logic_policies.txt")
OUTPUT_BASE_DIR = Path("logic_policies")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

stop_event = threading.Event()
pause_event = threading.Event()
client = OpenAI()

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
    return f"{name}_{h}.json"

def already_processed(path, output_dir):
    filename = safe_filename(path)
    full_path = output_dir / filename
    if full_path.exists():
        print(f"[] Already processed and saved: {path}")
        return True
    return False

def group_sections_by_path(jsonl_path: Path):
    groups = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            path = entry["path"]
            groups[path].append(entry)
    for path in groups:
        groups[path] = sorted(groups[path], key=lambda e: int(e["section_id"].split("__")[-1]))
    return dict(sorted(groups.items()))

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
                time.sleep(RETRY_DELAY * attempt)

def parse_chatgpt_response(content, path, section_ids):
    try:
        content = content.strip()
        if content.startswith("```json"):
            content = re.sub(r"^```json\\s*", "", content)
            content = re.sub(r"\\s*```$", "", content)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Top-level response is not a JSON object.")
        if "logic_policies" not in data or not isinstance(data["logic_policies"], list):
            raise ValueError("Missing or malformed 'logic_policies'.")
        data["path"] = path
        data["section_ids"] = section_ids
        return data
    except Exception as e:
        return {
            "path": path,
            "section_ids": section_ids,
            "error": f"Invalid JSON format: {e}",
            "raw_content": content
        }

def process_entry(path, sections, prompt_template, model_name):
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
            "error": f"Skipped: Estimated tokens exceed {MAX_TOKEN_INPUT} limit"
        }

    text = "\n\n".join(texts)
    messages = build_messages(prompt_template, path, section_ids, text)
    content = call_model_until_success(model_name, messages)
    return parse_chatgpt_response(content, path, section_ids)

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
                all_results.append(json.load(f))
        except Exception as e:
            print(f"[!] Failed to load {file}: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(all_results, key=lambda x: x["path"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def signal_handler(sig, frame):
    print("\n Gracefully stopping after current tasks...")
    stop_event.set()

# ========== Main Pipeline with Queue ==========
def worker(model_name, prompt_template, queue, results, output_dir):
    while not stop_event.is_set():
        try:
            path, sections = queue.get(timeout=1)
        except Empty:
            return
        try:
            result = process_entry(path, sections, prompt_template, model_name)
            results.append(result)
            save_individual_result(result, output_dir)
        except Exception as e:
            print(f"[!] Failed on {path}: {e}")
        finally:
            queue.task_done()

def run_all_models():
    signal.signal(signal.SIGINT, signal_handler)
    prompt_template = read_prompt_template()
    grouped = group_sections_by_path(INPUT_JSONL)

    queues = {model: Queue() for model in MODELS}
    output_dirs = {}
    for model in MODELS:
        output_dir = OUTPUT_BASE_DIR / model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[model] = output_dir
        for path, sections in grouped.items():
            if not already_processed(path, output_dir):
                queues[model].put((path, sections))

    threads = []
    for model in MODELS:
        results = []
        output_dir = output_dirs[model]
        for _ in range(MAX_WORKERS):
            t = threading.Thread(target=worker, args=(model, prompt_template, queues[model], results, output_dir))
            t.start()
            threads.append((t, model, results, output_dir))

    pbar = {model: tqdm(total=queues[model].qsize(), desc=f" Processing with {model}", dynamic_ncols=True) for model in MODELS}

    while any(t[0].is_alive() for t in threads):
        for _, model, results, _ in threads:
            prev = pbar[model].n
            pbar[model].update(len(results) - prev)
        time.sleep(1)

    for t, model, results, output_dir in threads:
        t.join()
        combined_jsonl = output_dir / f"combined_{model}.jsonl"
        write_combined_jsonl_from_dir(output_dir, combined_jsonl)
        print(f" Saved combined results to: {combined_jsonl.resolve()}")
        print(f" Total completed: {len(results)}")

def main():
    run_all_models()

if __name__ == "__main__":
    main()
