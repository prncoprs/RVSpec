
import os
import re
import json
import time
import signal
import hashlib
import threading
from pathlib import Path
from queue import Queue, Empty
from openai import OpenAI, OpenAIError
from tqdm import tqdm
import tiktoken

# ========== Environment Setup ==========
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
# Ensure the OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the script.")




# ========== Configuration ==========
MODELS = ["gpt-4o", "gpt-4.1"]
MAX_WORKERS = max(1, os.cpu_count() // 2)
RETRY_LIMIT = 3
RETRY_DELAY = 2
RATE_LIMIT_PAUSE = 60
MAX_TOKENS_INPUT = 12000

INPUT_DIR = Path("./logic_policies")
OUTPUT_BASE_DIR = Path("mtl") / Path("cpg_mtl")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

stop_event = threading.Event()
pause_event = threading.Event()
client = OpenAI()

PROMPT_PATH = Path("prompts/prompt_for_CPG_MTL.txt")

# ========== Helper Functions ==========
def estimate_tokens(text: str, model="gpt-4o"):
    model_tokenizer_map = {"gpt-4.1": "gpt-4o"}
    tokenizer_name = model_tokenizer_map.get(model, model)
    encoding = tiktoken.encoding_for_model(tokenizer_name)
    return len(encoding.encode(text))

def read_prompt_template():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def safe_filename(path: str, policy_id: int) -> str:
    h = hashlib.md5(f"{path}::{policy_id}".encode()).hexdigest()[:8]
    name = re.sub(r"[\\/]+", "__", path.strip("/"))
    return f"{name}_p{policy_id}_{h}.json"

def already_processed(path: str, policy_id: int, output_dir: Path) -> bool:
    filename = safe_filename(path, policy_id)
    if (output_dir / filename).exists():
        print(f"[] Already processed: {path} (policy_id={policy_id})")
        return True
    return False

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
            return response.choices[0].message.content.strip()
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

def build_messages(prompt_template, logic_obj):
    return [
        {"role": "system", "content": prompt_template.strip()},
        {"role": "user", "content": json.dumps(logic_obj, ensure_ascii=False, indent=2)}
    ]

def worker(model_name, prompt_template, queue, output_dir, progress_bar):
    while not stop_event.is_set():
        try:
            logic_obj = queue.get(timeout=1)
        except Empty:
            return
        try:
            path, policy_id = logic_obj.get("path"), logic_obj.get("policy_id")
            if already_processed(path, policy_id, output_dir):
                progress_bar.update(1)
                continue

            logic_obj_cp = dict(logic_obj)
            messages = build_messages(prompt_template, logic_obj)
            token_est = estimate_tokens(messages[-1]['content'], model=model_name)
            if token_est > MAX_TOKENS_INPUT:
                logic_obj_cp["error"] = f"Input too long: {token_est} tokens"
                print(f"[!] Skipped due to token size: {path} p{policy_id}")
            else:
                mtl_output = call_model_until_success(model_name, messages)
                logic_obj_cp["mtl"] = mtl_output if mtl_output else "[Error: No response]"
            fname = safe_filename(path, policy_id)
            with open(output_dir / fname, "w", encoding="utf-8") as f:
                json.dump(logic_obj_cp, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Worker exception: {e}")
        finally:
            queue.task_done()
            progress_bar.update(1)

def load_logic_policies(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def signal_handler(sig, frame):
    print("\n Gracefully stopping after current tasks...")
    stop_event.set()

def run_model(model_name):
    input_file = INPUT_DIR / f"extracted_logic_policies_{model_name}.jsonl"
    output_dir = OUTPUT_BASE_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_policies = list(load_logic_policies(input_file))
    queue = Queue()
    for logic_obj in all_policies:
        queue.put(logic_obj)

    print(f" Generating MTL for {model_name} - Total: {queue.qsize()}")
    prompt_template = read_prompt_template()

    threads = []
    pbar = tqdm(total=queue.qsize(), desc=f" {model_name} MTL", dynamic_ncols=True)

    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker, args=(model_name, prompt_template, queue, output_dir, pbar))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    pbar.close()

    print(f" Saved to: {output_dir.resolve()}")

def main():
    signal.signal(signal.SIGINT, signal_handler)
    for model in MODELS:
        run_model(model)

if __name__ == "__main__":
    main()
