
import os
import re
import json
import time
import signal
import hashlib
import threading
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from tqdm import tqdm
from datetime import datetime

# Import both API clients
try:
    from openai import OpenAI, OpenAIError
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai tiktoken")

try:
    import anthropic
    from anthropic import Anthropic, APIError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic not available. Install with: pip install anthropic")


# ========== Environment Setup ==========
# Set your API keys here
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY_HERE"




# Check for required API keys based on selected models
def check_api_keys(models):
    openai_models = [m for m in models if m.startswith(("gpt-", "o1-"))]
    claude_models = [m for m in models if m.startswith("claude-")]
    
    if openai_models and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set but OpenAI models are selected.")
    if claude_models and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set but Claude models are selected.")
    
    if openai_models and not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not available. Install with: pip install openai tiktoken")
    if claude_models and not ANTHROPIC_AVAILABLE:
        raise ImportError("Anthropic library not available. Install with: pip install anthropic")

    
# ========== Configuration ==========
# Mix and match GPT and Claude models as needed
MODELS = [
    "claude-sonnet-4-20250514",
    # "claude-opus-4-20250514",
    "gpt-4o",
    # "gpt-4o-mini",
    # "o1-preview",
    # "o1-mini"
    "gpt-4.1"
]

MAX_WORKERS = max(1, os.cpu_count() // 2)
RETRY_LIMIT = 3
RETRY_DELAY = 2
RATE_LIMIT_PAUSE = 60  # seconds

# Model-specific token limits
MODEL_TOKEN_LIMITS = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    # Claude models
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-opus-20240229": 200000,
}

### cz42: Adjust paths for openpilot
STRIP_PREFIX = "<RVSPEC_ROOT>/SpecSynthesis/doc/openpilot_doc"
INPUT_JSONL = Path("split_sections/split_sections-openpilot.jsonl")
PROMPT_PATH = Path("prompts/prompt_for_logic_policies.txt")
OUTPUT_BASE_DIR = Path("logic_policies-openpilot")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# ========== Logging Setup ==========
def setup_logging():
    """Setup logging for long-running processes"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"processing_openpilot__{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting processing job. Log file: {log_file}")
    return logger

logger = None

stop_event = threading.Event()
pause_event = threading.Event()

# Initialize clients
openai_client = None
anthropic_client = None

def initialize_clients():
    global openai_client, anthropic_client
    
    if os.environ.get("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        openai_client = OpenAI()
    if os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
        anthropic_client = Anthropic()

# ========== Helper Functions ==========
def is_openai_model(model_name):
    return model_name.startswith(("gpt-", "o1-"))

def is_claude_model(model_name):
    return model_name.startswith("claude-")

def estimate_tokens(text: str, model="gpt-4o"):
    """Estimate tokens for both OpenAI and Claude models"""
    if is_openai_model(model):
        if not OPENAI_AVAILABLE:
            return len(text) // 4  # Fallback estimation
        model_tokenizer_map = {
            "gpt-4.1": "gpt-4o",
            "o1-preview": "gpt-4o",
            "o1-mini": "gpt-4o",
        }
        tokenizer_name = model_tokenizer_map.get(model, model)
        try:
            encoding = tiktoken.encoding_for_model(tokenizer_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    elif is_claude_model(model):
        # Claude token estimation (approximate)
        return len(text) // 4
    else:
        # Fallback estimation
        return len(text) // 4

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

def build_messages(prompt_template, path, section_ids, text, model_name):
    """Build messages for both OpenAI and Claude APIs"""
    user_input = {
        "path": path,
        "section_ids": section_ids,
        "text": text
    }
    
    if is_openai_model(model_name):
        # OpenAI format
        return [
            {"role": "system", "content": prompt_template.strip()},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False, indent=2)}
        ]
    elif is_claude_model(model_name):
        # Claude format - combine system and user content
        full_prompt = f"{prompt_template.strip()}\n\n{json.dumps(user_input, ensure_ascii=False, indent=2)}"
        return [
            {"role": "user", "content": full_prompt}
        ]
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def call_openai_model(model_name, messages):
    """Call OpenAI API"""
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def call_claude_model(model_name, messages):
    """Call Claude API"""
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0,
        messages=messages
    )
    return response.content[0].text

def call_model_until_success(model_name, messages):
    """Unified function to call either OpenAI or Claude models"""
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            if pause_event.is_set():
                print("[] Global pause triggered. Waiting...")
                pause_event.wait()
            
            if is_openai_model(model_name):
                return call_openai_model(model_name, messages)
            elif is_claude_model(model_name):
                return call_claude_model(model_name, messages)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
                
        except Exception as e:
            err_msg = str(e)
            
            # Handle rate limiting for both APIs
            if any(phrase in err_msg.lower() for phrase in ["rate limit", "429", "too many requests"]):
                print(f"[Rate Limit] {err_msg}. Pausing all threads for {RATE_LIMIT_PAUSE}s...")
                pause_event.set()
                time.sleep(RATE_LIMIT_PAUSE)
                pause_event.clear()
            else:
                print(f"[!] Attempt {attempt}: {err_msg}")
                if attempt >= RETRY_LIMIT:
                    raise
                time.sleep(RETRY_DELAY * attempt)

def parse_model_response(content, path, section_ids):
    """Parse response from either OpenAI or Claude"""
    try:
        content = content.strip()
        # Handle code blocks
        if content.startswith("```json"):
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        elif content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        
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
    max_tokens = MODEL_TOKEN_LIMITS.get(model_name, 25000)

    for section in sections:
        estimated = estimate_tokens(section["text"], model=model_name)
        if total_tokens + estimated > max_tokens:
            print(f"[] Token limit exceeded. Truncated input for {path} to fit within {max_tokens} tokens.")
            break
        texts.append(section["text"])
        section_ids.append(section["section_id"])
        total_tokens += estimated

    if not texts:
        return {
            "path": path,
            "section_ids": [s["section_id"] for s in sections],
            "error": f"Skipped: Estimated tokens exceed {max_tokens} limit"
        }

    text = "\n\n".join(texts)
    messages = build_messages(prompt_template, path, section_ids, text, model_name)
    content = call_model_until_success(model_name, messages)
    return parse_model_response(content, path, section_ids)

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
    logger.info(" Gracefully stopping after current tasks...")
    print("\n Gracefully stopping after current tasks...")
    stop_event.set()

def save_progress(processed_count, total_count, model_name):
    """Save progress to a file for monitoring"""
    progress_file = OUTPUT_BASE_DIR / f"progress_{model_name}.json"
    progress_data = {
        "processed": processed_count,
        "total": total_count,
        "percentage": (processed_count / total_count * 100) if total_count > 0 else 0,
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

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
    global logger
    logger = setup_logging()
    
    # Check API keys and initialize clients
    check_api_keys(MODELS)
    initialize_clients()
    
    signal.signal(signal.SIGINT, signal_handler)
    prompt_template = read_prompt_template()
    grouped = group_sections_by_path(INPUT_JSONL)

    logger.info(f"Starting processing with models: {MODELS}")
    logger.info(f"Total files to process: {len(grouped)}")

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
            current = len(results)
            pbar[model].update(current - prev)
            
            # Save progress periodically
            if current > 0 and current % 10 == 0:
                save_progress(current, pbar[model].total, model)
        time.sleep(1)

    for t, model, results, output_dir in threads:
        t.join()
        pbar[model].close()
        combined_jsonl = output_dir / f"combined_{model}.jsonl"
        write_combined_jsonl_from_dir(output_dir, combined_jsonl)
        logger.info(f" Saved combined results to: {combined_jsonl.resolve()}")
        logger.info(f" Total completed for {model}: {len(results)}")
        print(f" Saved combined results to: {combined_jsonl.resolve()}")
        print(f" Total completed: {len(results)}")
        
        # Final progress save
        save_progress(len(results), pbar[model].total, model)

    logger.info("Processing completed successfully!")

def main():
    run_all_models()

if __name__ == "__main__":
    main()