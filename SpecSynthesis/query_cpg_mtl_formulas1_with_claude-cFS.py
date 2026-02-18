import os
import re
import json
import time
import signal
import hashlib
import threading
import logging
from pathlib import Path
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



def check_api_keys(models):
    """Check for required API keys based on selected models"""
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
    "gpt-4.1"
]

MAX_WORKERS = max(1, os.cpu_count() // 2)
RETRY_LIMIT = 3
RETRY_DELAY = 2
RATE_LIMIT_PAUSE = 60

# Model-specific token limits
MODEL_TOKEN_LIMITS = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-4.1": 128000,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    # Claude models
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-opus-20240229": 200000,
}

INPUT_DIR = Path("./logic_policies-cFS")
OUTPUT_BASE_DIR = Path("mtl") / Path("cpg_mtl-cFS")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

stop_event = threading.Event()
pause_event = threading.Event()

# Initialize clients
openai_client = None
anthropic_client = None

PROMPT_PATH = Path("prompts/prompt_for_CPG_MTL.txt")

# ========== Logging Setup ==========
def setup_logging():
    """Setup logging for long-running processes"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mtl_generation_cFS_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MTL generation job. Log file: {log_file}")
    return logger

logger = None

def initialize_clients():
    """Initialize API clients based on available keys"""
    global openai_client, anthropic_client
    
    if os.environ.get("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        openai_client = OpenAI()
    if os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
        anthropic_client = Anthropic()

# ========== Helper Functions ==========
def is_openai_model(model_name):
    """Check if the model is an OpenAI model"""
    return model_name.startswith(("gpt-", "o1-"))

def is_claude_model(model_name):
    """Check if the model is a Claude model"""
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

def read_prompt_template():
    """Read the MTL generation prompt template"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def safe_filename(path: str, policy_id: int) -> str:
    """Generate safe filename for output files"""
    h = hashlib.md5(f"{path}::{policy_id}".encode()).hexdigest()[:8]
    name = re.sub(r"[\\/]+", "__", path.strip("/"))
    return f"{name}_p{policy_id}_{h}.json"

def already_processed(path: str, policy_id: int, output_dir: Path) -> bool:
    """Check if this policy has already been processed"""
    filename = safe_filename(path, policy_id)
    if (output_dir / filename).exists():
        print(f"[] Already processed: {path} (policy_id={policy_id})")
        return True
    return False

def build_messages(prompt_template, logic_obj, model_name):
    """Build messages for both OpenAI and Claude APIs"""
    if is_openai_model(model_name):
        # OpenAI format
        return [
            {"role": "system", "content": prompt_template.strip()},
            {"role": "user", "content": json.dumps(logic_obj, ensure_ascii=False, indent=2)}
        ]
    elif is_claude_model(model_name):
        # Claude format - combine system and user content
        full_prompt = f"{prompt_template.strip()}\n\n{json.dumps(logic_obj, ensure_ascii=False, indent=2)}"
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
    return response.choices[0].message.content.strip()

def call_claude_model(model_name, messages):
    """Call Claude API"""
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0,
        messages=messages
    )
    return response.content[0].text.strip()

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
                    logger.error(f"Failed after {RETRY_LIMIT} attempts: {err_msg}")
                    return "[Error: Failed after multiple attempts]"
                time.sleep(RETRY_DELAY * attempt)

def worker(model_name, prompt_template, queue, output_dir, progress_bar):
    """Worker thread function for processing logic policies"""
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

            # Create a copy to avoid modifying the original
            logic_obj_cp = dict(logic_obj)
            
            # Build messages for the specific model type
            messages = build_messages(prompt_template, logic_obj, model_name)
            
            # Check token limits
            max_tokens = MODEL_TOKEN_LIMITS.get(model_name, 25000)
            token_est = estimate_tokens(messages[-1]['content'], model=model_name)
            
            if token_est > max_tokens:
                logic_obj_cp["error"] = f"Input too long: {token_est} tokens (max: {max_tokens})"
                print(f"[!] Skipped due to token size: {path} p{policy_id}")
                logger.warning(f"Skipped {path} p{policy_id}: {token_est} tokens > {max_tokens}")
            else:
                # Generate MTL formula
                mtl_output = call_model_until_success(model_name, messages)
                logic_obj_cp["mtl"] = mtl_output if mtl_output else "[Error: No response]"
                logic_obj_cp["model_used"] = model_name
                logic_obj_cp["tokens_estimated"] = token_est
                
            # Save result
            fname = safe_filename(path, policy_id)
            with open(output_dir / fname, "w", encoding="utf-8") as f:
                json.dump(logic_obj_cp, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"[!] Worker exception: {e}")
            logger.error(f"Worker exception for {path} p{policy_id}: {e}")
        finally:
            queue.task_done()
            progress_bar.update(1)

def load_logic_policies(input_path):
    """Load logic policies from JSONL file"""
    policies = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                policy = json.loads(line)
                policies.append(policy)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on line {line_num} in {input_path}: {e}")
                print(f"[!] Skipping malformed JSON on line {line_num}")
    return policies

def save_progress(processed_count, total_count, model_name):
    """Save progress to a file for monitoring"""
    progress_file = OUTPUT_BASE_DIR / f"mtl_progress_{model_name}.json"
    progress_data = {
        "processed": processed_count,
        "total": total_count,
        "percentage": (processed_count / total_count * 100) if total_count > 0 else 0,
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    logger.info(" Gracefully stopping after current tasks...")
    print("\n Gracefully stopping after current tasks...")
    stop_event.set()

def run_model(model_name):
    """Process logic policies for a specific model"""
    input_file = INPUT_DIR / f"extracted_logic_policies_{model_name}.jsonl"
    output_dir = OUTPUT_BASE_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"[!] Skipping {model_name}: {input_file} does not exist.")
        logger.warning(f"Input file not found: {input_file}")
        return

    # Load all policies
    all_policies = load_logic_policies(input_file)
    if not all_policies:
        print(f"[!] No valid policies found in {input_file}")
        return

    # Create queue
    queue = Queue()
    for logic_obj in all_policies:
        queue.put(logic_obj)

    print(f" Generating MTL for {model_name} - Total: {queue.qsize()}")
    logger.info(f"Starting MTL generation for {model_name}: {queue.qsize()} policies")
    
    prompt_template = read_prompt_template()

    # Start worker threads
    threads = []
    pbar = tqdm(total=queue.qsize(), desc=f" {model_name} MTL", dynamic_ncols=True)

    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker, args=(model_name, prompt_template, queue, output_dir, pbar))
        t.start()
        threads.append(t)

    # Monitor progress and save periodically
    initial_size = queue.qsize()
    while any(t.is_alive() for t in threads):
        processed = initial_size - queue.qsize()
        if processed > 0 and processed % 10 == 0:
            save_progress(processed, initial_size, model_name)
        time.sleep(1)

    # Wait for all threads to complete
    for t in threads:
        t.join()
    pbar.close()

    # Final progress save
    save_progress(initial_size, initial_size, model_name)
    
    print(f" Saved to: {output_dir.resolve()}")
    logger.info(f"Completed MTL generation for {model_name}")

def main():
    """Main function"""
    global logger
    logger = setup_logging()
    
    # Check API keys and initialize clients
    check_api_keys(MODELS)
    initialize_clients()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info(f"Starting MTL generation with models: {MODELS}")
    
    for model in MODELS:
        if not stop_event.is_set():
            run_model(model)
    
    logger.info("MTL generation completed successfully!")

if __name__ == "__main__":
    main()