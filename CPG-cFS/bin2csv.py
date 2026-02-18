#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch convert COSMOS generated *_tlm.bin files to CSV using TlmExtractor in container,
outputting to specified host directory.

Note:
- Docker container must have access to the project directory (current container is mounted this way).
- TlmExtractor config file (e.g., nos3_all_autogen.txt) should be placed in
  <COSMOS_ROOT>/config/tools/tlm_extractor/.
- We switch to COSMOS_SYSTEM_DIR in container before running (to avoid path auto-concatenation issues).
- Input files are copied from host to container, processed, then output copied back to host.
"""

from pathlib import Path
import subprocess
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import signal
import atexit
import os
import psutil

# ===================== Configurable Variables =====================

# Input bin directory (host absolute path; files will be copied to container for processing)
INPUT_LOGS_DIR = Path("<DATA_DIR>")

# Final CSV output directory (host absolute path)
HOST_OUTPUT_DIR = Path("<DATA_DIR>")

# COSMOS project root directory (host and container paths are consistent)
COSMOS_ROOT = Path("<RVSPEC_ROOT>/UAV/nos3/gsw/cosmos")

# COSMOS system working directory (must switch to this in container and use --system system.txt)
COSMOS_SYSTEM_DIR = COSMOS_ROOT / "config/system"

# TlmExtractor config filename (placed in config/tools/tlm_extractor directory; filename only, no path)
EXTRACTOR_CONFIG_BASENAME = "nos3_all_autogen.txt"

# Container name (name seen in docker ps)
CONTAINER_NAME = "cosmos-openc3-operator-1"

# Absolute path of Ruby TlmExtractor in container
TLM_EXTRACTOR_PATH = COSMOS_ROOT / "tools" / "TlmExtractor"

# Temporary directories in container
TEMP_INPUT_DIR = COSMOS_ROOT / "outputs" / "temp_input"  # For copying bin files
TEMP_EXPORT_DIR = COSMOS_ROOT / "outputs" / "exports"    # For CSV output

# Only process files matching this pattern
BIN_GLOB = "*_tlm.bin"

# Whether to overwrite existing target CSV (True=overwrite, False=skip)
OVERWRITE = True

# Number of worker threads for parallel processing
MAX_WORKERS = 10

# ===================== Logging Configuration =====================

logger = logging.getLogger("tlm_to_csv")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ===================== Global State Management =====================

class ProcessManager:
    """Manages all processes and threads for proper cleanup"""
    
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.active_processes = set()
        self.process_lock = threading.Lock()
        self.executor = None
        self.futures = []
        self.main_thread = threading.current_thread()
        self.cleanup_completed = False
        
    def register_process(self, process):
        """Register a subprocess for tracking"""
        with self.process_lock:
            self.active_processes.add(process)
    
    def unregister_process(self, process):
        """Unregister a subprocess"""
        with self.process_lock:
            self.active_processes.discard(process)
    
    def register_executor(self, executor):
        """Register the thread pool executor"""
        self.executor = executor
    
    def register_future(self, future):
        """Register a future task"""
        self.futures.append(future)
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        logger.info("Shutdown requested...")
        self.shutdown_event.set()
    
    def is_shutdown_requested(self):
        """Check if shutdown was requested"""
        return self.shutdown_event.is_set()
    
    def cleanup_all(self):
        """Clean up all resources"""
        if self.cleanup_completed:
            return
            
        logger.info("Starting comprehensive cleanup...")
        
        # Step 1: Signal shutdown
        self.shutdown_event.set()
        
        # Step 2: Cancel all futures
        if self.futures:
            logger.info("Cancelling %d pending tasks...", len(self.futures))
            for future in self.futures:
                try:
                    if not future.done():
                        future.cancel()
                except Exception as e:
                    logger.debug("Error cancelling future: %s", e)
        
        # Step 3: Shutdown executor
        if self.executor:
            logger.info("Shutting down thread pool executor...")
            try:
                self.executor.shutdown(wait=False)
                # Give it a moment to start shutdown
                time.sleep(0.5)
                # Force shutdown if needed
                if hasattr(self.executor, '_threads'):
                    for thread in self.executor._threads:
                        if thread.is_alive():
                            logger.debug("Waiting for thread %s to finish...", thread.name)
            except Exception as e:
                logger.warning("Error shutting down executor: %s", e)
        
        # Step 4: Terminate all docker processes
        self.terminate_all_processes()
        
        # Step 5: Kill any remaining child processes
        self.kill_child_processes()
        
        self.cleanup_completed = True
        logger.info("Cleanup completed.")
    
    def terminate_all_processes(self):
        """Terminate all tracked docker processes"""
        with self.process_lock:
            if not self.active_processes:
                return
                
            logger.info("Terminating %d active docker processes...", len(self.active_processes))
            for process in list(self.active_processes):
                try:
                    if process.poll() is None:  # Process is still running
                        # Try graceful termination first
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                            logger.debug("Process terminated gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if it doesn't terminate gracefully
                            logger.debug("Force killing process...")
                            process.kill()
                            try:
                                process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                # If even kill doesn't work, try OS-level kill
                                try:
                                    os.kill(process.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass  # Process already dead
                except Exception as e:
                    logger.debug("Error terminating process: %s", e)
            
            self.active_processes.clear()
    
    def kill_child_processes(self):
        """Kill any remaining child processes using psutil"""
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            
            if children:
                logger.info("Found %d child processes to terminate...", len(children))
                
                # Send SIGTERM to all children
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                # Wait a moment for graceful termination
                gone, alive = psutil.wait_procs(children, timeout=3)
                
                # Force kill any remaining processes
                if alive:
                    logger.info("Force killing %d remaining processes...", len(alive))
                    for child in alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
        except Exception as e:
            logger.debug("Error killing child processes: %s", e)

# Create global process manager
process_manager = ProcessManager()

# ===================== Signal Handlers =====================

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals"""
    signal_name = signal.Signals(signum).name
    logger.warning("Received signal %s (%d). Initiating shutdown...", signal_name, signum)
    
    # Request shutdown
    process_manager.request_shutdown()
    
    # If we're in the main thread, start cleanup
    if threading.current_thread() == process_manager.main_thread:
        process_manager.cleanup_all()
        cleanup_partial_files()
        
        # Exit immediately after cleanup
        logger.info("Exiting...")
        sys.exit(130 if signum == signal.SIGINT else 1)

def cleanup_on_exit():
    """Cleanup function registered with atexit"""
    if not process_manager.cleanup_completed:
        logger.info("Running exit cleanup...")
        process_manager.cleanup_all()
        cleanup_partial_files()

# ===================== Helper Functions =====================

def cleanup_partial_files():
    """Clean up any partial CSV files that may have been created"""
    try:
        if process_manager.is_shutdown_requested():
            logger.info("Checking for partial files in target directory...")
            partial_files = []
            for pattern in ["*.csv.tmp", "*.csv.partial", "*.csv.downloading"]:
                partial_files.extend(HOST_OUTPUT_DIR.glob(pattern))
            
            if partial_files:
                logger.info("Cleaning up %d partial files...", len(partial_files))
                for partial_file in partial_files:
                    try:
                        partial_file.unlink()
                        logger.debug("Removed partial file: %s", partial_file)
                    except Exception as e:
                        logger.debug("Failed to remove partial file %s: %s", partial_file, e)
            
            # Also clean up temp files in container
            try:
                cleanup_cmd = f"rm -rf {sh_quote(str(TEMP_INPUT_DIR))}/* {sh_quote(str(TEMP_EXPORT_DIR))}/*"
                run_in_container(cleanup_cmd, ignore_shutdown=True)
                logger.info("Cleaned up container temp files")
            except Exception:
                pass  # Ignore errors during cleanup
                
    except Exception as e:
        logger.debug("Error during file cleanup: %s", e)

def check_prereqs():
    """Check prerequisites and provide clear error messages"""
    for p in [INPUT_LOGS_DIR, COSMOS_ROOT, COSMOS_SYSTEM_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
    
    # Config file must exist in config/tools/tlm_extractor
    cfg_path = COSMOS_ROOT / "config" / "tools" / "tlm_extractor" / EXTRACTOR_CONFIG_BASENAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find extraction config file: {cfg_path}")
    
    # Check if config file has syntax errors by reading a few lines
    try:
        with open(cfg_path, 'r') as f:
            lines = f.readlines()[:10]  # Read first 10 lines to check for obvious issues
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('ITEM') and len(line.split()) < 3:
                    logger.warning("Possible config file issue at line %d: %s", i, line)
                    logger.warning("TlmExtractor config may have syntax errors")
    except Exception as e:
        logger.warning("Could not validate config file: %s", e)
    
    # Create temporary input directory in container
    try:
        create_temp_dir_cmd = f"mkdir -p {sh_quote(str(TEMP_INPUT_DIR))} {sh_quote(str(TEMP_EXPORT_DIR))}"
        mkdir_result = run_in_container(create_temp_dir_cmd)
        if mkdir_result.returncode != 0:
            logger.warning("Could not create temp directories in container")
    except Exception as e:
        logger.warning("Error creating temp directories: %s", e)
    
    # Host target directory
    HOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check write permissions for target directory
    test_file = HOST_OUTPUT_DIR / "test_write_permission.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
        logger.info("Write permission confirmed for target directory")
    except Exception as e:
        logger.warning("May have write permission issues with target directory: %s", e)

def run_in_container(cmd: str, ignore_shutdown: bool = False) -> subprocess.CompletedProcess:
    """
    Execute a bash -lc command in container, return CompletedProcess.
    """
    if not ignore_shutdown and process_manager.is_shutdown_requested():
        raise InterruptedError("Shutdown requested")
    
    full_cmd = [
        "docker", "exec", "-i", "--user", "root", CONTAINER_NAME,
        "bash", "-lc", cmd
    ]
    logger.debug("Docker exec command: %s", " ".join(full_cmd))
    
    # Start the process
    process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Track active process
    process_manager.register_process(process)
    
    try:
        # Wait for completion or shutdown signal
        while process.poll() is None:
            if not ignore_shutdown and process_manager.is_shutdown_requested():
                logger.debug("Terminating docker process due to shutdown request")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
                raise InterruptedError("Process terminated due to shutdown")
            time.sleep(0.05)  # Small delay to avoid busy waiting
        
        # Get the result
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(full_cmd, process.returncode, stdout, stderr)
    
    finally:
        # Remove from active processes
        process_manager.unregister_process(process)

def convert_one_bin(bin_path: Path, pbar: tqdm = None) -> tuple[bool, str]:
    """
    Convert one bin file to CSV using TlmExtractor in container.
    Returns (success: bool, message: str).
    """
    try:
        # Check for shutdown signal at the start
        if process_manager.is_shutdown_requested():
            return False, f"Conversion cancelled due to shutdown: {bin_path.name}"
        
        target_csv = HOST_OUTPUT_DIR / f"{bin_path.stem}.csv"
        if target_csv.exists() and not OVERWRITE:
            msg = f"Already exists and not overwriting, skipping: {target_csv.name}"
            logger.info(msg)
            if pbar:
                pbar.set_postfix_str(f"Skipped: {bin_path.name}")
                pbar.update(1)
            return True, msg

        # Define paths for container processing
        container_input_bin = TEMP_INPUT_DIR / bin_path.name
        temp_csv = TEMP_EXPORT_DIR / f"{bin_path.stem}.csv"

        # Check for shutdown signal before starting
        if process_manager.is_shutdown_requested():
            return False, f"Conversion cancelled due to shutdown: {bin_path.name}"

        logger.info("Starting conversion: %s", bin_path.name)
        if pbar:
            pbar.set_postfix_str(f"Copying: {bin_path.name}")

        # Step 1: Copy bin file from host to container
        try:
            docker_cp_input_cmd = [
                "docker", "cp", 
                str(bin_path),
                f"{CONTAINER_NAME}:{container_input_bin}"
            ]
            
            logger.debug("Copying input to container: %s", " ".join(docker_cp_input_cmd))
            
            # Use subprocess with proper tracking
            cp_process = subprocess.Popen(docker_cp_input_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process_manager.register_process(cp_process)
            
            try:
                stdout, stderr = cp_process.communicate(timeout=30)
                if cp_process.returncode != 0:
                    error_msg = f"Failed to copy input file to container: {stderr}"
                    logger.error(error_msg)
                    if pbar:
                        pbar.update(1)
                    return False, error_msg
            finally:
                process_manager.unregister_process(cp_process)
                
            logger.debug("Successfully copied input to container: %s", container_input_bin)
            
        except subprocess.TimeoutExpired:
            cp_process.kill()
            error_msg = f"Timeout copying input file"
            logger.error(error_msg)
            if pbar:
                pbar.update(1)
            return False, error_msg
        except Exception as copy_error:
            error_msg = f"Failed to copy input file: {copy_error}"
            logger.error(error_msg)
            if pbar:
                pbar.update(1)
            return False, error_msg

        if pbar:
            pbar.set_postfix_str(f"Converting: {bin_path.name}")

        # Step 2: Run TlmExtractor in container
        cmd = (
            f"cd {sh_quote(str(COSMOS_SYSTEM_DIR))} && "
            f"ruby {sh_quote(str(TLM_EXTRACTOR_PATH))} "
            f"--system system.txt "
            f"-c {sh_quote(EXTRACTOR_CONFIG_BASENAME)} "
            f"-i {sh_quote(str(container_input_bin))} "
            f"-o {sh_quote(str(temp_csv))}"
        )

        cp = run_in_container(cmd)

        # Check for shutdown signal after conversion
        if process_manager.is_shutdown_requested():
            # Clean up input file in container
            cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
            run_in_container(cleanup_input_cmd, ignore_shutdown=True)
            return False, f"Conversion cancelled due to shutdown: {bin_path.name}"

        if cp.returncode != 0:
            error_msg = f"TlmExtractor failed: {bin_path}"
            logger.error(error_msg)
            logger.error("STDOUT:\n%s", cp.stdout.strip())
            logger.error("STDERR:\n%s", cp.stderr.strip())
            
            # Clean up input file
            cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
            run_in_container(cleanup_input_cmd, ignore_shutdown=True)
            
            if pbar:
                pbar.update(1)
            return False, f"TlmExtractor failed (rc={cp.returncode}): {bin_path}"

        logger.info("Container generated: %s", temp_csv)

        if pbar:
            pbar.set_postfix_str(f"Copying output: {bin_path.name}")

        # Step 3: Set permissions to 666 for the output file
        try:
            chmod_cmd = f"chmod 666 {sh_quote(str(temp_csv))}"
            chmod_result = run_in_container(chmod_cmd)
            if chmod_result.returncode == 0:
                logger.debug("Set permissions 666 for output file")
            else:
                logger.warning("Could not set permissions for output file")
        except Exception as e:
            logger.warning("Error setting permissions: %s", e)

        # Step 4: Copy CSV from container to host target directory
        try:
            # Create a temporary marker file to indicate incomplete transfer
            temp_marker = target_csv.with_suffix('.csv.downloading')
            
            docker_cp_output_cmd = [
                "docker", "cp", 
                f"{CONTAINER_NAME}:{temp_csv}",
                str(temp_marker)
            ]
            
            logger.debug("Copying output from container: %s", " ".join(docker_cp_output_cmd))
            
            cp_process = subprocess.Popen(docker_cp_output_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process_manager.register_process(cp_process)
            
            try:
                stdout, stderr = cp_process.communicate(timeout=30)
                if cp_process.returncode == 0:
                    # Rename from temp to final name
                    temp_marker.rename(target_csv)
                    logger.info("Successfully copied to: %s", target_csv)
                    
                    # Clean up both temp files in container
                    cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
                    cleanup_output_cmd = f"rm -f {sh_quote(str(temp_csv))}"
                    cleanup_combined_cmd = f"{cleanup_input_cmd} && {cleanup_output_cmd}"
                    
                    cleanup_result = run_in_container(cleanup_combined_cmd, ignore_shutdown=True)
                    if cleanup_result.returncode == 0:
                        logger.debug("Cleaned up temp files in container")
                    else:
                        logger.warning("Could not clean up some temp files in container")
                    
                    success_msg = f"Successfully processed: {target_csv}"
                else:
                    error_msg = f"Failed to copy output from container: {stderr}"
                    logger.error(error_msg)
                    
                    # Clean up temp marker if it exists
                    if temp_marker.exists():
                        temp_marker.unlink()
                    
                    # Clean up input file even if output copy failed
                    cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
                    run_in_container(cleanup_input_cmd, ignore_shutdown=True)
                    
                    if pbar:
                        pbar.update(1)
                    return False, error_msg
            finally:
                process_manager.unregister_process(cp_process)
                
        except subprocess.TimeoutExpired:
            cp_process.kill()
            error_msg = f"Timeout copying output file"
            logger.error(error_msg)
            
            # Clean up temp files
            if temp_marker.exists():
                temp_marker.unlink()
            cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
            run_in_container(cleanup_input_cmd, ignore_shutdown=True)
            
            if pbar:
                pbar.update(1)
            return False, error_msg
        except Exception as copy_error:
            error_msg = f"Failed to copy output file via docker cp: {copy_error}"
            logger.error(error_msg)
            
            # Clean up input file
            cleanup_input_cmd = f"rm -f {sh_quote(str(container_input_bin))}"
            run_in_container(cleanup_input_cmd, ignore_shutdown=True)
            
            if pbar:
                pbar.update(1)
            return False, error_msg
        
        if pbar:
            pbar.set_postfix_str(f"Completed: {bin_path.name}")
            pbar.update(1)
        
        return True, success_msg

    except InterruptedError as e:
        # Handle interruption gracefully
        if pbar:
            pbar.update(1)
        return False, f"Interrupted: {bin_path.name} - {str(e)}"
    except Exception as e:
        error_msg = f"Processing failed: {bin_path.name} | Reason: {e}"
        logger.error(error_msg)
        if pbar:
            pbar.update(1)
        return False, error_msg

def sh_quote(s: str) -> str:
    """Simple bash string quoting (prevent spaces, special characters)"""
    return "'" + s.replace("'", "'\"'\"'") + "'"

def process_files_parallel(bin_files: list[Path]) -> tuple[int, int]:
    """
    Process bin files in parallel using ThreadPoolExecutor.
    Returns (success_count, fail_count).
    """
    success_count = 0
    fail_count = 0
    
    # Create progress bar
    with tqdm(total=len(bin_files), desc="Converting files", unit="file") as pbar:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Register executor with process manager
            process_manager.register_executor(executor)
            
            # Submit all tasks
            futures = []
            for bin_file in bin_files:
                if process_manager.is_shutdown_requested():
                    break
                future = executor.submit(convert_one_bin, bin_file, pbar)
                futures.append(future)
                process_manager.register_future(future)
            
            try:
                # Process completed tasks
                for future in as_completed(futures):
                    # Check for shutdown signal
                    if process_manager.is_shutdown_requested():
                        logger.info("Shutdown requested, cancelling remaining tasks...")
                        break
                    
                    try:
                        success, message = future.result(timeout=1)
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.error("Unexpected error: %s", e)
                        fail_count += 1
                        pbar.update(1)
            
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received in process_files_parallel")
                process_manager.request_shutdown()
                raise
    
    return success_count, fail_count

# ===================== Main Process =====================

def main():
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Register cleanup function to run on exit
    atexit.register(cleanup_on_exit)
    
    try:
        check_prereqs()
    except Exception as e:
        logger.critical("Prerequisite check failed: %s", e)
        sys.exit(1)

    bins = sorted(INPUT_LOGS_DIR.glob(BIN_GLOB))
    if not bins:
        logger.warning("No matching bin files found in directory: %s", INPUT_LOGS_DIR)
        return

    logger.info("Found %d bin files to process.", len(bins))
    logger.info("Using %d worker threads for parallel processing.", MAX_WORKERS)
    logger.info("Press Ctrl+C to stop processing gracefully...")

    start_time = time.time()
    
    try:
        # Process files in parallel
        success, fail = process_files_parallel(bins)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if process_manager.is_shutdown_requested():
            logger.info("Processing interrupted by user. Partial results:")
        else:
            logger.info("Processing completed successfully.")
        
        logger.info("Success: %d, Failed: %d", success, fail)
        logger.info("Total processing time: %.2f seconds", elapsed_time)
        if success > 0:
            logger.info("Average time per successful file: %.2f seconds", elapsed_time / success)
    
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt in main, cleaning up...")
        # Cleanup will be handled by signal handler and atexit
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error("Unexpected error in main: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()