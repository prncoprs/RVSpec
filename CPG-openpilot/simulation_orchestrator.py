#!/usr/bin/env python3
"""
CARLA-OpenPilot CPG-SIM Orchestrator - Fixed Version
Manages Docker containers, configurations, and simulation execution
Removed problematic external activation commands - now uses bridge auto-activation
"""

import argparse
import json
import os
import signal
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import threading


class DockerManager:
    """Manages CARLA and OpenPilot Docker containers with complete logging"""
    
    def __init__(self, config_paths: Dict[str, str]):
        self.config_paths = config_paths
        self.carla_container_name = "carla_sim"
        self.openpilot_container_name = "openpilot_client"
        
    def is_container_running(self, container_name: str) -> bool:
        """Check if a Docker container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            is_running = container_name in result.stdout
            print(f"[DOCKER-DEBUG] Container {container_name} running: {is_running}")
            return is_running
        except subprocess.CalledProcessError as e:
            print(f"[DOCKER-DEBUG] Error checking container {container_name}: {e}")
            return False
    
    def get_container_status(self, container_name: str) -> str:
        """Get detailed container status"""
        try:
            result = subprocess.run([
                "docker", "ps", "-a", "--filter", f"name={container_name}", 
                "--format", "{{.Names}}\t{{.Status}}\t{{.CreatedAt}}"
            ], capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                status = result.stdout.strip()
                print(f"[DOCKER-DEBUG] {container_name} status: {status}")
                return status
            else:
                print(f"[DOCKER-DEBUG] Container {container_name} not found")
                return "Container not found"
        except subprocess.CalledProcessError as e:
            print(f"[DOCKER-DEBUG] Error getting status for {container_name}: {e}")
            return f"Error: {e}"
    
    def get_all_containers_status(self):
        """Get status of all Docker containers"""
        try:
            result = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True, check=True)
            print(f"[DOCKER-DEBUG] All containers status:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[DOCKER-DEBUG] Error getting all containers: {e}")
    
    def stop_container(self, container_name: str):
        """Stop a Docker container"""
        try:
            print(f"[DOCKER-DEBUG] Attempting to stop container: {container_name}")
            if self.is_container_running(container_name):
                print(f"[DOCKER-DEBUG] Container {container_name} is running, stopping it...")
                subprocess.run(["docker", "kill", container_name], check=True)
                time.sleep(3)
                print(f"[DOCKER-DEBUG] Container {container_name} stopped successfully")
            else:
                print(f"[DOCKER-DEBUG] Container {container_name} was not running")
        except subprocess.CalledProcessError as e:
            print(f"[DOCKER-DEBUG] Warning: Failed to stop container {container_name}: {e}")
    
    def wait_for_carla_ready(self, host="127.0.0.1", port=2000, timeout=90):
        """Wait for CARLA server to be ready"""
        import socket
        
        print(f"[CARLA-DEBUG] Waiting for CARLA server at {host}:{port}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if container is still running first
            if not self.is_container_running(self.carla_container_name):
                print(f"[CARLA-DEBUG] ERROR: CARLA container stopped while waiting for server!")
                self.get_container_status(self.carla_container_name)
                raise RuntimeError("CARLA container stopped unexpectedly")
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    elapsed = time.time() - start_time
                    print(f"[CARLA-DEBUG] CARLA server is ready! (took {elapsed:.1f}s)")
                    time.sleep(5)  # Give CARLA more time to fully initialize
                    return True
                    
            except Exception as e:
                print(f"[CARLA-DEBUG] Connection attempt failed: {e}")
            
            elapsed = time.time() - start_time
            print(f"[CARLA-DEBUG] Still waiting for CARLA... ({elapsed:.1f}s/{timeout}s)")
            time.sleep(3)
        
        print(f"[CARLA-DEBUG] Final container check before timeout:")
        self.get_container_status(self.carla_container_name)
        raise RuntimeError(f"CARLA server not ready after {timeout} seconds")
    
    def start_carla_container(self, quality_level="Low", fps=20, log_dir=None):
        """Start CARLA container with complete logging"""
        print(f"[CARLA-DEBUG] === Starting CARLA Container ===")
        
        # First, make sure no existing container is running
        self.stop_container(self.carla_container_name)
        time.sleep(2)
        
        if self.is_container_running(self.carla_container_name):
            print(f"[CARLA-DEBUG] CARLA container already running, skipping start")
            return
        
        print(f"[CARLA-DEBUG] Building CARLA docker command...")
        
        # Build CARLA docker command with detached mode
        carla_cmd = [
            "docker", "run",
            "--name", self.carla_container_name,
            "--rm", "-d",  # Detached mode
            "--gpus", "all",
            "--net=host",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "--memory=8g",
            "--memory-swap=8g",
            "carlasim/carla:0.9.13",
            "/bin/bash", "./CarlaUE4.sh",
            "-nosound", "-RenderOffScreen", "-benchmark",
            f"-fps={fps}", f"-quality-level={quality_level}"
        ]
        
        print(f"[CARLA-DEBUG] CARLA command: {' '.join(carla_cmd)}")
        
        try:
            print(f"[CARLA-DEBUG] Executing docker run command...")
            result = subprocess.run(carla_cmd, check=True, capture_output=True, text=True, timeout=30)
            container_id = result.stdout.strip()
            print(f"[CARLA-DEBUG] CARLA container started with ID: {container_id}")
            
            # Start logging CARLA output in background
            if log_dir:
                carla_log_file = Path(log_dir) / "carla_output.log"
                print(f"[CARLA-DEBUG] Starting CARLA log collection to: {carla_log_file}")
                
                def collect_carla_logs():
                    try:
                        with open(carla_log_file, 'w') as f:
                            f.write(f"=== CARLA Container Logs - Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                            f.flush()
                            
                            logs_cmd = ["docker", "logs", "-f", self.carla_container_name]
                            process = subprocess.Popen(logs_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                            
                            for line in process.stdout:
                                f.write(line)
                                f.flush()
                                if any(keyword in line.lower() for keyword in ['error', 'warning', 'crash', 'exception', 'signal']):
                                    print(f"[CARLA-LOG] {line.strip()}")
                                    
                    except Exception as e:
                        print(f"[CARLA-DEBUG] Error collecting logs: {e}")
                
                log_thread = threading.Thread(target=collect_carla_logs, daemon=True)
                log_thread.start()
                self.carla_log_thread = log_thread
            
            # Give CARLA time to start up
            print(f"[CARLA-DEBUG] Giving CARLA time to initialize...")
            time.sleep(10)
            
            # Verify container started
            if self.is_container_running(self.carla_container_name):
                print(f"[CARLA-DEBUG] CARLA container verified running")
            else:
                print(f"[CARLA-DEBUG] ERROR: CARLA container not running after start!")
                self.get_container_status(self.carla_container_name)
                if log_dir:
                    immediate_logs = subprocess.run(["docker", "logs", self.carla_container_name], 
                                                  capture_output=True, text=True)
                    print(f"[CARLA-DEBUG] Container logs: {immediate_logs.stdout}")
                    print(f"[CARLA-DEBUG] Container errors: {immediate_logs.stderr}")
                raise RuntimeError("CARLA container failed to start")
            
            print(f"[CARLA-DEBUG] Waiting for CARLA server to be ready...")
            self.wait_for_carla_ready()
            print(f"[CARLA-DEBUG] CARLA startup completed successfully")
            
        except subprocess.TimeoutExpired:
            print(f"[CARLA-DEBUG] ERROR: CARLA container start timed out")
            raise RuntimeError("CARLA container start timed out")
        except subprocess.CalledProcessError as e:
            print(f"[CARLA-DEBUG] ERROR: Failed to start CARLA container")
            print(f"[CARLA-DEBUG] Return code: {e.returncode}")
            print(f"[CARLA-DEBUG] stdout: {e.stdout}")
            print(f"[CARLA-DEBUG] stderr: {e.stderr}")
            raise RuntimeError(f"Failed to start CARLA container: {e}")
    
    def start_openpilot_container(self, parameter_config_path: str, 
                                logs_output_path: str, bridge_script_path: str,
                                dual_camera=False, high_quality=False):
        """Start OpenPilot container with complete logging"""
        print(f"[OPENPILOT-DEBUG] === Starting OpenPilot Container ===")
        
        # Check CARLA status before starting OpenPilot
        print(f"[OPENPILOT-DEBUG] Pre-start: Checking CARLA container status...")
        carla_running = self.is_container_running(self.carla_container_name)
        print(f"[OPENPILOT-DEBUG] Pre-start: CARLA running = {carla_running}")
        
        if not carla_running:
            print(f"[OPENPILOT-DEBUG] ERROR: CARLA container not running before OpenPilot start!")
            raise RuntimeError("CARLA must be running before starting OpenPilot")
        
        # Stop existing OpenPilot container
        if self.is_container_running(self.openpilot_container_name):
            print(f"[OPENPILOT-DEBUG] Stopping existing OpenPilot container...")
            self.stop_container(self.openpilot_container_name)
        
        print(f"[OPENPILOT-DEBUG] Building OpenPilot docker command...")
        
        # Prepare bridge arguments
        bridge_args = []
        if dual_camera:
            bridge_args.append("--dual_camera")
        if high_quality:
            bridge_args.append("--high_quality")
        bridge_args.extend(["--config", "/shared/config/current_run.json"])
        bridge_args_str = " ".join(bridge_args)
        
        # Simplified script without problematic activation commands
        script_content = f"""#!/bin/bash
echo "[CONTAINER-DEBUG] Starting OpenPilot and Bridge..."
echo "[CONTAINER-DEBUG] Timestamp: $(date)"
echo "[CONTAINER-DEBUG] Working directory: $(pwd)"

cd /openpilot/tools/sim

# Create log files
mkdir -p /shared/logs
OPENPILOT_LOG="/shared/logs/openpilot_stdout.log"
BRIDGE_LOG="/shared/logs/bridge_stdout.log"
COMBINED_LOG="/shared/logs/combined.log"

echo "[CONTAINER-DEBUG] Starting OpenPilot..." | tee -a $COMBINED_LOG
./launch_openpilot.sh > $OPENPILOT_LOG 2>&1 &
OPENPILOT_PID=$!
echo "[CONTAINER-DEBUG] OpenPilot started with PID: $OPENPILOT_PID" | tee -a $COMBINED_LOG

# Wait for OpenPilot to initialize
echo "[CONTAINER-DEBUG] Waiting for OpenPilot to initialize..." | tee -a $COMBINED_LOG
sleep 15

# Check if OpenPilot is still running
if ! kill -0 $OPENPILOT_PID 2>/dev/null; then
    echo "[CONTAINER-ERROR] OpenPilot process died during initialization!" | tee -a $COMBINED_LOG
    exit 1
fi

echo "[CONTAINER-DEBUG] OpenPilot initialization complete" | tee -a $COMBINED_LOG

# Start the bridge (which now handles auto-activation internally)
echo "[CONTAINER-DEBUG] Starting bridge with args: {bridge_args_str}" | tee -a $COMBINED_LOG
python /shared/bridge/carla_bridge.py {bridge_args_str} > $BRIDGE_LOG 2>&1
BRIDGE_EXIT_CODE=$?

echo "[CONTAINER-DEBUG] Bridge exited with code: $BRIDGE_EXIT_CODE" | tee -a $COMBINED_LOG

# Cleanup
kill $OPENPILOT_PID 2>/dev/null || true
wait $OPENPILOT_PID 2>/dev/null || true

echo "[CONTAINER-DEBUG] Container script completed" | tee -a $COMBINED_LOG
exit $BRIDGE_EXIT_CODE
"""
        
        # Write script to shared location
        shared_scripts_dir = Path(self.config_paths['SHARED_SCRIPTS'])
        shared_scripts_dir.mkdir(exist_ok=True, parents=True)
        script_path = shared_scripts_dir / "run_simulation.sh"
        
        print(f"[OPENPILOT-DEBUG] Writing run script to: {script_path}")
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        # Create logs directory
        logs_dir = Path(logs_output_path)
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Build OpenPilot docker command
        openpilot_cmd = [
            "docker", "run", "--net=host",
            "--name", self.openpilot_container_name,
            "--rm", "-it",
            "--gpus", "all",
            "--device=/dev/dri:/dev/dri",
            "--device=/dev/input:/dev/input",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            "--shm-size", "2G",
            "-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
            "-e", "QT_X11_NO_MITSHM=1",
            
            # Volume mounts
            "-v", f"{parameter_config_path}:/shared/config:ro",
            "-v", f"{logs_output_path}:/shared/logs:rw", 
            "-v", f"{bridge_script_path}:/shared/bridge/carla_bridge.py:ro",
            "-v", f"{script_path}:/openpilot/tools/sim/run_simulation.sh:ro",
            
            # Working directory
            "-w", "/openpilot/tools/sim",
            
            # Container image
            "ghcr.io/commaai/openpilot-sim@sha256:7f8a84850f2be55a063b6dc62ca080d4c63c3b2ae0c0a36a8ebb93171cb723be",
            
            # Command to run
            "/bin/bash", "./run_simulation.sh"
        ]
        
        print(f"[OPENPILOT-DEBUG] OpenPilot command: {' '.join(openpilot_cmd[:15])}... (truncated)")
        
        # Set xhost permissions
        try:
            subprocess.run(["xhost", "+local:root"], check=True)
            print(f"[OPENPILOT-DEBUG] xhost permissions set")
        except subprocess.CalledProcessError:
            print(f"[OPENPILOT-DEBUG] Warning: Could not set xhost permissions")
        
        # Start container with full output capture
        try:
            print(f"[OPENPILOT-DEBUG] Starting OpenPilot container...")
            
            # Start container and capture all output
            openpilot_log_file = logs_dir / "openpilot_container_output.log"
            print(f"[OPENPILOT-DEBUG] Container output will be logged to: {openpilot_log_file}")
            
            with open(openpilot_log_file, 'w') as log_file:
                log_file.write(f"=== OpenPilot Container Output - Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                log_file.write(f"Command: {' '.join(openpilot_cmd)}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()
                
                self.openpilot_process = subprocess.Popen(
                    openpilot_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Start output capture thread
                def capture_output():
                    try:
                        for line in self.openpilot_process.stdout:
                            log_file.write(line)
                            log_file.flush()
                            if any(keyword in line.lower() for keyword in ['error', 'warning', 'debug', 'failed', 'success', 'autoactivation']):
                                print(f"[OPENPILOT-LOG] {line.strip()}")
                    except Exception as e:
                        print(f"[OPENPILOT-DEBUG] Error capturing output: {e}")
                
                output_thread = threading.Thread(target=capture_output, daemon=True)
                output_thread.start()
                self.openpilot_output_thread = output_thread
            
            print(f"[OPENPILOT-DEBUG] OpenPilot container started with PID: {self.openpilot_process.pid}")
            
            # Give container time to start
            time.sleep(5)
            
            # Check if process is still running
            if self.openpilot_process.poll() is not None:
                print(f"[OPENPILOT-DEBUG] ERROR: OpenPilot container exited immediately!")
                with open(openpilot_log_file, 'r') as f:
                    output = f.read()
                print(f"[OPENPILOT-DEBUG] Container output:\n{output}")
                raise RuntimeError("OpenPilot container failed to start")
            
            return self.openpilot_process
            
        except Exception as e:
            print(f"[OPENPILOT-DEBUG] ERROR: Failed to start OpenPilot container: {e}")
            raise RuntimeError(f"Failed to start OpenPilot container: {e}")
    
    def cleanup_containers(self):
        """Stop all containers with detailed logging"""
        print(f"[CLEANUP-DEBUG] === Cleaning up containers ===")
        self.get_all_containers_status()
        
        # Stop log collection threads if they exist
        if hasattr(self, 'carla_log_thread'):
            print(f"[CLEANUP-DEBUG] Stopping CARLA log collection...")
        if hasattr(self, 'openpilot_output_thread'):
            print(f"[CLEANUP-DEBUG] Stopping OpenPilot output collection...")
        
        print(f"[CLEANUP-DEBUG] Stopping OpenPilot container...")
        self.stop_container(self.openpilot_container_name)
        print(f"[CLEANUP-DEBUG] Stopping CARLA container...")
        self.stop_container(self.carla_container_name)
        time.sleep(3)
        print(f"[CLEANUP-DEBUG] Cleanup completed")


class CPGSimOrchestrator:
    """Main orchestrator for CPG-SIM execution"""
    
    def __init__(self, parameter_sets_dir: str, output_dir: str, bridge_script: str,
                 carla_host="127.0.0.1", carla_port=2000):
        self.parameter_sets_dir = Path(parameter_sets_dir)
        self.output_dir = Path(output_dir)
        self.bridge_script = Path(bridge_script)
        self.carla_host = carla_host
        self.carla_port = carla_port
        
        # Validate inputs
        if not self.parameter_sets_dir.exists():
            raise FileNotFoundError(f"Parameter sets directory not found: {self.parameter_sets_dir}")
        if not self.bridge_script.exists():
            raise FileNotFoundError(f"Bridge script not found: {self.bridge_script}")
        
        # Create output directories
        self.logs_dir = self.output_dir / "simulation_logs"
        self.shared_config_dir = self.output_dir / "shared_config"
        self.shared_scripts_dir = self.output_dir / "shared_scripts"
        
        for dir_path in [self.logs_dir, self.shared_config_dir, self.shared_scripts_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Configuration paths for Docker manager
        self.config_paths = {
            'PARAMETER_SETS': str(self.parameter_sets_dir),
            'OUTPUT_DIR': str(self.output_dir),
            'LOGS_DIR': str(self.logs_dir),
            'SHARED_CONFIG': str(self.shared_config_dir),
            'SHARED_SCRIPTS': str(self.shared_scripts_dir),
            'BRIDGE_SCRIPT': str(self.bridge_script)
        }
        
        self.docker_manager = DockerManager(self.config_paths)
        self.current_run = None
        self.keep_running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[SIGNAL-DEBUG] Received signal {signum}, shutting down...")
        self.keep_running = False
        self.docker_manager.cleanup_containers()
    
    def get_parameter_sets(self) -> List[Path]:
        """Get list of parameter set directories"""
        param_dirs = []
        if self.parameter_sets_dir.exists():
            param_dirs = sorted([d for d in self.parameter_sets_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('param_')])
        
        print(f"[PARAMS-DEBUG] Found {len(param_dirs)} parameter sets")
        for i, param_dir in enumerate(param_dirs[:5]):
            print(f"[PARAMS-DEBUG]   {i}: {param_dir.name}")
        if len(param_dirs) > 5:
            print(f"[PARAMS-DEBUG]   ... and {len(param_dirs) - 5} more")
        
        return param_dirs
    
    def prepare_run_configuration(self, param_set_dir: Path) -> Path:
        """Prepare configuration for a single run"""
        config_file = param_set_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Validate config file
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG-DEBUG] Loaded config with keys: {list(config.keys())}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_file}: {e}")
        
        # Copy config to shared location for container access
        shared_config_file = self.shared_config_dir / "current_run.json"
        shutil.copy2(config_file, shared_config_file)
        
        print(f"[CONFIG-DEBUG] Prepared configuration for run: {param_set_dir.name}")
        return shared_config_file
    
    def create_run_log_directory(self, param_set_id: str) -> Path:
        """Create log directory for a specific run"""
        run_log_dir = self.logs_dir / param_set_id
        run_log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different types of logs
        (run_log_dir / "carla").mkdir(exist_ok=True)
        (run_log_dir / "openpilot").mkdir(exist_ok=True)
        (run_log_dir / "bridge").mkdir(exist_ok=True)
        (run_log_dir / "container").mkdir(exist_ok=True)
        
        print(f"[LOGS-DEBUG] Created log directory: {run_log_dir}")
        print(f"[LOGS-DEBUG] Log subdirectories: carla/, openpilot/, bridge/, container/")
        return run_log_dir
    
    def monitor_simulation(self, param_set_dir: Path, openpilot_process, 
                         simulation_duration: int, timeout_margin: int = 120) -> bool:
        """Monitor simulation progress with detailed container tracking"""
        print(f"[MONITOR-DEBUG] === Starting Simulation Monitoring ===")
        print(f"[MONITOR-DEBUG] Duration: {simulation_duration}s + {timeout_margin}s margin")
        
        start_time = time.time()
        max_duration = simulation_duration + timeout_margin
        check_interval = 15
        
        while time.time() - start_time < max_duration and self.keep_running:
            elapsed = time.time() - start_time
            remaining = max_duration - elapsed
            
            print(f"\n[MONITOR-DEBUG] === Status Check at {elapsed:.0f}s ===")
            print(f"[MONITOR-DEBUG] Remaining time: {remaining:.0f}s")
            
            # Check OpenPilot process status
            openpilot_poll = openpilot_process.poll()
            print(f"[MONITOR-DEBUG] OpenPilot process poll: {openpilot_poll}")
            
            # Check container statuses
            carla_running = self.docker_manager.is_container_running(self.docker_manager.carla_container_name)
            openpilot_running = self.docker_manager.is_container_running(self.docker_manager.openpilot_container_name)
            
            print(f"[MONITOR-DEBUG] CARLA container running: {carla_running}")
            print(f"[MONITOR-DEBUG] OpenPilot container running: {openpilot_running}")
            
            # Detailed status check
            if not carla_running:
                print(f"[MONITOR-DEBUG] CARLA CONTAINER STOPPED!")
                self.docker_manager.get_container_status(self.docker_manager.carla_container_name)
                return False
            
            # Check if OpenPilot process finished
            if openpilot_poll is not None:
                print(f"[MONITOR-DEBUG] OpenPilot process finished with code: {openpilot_poll}")
                if openpilot_poll == 0:
                    print(f"[MONITOR-DEBUG] Simulation completed successfully")
                    return True
                else:
                    print(f"[MONITOR-DEBUG] Simulation failed with return code: {openpilot_poll}")
                    return False
            
            # Check if we've reached the expected simulation duration
            if elapsed >= simulation_duration:
                print(f"[MONITOR-DEBUG] Simulation duration reached, in grace period...")
            
            print(f"[MONITOR-DEBUG] Continuing monitoring...")
            time.sleep(check_interval)
        
        # If we get here, simulation timed out
        print(f"[MONITOR-DEBUG] Simulation timed out after {max_duration}s, terminating...")
        try:
            openpilot_process.terminate()
            openpilot_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"[MONITOR-DEBUG] Process didn't terminate, killing...")
            openpilot_process.kill()
            openpilot_process.wait(timeout=5)
        
        return False
    
    def collect_run_results(self, param_set_dir: Path, run_log_dir: Path):
        """Collect and organize results from a simulation run"""
        print(f"[RESULTS-DEBUG] Collecting results for run: {param_set_dir.name}")
        
        # Copy parameter configuration to results
        config_file = param_set_dir / "config.json"
        if config_file.exists():
            shutil.copy2(config_file, run_log_dir / "run_config.json")
            print(f"[RESULTS-DEBUG] Copied config file")
        
        # Create run summary
        summary = {
            "run_id": param_set_dir.name,
            "completion_time": time.time(),
            "status": "completed",
            "log_directory": str(run_log_dir),
            "parameter_set_path": str(param_set_dir)
        }
        
        summary_file = run_log_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[RESULTS-DEBUG] Results collected in: {run_log_dir}")
    
    def run_single_simulation(self, param_set_dir: Path, run_options: Dict) -> bool:
        """Execute a single simulation run with detailed debugging"""
        param_set_id = param_set_dir.name
        print(f"\n{'='*80}")
        print(f"[RUN-DEBUG] Starting simulation run: {param_set_id}")
        print(f"{'='*80}")
        
        try:
            # Load configuration to get simulation parameters
            config_file = param_set_dir / "config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            simulation_duration = config.get('run_metadata', {}).get('simulation_duration', 300)
            print(f"[RUN-DEBUG] Simulation duration: {simulation_duration}s")
            
            # Prepare run environment
            print(f"[RUN-DEBUG] Preparing run environment...")
            self.prepare_run_configuration(param_set_dir)
            run_log_dir = self.create_run_log_directory(param_set_id)
            
            # Start OpenPilot container with current configuration
            print(f"[RUN-DEBUG] Starting OpenPilot container...")
            openpilot_process = self.docker_manager.start_openpilot_container(
                parameter_config_path=str(self.shared_config_dir),
                logs_output_path=str(run_log_dir),
                bridge_script_path=str(self.bridge_script),
                dual_camera=run_options.get('dual_camera', False),
                high_quality=run_options.get('high_quality', False)
            )
            
            print(f"[RUN-DEBUG] OpenPilot container started, bridge will handle auto-activation")
            
            # Monitor simulation
            print(f"[RUN-DEBUG] Starting simulation monitoring...")
            success = self.monitor_simulation(
                param_set_dir, openpilot_process, simulation_duration
            )
            
            # Collect results
            if success:
                print(f"[RUN-DEBUG] Simulation successful, collecting results...")
                self.collect_run_results(param_set_dir, run_log_dir)
            else:
                print(f"[RUN-DEBUG] Simulation failed, logging failure...")
                # Still create a summary for failed runs
                summary = {
                    "run_id": param_set_id,
                    "completion_time": time.time(),
                    "status": "failed",
                    "log_directory": str(run_log_dir),
                    "parameter_set_path": str(param_set_dir)
                }
                summary_file = run_log_dir / "run_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            
            # Clean up OpenPilot container for next run
            print(f"[RUN-DEBUG] Cleaning up OpenPilot container...")
            self.docker_manager.stop_container(self.docker_manager.openpilot_container_name)
            
            # Brief pause between runs
            if self.keep_running:
                print(f"[RUN-DEBUG] Pausing before next run...")
                time.sleep(10)
            
            print(f"[RUN-DEBUG] Run {param_set_id} completed with success={success}")
            return success
            
        except Exception as e:
            print(f"[RUN-DEBUG] ERROR in simulation run {param_set_id}: {e}")
            print(f"[RUN-DEBUG] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_cpg_sim(self, start_from: int = 0, max_runs: Optional[int] = None,
                          run_options: Optional[Dict] = None):
        """Execute complete CPG-SIM with detailed debugging"""
        if run_options is None:
            run_options = {}
        
        print(f"[MAIN-DEBUG] *** Starting CARLA-OpenPilot CPG-SIM ***")
        print(f"[MAIN-DEBUG] Parameter sets directory: {self.parameter_sets_dir}")
        print(f"[MAIN-DEBUG] Output directory: {self.output_dir}")
        print(f"[MAIN-DEBUG] Bridge script: {self.bridge_script}")
        
        # Get parameter sets
        param_sets = self.get_parameter_sets()
        if not param_sets:
            raise RuntimeError("No parameter sets found!")
        
        # Apply run limits
        if start_from > 0:
            param_sets = param_sets[start_from:]
            print(f"[MAIN-DEBUG] Starting from run {start_from}")
        
        if max_runs is not None:
            param_sets = param_sets[:max_runs]
            print(f"[MAIN-DEBUG] Limited to {max_runs} runs")
        
        print(f"[MAIN-DEBUG] Will execute {len(param_sets)} simulation runs")
        
        # Initialize CARLA container once
        print(f"[MAIN-DEBUG] Initializing CARLA container...")
        try:
            # Pass the logs directory for CARLA logging
            self.docker_manager.start_carla_container(
                quality_level=run_options.get('quality_level', 'Low'),
                fps=run_options.get('fps', 20),
                log_dir=str(self.logs_dir)
            )
        except Exception as e:
            print(f"[MAIN-DEBUG] Failed to start CARLA: {e}")
            return False
        
        # Execute runs
        successful_runs = 0
        failed_runs = 0
        
        for i, param_set_dir in enumerate(param_sets):
            if not self.keep_running:
                print(f"[MAIN-DEBUG] Received shutdown signal, stopping...")
                break
            
            print(f"\n[MAIN-DEBUG] === Progress: {i+1}/{len(param_sets)} ===")
            
            success = self.run_single_simulation(param_set_dir, run_options)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
            
            print(f"[MAIN-DEBUG] Run {param_set_dir.name}: {'SUCCESS' if success else 'FAILED'}")
            print(f"[MAIN-DEBUG] Overall progress: {successful_runs} successful, {failed_runs} failed")
        
        # Cleanup
        print(f"[MAIN-DEBUG] Final cleanup...")
        self.docker_manager.cleanup_containers()
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"[MAIN-DEBUG] CPG-SIM Complete!")
        print(f"[MAIN-DEBUG] Total runs: {successful_runs + failed_runs}")
        print(f"[MAIN-DEBUG] Successful: {successful_runs}")
        print(f"[MAIN-DEBUG] Failed: {failed_runs}")
        print(f"[MAIN-DEBUG] Results saved in: {self.output_dir}")
        print(f"{'='*80}")
        
        return failed_runs == 0


def main():
    parser = argparse.ArgumentParser(description='CARLA-OpenPilot CPG-SIM Orchestrator')
    parser.add_argument('--parameter-sets', 
                       default='<RVSPEC_ROOT>/CPG-openpilot/parameter_generation/parameter_output/parameter_sets',
                       help='Directory containing parameter sets')
    parser.add_argument('--output-dir', 
                       default='<DATA_DIR>',
                       help='Output directory for simulation results')
    parser.add_argument('--bridge-script', 
                       default='<RVSPEC_ROOT>/UAV/openpilot/openpilot/tools/sim/carla_bridge.py',
                       help='Path to carla_bridge.py script')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from specific run number (for resuming)')
    parser.add_argument('--max-runs', type=int, default=1,
                       help='Maximum number of runs to execute (default: 1 for testing)')
    parser.add_argument('--dual-camera', action='store_true',
                       help='Enable dual camera setup')
    parser.add_argument('--high-quality', action='store_true',
                       help='Enable high quality rendering')
    parser.add_argument('--quality-level', default='Low',
                       choices=['Low', 'Medium', 'High', 'Epic'],
                       help='CARLA rendering quality level')
    parser.add_argument('--fps', type=int, default=20,
                       help='CARLA simulation FPS')
    parser.add_argument('--carla-host', default='127.0.0.1',
                       help='CARLA server host')
    parser.add_argument('--carla-port', type=int, default=2000,
                       help='CARLA server port')
    
    args = parser.parse_args()
    
    # Prepare run options
    run_options = {
        'dual_camera': args.dual_camera,
        'high_quality': args.high_quality,
        'quality_level': args.quality_level,
        'fps': args.fps
    }
    
    print(f"[MAIN-DEBUG] Run options: {run_options}")
    
    # Create orchestrator and run CPG simulation
    try:
        orchestrator = CPGSimOrchestrator(
            parameter_sets_dir=args.parameter_sets,
            output_dir=args.output_dir,
            bridge_script=args.bridge_script,
            carla_host=args.carla_host,
            carla_port=args.carla_port
        )
        
        success = orchestrator.run_cpg_sim(
            start_from=args.start_from,
            max_runs=args.max_runs,
            run_options=run_options
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n[MAIN-DEBUG] Received Ctrl+C, shutting down...")
        return 1
    except Exception as e:
        print(f"[MAIN-DEBUG] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())