#!/usr/bin/env python3
"""
cFS Simulation Runner
Automated script to run sequential cFS/NOS3 simulation experiments with different parameter configurations
"""

import argparse
import logging
import signal
import subprocess
import time
import shutil
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import re


class CFSSimulationRunner:
    def __init__(self, parameter_output_dir: str, nos3_root: str, cosmos_logs_dir: str, 
                 data_output_dir: str, wait_time: int = 60, start_config: int = 0):
        """
        Initialize the cFS simulation runner
        
        Args:
            parameter_output_dir: Path to parameter_sets directory
            nos3_root: Path to NOS3 root directory
            cosmos_logs_dir: Path to COSMOS logs directory
            data_output_dir: Path to output data directory
            wait_time: Wait time in seconds after launch completion
            start_config: Starting configuration number
        """
        # Path setup using pathlib
        self.parameter_output_dir = Path(parameter_output_dir)
        self.nos3_root = Path(nos3_root)
        self.config_target_dir = Path(nos3_root) / "cfg" / "build" / "InOut"
        self.cosmos_logs_dir = Path(cosmos_logs_dir)
        self.data_output_dir = Path(data_output_dir)
        self.sim_log_dir = Path("./sim_log")
        
        # Simulation parameters
        self.wait_time = wait_time
        self.start_config = start_config
        
        # Runtime state - single terminal for all experiments
        self.terminal_process = None
        self.simulation_running = False
        self.logger = None
        
        # Required config files
        self.required_config_files = ['Inp_Sim.txt', 'Orb_LEO.txt', 'SC_NOS3.txt']
        
        # Setup
        self.setup_logging()
        self.setup_signal_handlers()
        self.create_directories()
        
    def setup_logging(self):
        """Setup logging to both file and console"""
        # Create log directory
        self.sim_log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('CFSSimRunner')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_filename = self.sim_log_dir / f"simulation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_filename}")
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.cleanup_and_exit)
        signal.signal(signal.SIGTERM, self.cleanup_and_exit)
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [self.sim_log_dir, self.data_output_dir]
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Created/verified directory: {directory}")
    
    def setup_shared_terminal(self) -> bool:
        """Setup a single gnome-terminal for all experiments"""
        try:
            self.logger.info("Setting up shared gnome-terminal for all experiments")
            
            # Create gnome-terminal command
            terminal_cmd = [
                'gnome-terminal',
                '--title', 'NOS3 Simulation Runner',
                '--geometry=120x30',
                '--working-directory', str(self.nos3_root)
            ]
            
            # Start the terminal (it will stay open for all experiments)
            self.terminal_process = subprocess.Popen(
                terminal_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.nos3_root
            )
            
            # Wait a moment for terminal to initialize
            time.sleep(3)
            self.logger.info("Shared terminal setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup shared terminal: {e}")
            return False
    
    def find_config_directories(self) -> List[Path]:
        """Find all configuration directories"""
        if not self.parameter_output_dir.exists():
            raise FileNotFoundError(f"Parameter output directory not found: {self.parameter_output_dir}")
        
        config_dirs = []
        pattern = re.compile(r'config_(\d+)')
        
        for item in self.parameter_output_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    config_num = int(match.group(1))
                    if config_num >= self.start_config:
                        config_dirs.append(item)
        
        # Sort by configuration number
        config_dirs.sort(key=lambda x: int(re.search(r'config_(\d+)', x.name).group(1)))
        
        self.logger.info(f"Found {len(config_dirs)} configuration directories (starting from config_{self.start_config:04d})")
        return config_dirs
    
    def validate_config_directory(self, config_dir: Path) -> bool:
        """Validate that config directory contains all required files"""
        missing_files = []
        for required_file in self.required_config_files:
            if not (config_dir / required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            self.logger.error(f"Config {config_dir.name} missing files: {missing_files}")
            return False
        
        return True
    
    def copy_config_files(self, config_dir: Path) -> bool:
        """Copy configuration files to target directory"""
        try:
            self.logger.info(f"Copying configuration files from {config_dir.name}")
            
            # Ensure target directory exists
            self.config_target_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy each required file
            for config_file in self.required_config_files:
                src_file = config_dir / config_file
                dst_file = self.config_target_dir / config_file
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.logger.info(f"Copied {config_file} -> {dst_file}")
                else:
                    self.logger.error(f"Source file not found: {src_file}")
                    return False
            
            self.logger.info(f"Successfully copied all configuration files for {config_dir.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy configuration files: {e}")
            return False
    
    def clear_cosmos_logs(self) -> bool:
        """Clear all files in COSMOS logs directory before experiment"""
        try:
            if self.cosmos_logs_dir.exists():
                files_to_remove = list(self.cosmos_logs_dir.glob("*"))
                self.logger.info(f"Clearing {len(files_to_remove)} files from COSMOS logs directory")
                
                for file_path in files_to_remove:
                    if file_path.is_file():
                        file_path.unlink()
                        self.logger.debug(f"Removed: {file_path.name}")
                
                self.logger.info("COSMOS logs directory cleared successfully")
                return True
            else:
                self.logger.warning(f"COSMOS logs directory does not exist: {self.cosmos_logs_dir}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error clearing COSMOS logs directory: {e}")
            return False
    
    def collect_all_cosmos_files(self, config_id: str) -> bool:
        """Collect all files from COSMOS logs directory and move to output directory"""
        try:
            if not self.cosmos_logs_dir.exists():
                self.logger.error(f"COSMOS logs directory does not exist: {self.cosmos_logs_dir}")
                return False
            
            # Find all files in logs directory
            all_files = [f for f in self.cosmos_logs_dir.iterdir() if f.is_file()]
            
            if not all_files:
                self.logger.error("No files found in COSMOS logs directory!")
                return False
            
            self.logger.info(f"Found {len(all_files)} files in COSMOS logs directory:")
            for file_path in all_files:
                self.logger.info(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
            
            # Ensure output directory exists
            self.data_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Move all files with config prefix
            moved_files = []
            for file_path in all_files:
                # Create new filename with config prefix
                new_filename = f"{config_id}_{file_path.name}"
                dst_path = self.data_output_dir / new_filename
                
                # Move file
                shutil.move(str(file_path), str(dst_path))
                moved_files.append(new_filename)
                self.logger.info(f"Moved {file_path.name} -> {dst_path}")
            
            self.logger.info(f"Successfully moved all {len(moved_files)} files for {config_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting COSMOS files: {e}")
            return False
    
    def run_nos3_simulation(self, config_id: str) -> bool:
        """Run NOS3 simulation for a specific configuration"""
        try:
            self.logger.info(f"Starting NOS3 simulation for {config_id}")
            
            # Clear COSMOS logs directory before starting
            if not self.clear_cosmos_logs():
                self.logger.warning("Failed to clear COSMOS logs, but continuing...")
            
            # Run make launch command
            self.logger.info("Executing 'make launch' command")
            launch_process = subprocess.Popen(
                ['make', 'launch'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.nos3_root
            )
            
            # Monitor launch completion
            launch_success = self.monitor_launch_completion(launch_process)
            
            if not launch_success:
                self.logger.error("Failed to launch NOS3 simulation")
                # Try to stop anyway in case something partially started
                self.stop_simulation()
                return False
            
            # Wait specified time for telemetry collection
            self.logger.info(f"Simulation launched successfully. Waiting {self.wait_time} seconds for telemetry collection...")
            self.simulation_running = True
            
            # Precise timing
            start_time = time.time()
            while time.time() - start_time < self.wait_time:
                if not self.simulation_running:  # Check for interruption
                    break
                time.sleep(0.1)  # Small sleep to avoid busy waiting
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Wait period completed. Actual wait time: {elapsed_time:.2f} seconds")
            
            # ALWAYS stop simulation first (this must happen before file collection)
            self.logger.info("Stopping NOS3 simulation...")
            stop_success = self.stop_simulation()
            if not stop_success:
                self.logger.warning("Issues occurred while stopping simulation, but continuing with file collection...")
            
            # Now collect ALL files from COSMOS logs directory
            self.logger.info("Collecting all COSMOS files...")
            collect_success = self.collect_all_cosmos_files(config_id)
            
            if not collect_success:
                self.logger.error("Failed to collect COSMOS files!")
                self.simulation_running = False
                return False
            
            self.simulation_running = False
            return True
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {e}")
            # Always try to stop simulation on error
            try:
                self.stop_simulation()
            except Exception as stop_error:
                self.logger.error(f"Error stopping simulation after exception: {stop_error}")
            self.simulation_running = False
            return False
    
    def monitor_launch_completion(self, process: subprocess.Popen) -> bool:
        """Monitor make launch output for completion"""
        self.logger.info("Monitoring launch process output...")
        
        completion_marker = "Docker launch script completed!"
        timeout_seconds = 300  # 5 minutes timeout
        start_time = time.time()
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Log important output lines
                    line_stripped = line.strip()
                    if any(keyword in line_stripped.lower() for keyword in ['error', 'failed', 'docker', 'completed']):
                        self.logger.info(f"Launch output: {line_stripped}")
                    
                    # Check for completion marker
                    if completion_marker in line:
                        self.logger.info("Launch completion detected!")
                        return True
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.error(f"Launch timeout after {timeout_seconds} seconds")
                    process.terminate()
                    return False
                
                # Check if process ended
                if process.poll() is not None:
                    break
            
            # Process ended without completion marker
            return_code = process.wait()
            if return_code == 0:
                self.logger.warning("Launch process completed successfully but completion marker not found")
                return True
            else:
                self.logger.error(f"Launch process failed with return code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error monitoring launch process: {e}")
            return False
    
    def stop_simulation(self) -> bool:
        """Stop the NOS3 simulation and wait for completion"""
        try:
            self.logger.info("Executing 'make stop' command and waiting for completion...")
            
            stop_process = subprocess.Popen(
                ['make', 'stop'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.nos3_root
            )
            
            # Monitor stop process with timeout and wait for completion
            timeout_seconds = 120  # 2 minutes timeout
            start_time = time.time()
            
            # Wait for process to complete
            while stop_process.poll() is None:
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning("Stop process timeout, forcing termination")
                    stop_process.terminate()
                    time.sleep(5)
                    if stop_process.poll() is None:
                        stop_process.kill()
                    break
                time.sleep(1)
                # Log progress every 10 seconds
                elapsed = int(time.time() - start_time)
                if elapsed > 0 and elapsed % 10 == 0:
                    self.logger.info(f"Waiting for make stop to complete... ({elapsed}s)")
            
            return_code = stop_process.wait()
            elapsed_stop_time = time.time() - start_time
            
            if return_code == 0:
                self.logger.info(f"Simulation stopped successfully in {elapsed_stop_time:.1f} seconds")
                return True
            else:
                self.logger.warning(f"Stop process returned code: {return_code} after {elapsed_stop_time:.1f} seconds")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return False
    
    def run_experiment_sequence(self) -> None:
        """Run the complete sequence of experiments"""
        try:
            # Setup shared terminal first
            if not self.setup_shared_terminal():
                self.logger.error("Failed to setup shared terminal. Exiting.")
                return
            
            # Find all configuration directories
            config_dirs = self.find_config_directories()
            
            if not config_dirs:
                self.logger.error("No configuration directories found!")
                return
            
            total_configs = len(config_dirs)
            self.logger.info(f"Starting experiment sequence with {total_configs} configurations")
            
            successful_experiments = 0
            failed_experiments = 0
            
            for i, config_dir in enumerate(config_dirs, 1):
                config_id = config_dir.name
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"EXPERIMENT {i}/{total_configs}: {config_id}")
                self.logger.info(f"{'='*80}")
                
                # Validate configuration directory
                if not self.validate_config_directory(config_dir):
                    self.logger.error(f"Skipping {config_id} due to missing files")
                    failed_experiments += 1
                    continue
                
                # Copy configuration files
                if not self.copy_config_files(config_dir):
                    self.logger.error(f"Failed to copy config files for {config_id}")
                    failed_experiments += 1
                    continue
                
                # Run simulation
                sim_success = self.run_nos3_simulation(config_id)
                
                if sim_success:
                    successful_experiments += 1
                    self.logger.info(f" Experiment {config_id} completed successfully")
                else:
                    failed_experiments += 1
                    self.logger.error(f" Experiment {config_id} failed")
                    
                    # Option to continue or stop on failure
                    self.logger.info("Continuing to next experiment...")
                
                # Brief pause between experiments
                if i < total_configs:
                    self.logger.info("Pausing 5 seconds before next experiment...")
                    time.sleep(5)
            
            # Final summary
            self.logger.info(f"\n{'='*80}")
            self.logger.info("EXPERIMENT SEQUENCE COMPLETED")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Total experiments: {total_configs}")
            self.logger.info(f"Successful: {successful_experiments}")
            self.logger.info(f"Failed: {failed_experiments}")
            self.logger.info(f"Success rate: {successful_experiments/total_configs*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Fatal error in experiment sequence: {e}")
            raise
        finally:
            # Clean up shared terminal
            self.cleanup_terminal()
    
    def cleanup_terminal(self):
        """Clean up the shared terminal"""
        if self.terminal_process:
            try:
                self.logger.info("Cleaning up shared terminal")
                self.terminal_process.terminate()
                self.terminal_process.wait(timeout=10)
            except Exception as e:
                self.logger.warning(f"Error cleaning up terminal: {e}")
            finally:
                self.terminal_process = None
    
    def cleanup_and_exit(self, signal_num, frame):
        """Handle graceful shutdown on signal"""
        self.logger.info(f"\nReceived signal {signal_num}. Initiating graceful shutdown...")
        
        # Stop current simulation if running
        if self.simulation_running:
            self.logger.info("Stopping current simulation...")
            self.simulation_running = False
            self.stop_simulation()
        
        # Clean up terminal process
        self.cleanup_terminal()
        
        self.logger.info("Cleanup completed. Exiting...")
        sys.exit(0)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run cFS/NOS3 simulation experiments with parameter configurations')
    
    # Path arguments
    parser.add_argument('--parameter-dir', 
                       default='<RVSPEC_ROOT>/CPG-cFS/parameter_generation/parameter_output/parameter_sets',
                       help='Path to parameter_sets directory')
    parser.add_argument('--nos3-root',
                       default='<RVSPEC_ROOT>/UAV/nos3',
                       help='Path to NOS3 root directory')
    parser.add_argument('--cosmos-logs',
                       default='<RVSPEC_ROOT>/UAV/nos3/gsw/cosmos/outputs/logs',
                       help='Path to COSMOS logs directory')
    parser.add_argument('--data-output',
                       default='<DATA_DIR>',
                       help='Path to output data directory')
    
    # Simulation parameters
    parser.add_argument('--wait-time', type=int, default=60,
                       help='Wait time in seconds after launch completion (default: 60)')
    parser.add_argument('--start-config', type=int, default=0,
                       help='Start from specific config number (default: 0)')
    parser.add_argument('--max-configs', type=int,
                       help='Maximum number of configs to run (optional)')
    
    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - validate configs but don\'t run simulations')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize simulation runner
        runner = CFSSimulationRunner(
            parameter_output_dir=args.parameter_dir,
            nos3_root=args.nos3_root,
            cosmos_logs_dir=args.cosmos_logs,
            data_output_dir=args.data_output,
            wait_time=args.wait_time,
            start_config=args.start_config
        )
        
        if args.verbose:
            runner.logger.setLevel(logging.DEBUG)
        
        # Validate paths
        required_paths = [
            (Path(args.parameter_dir), "Parameter directory"),
            (Path(args.nos3_root), "NOS3 root directory"),
            (Path(args.cosmos_logs), "COSMOS logs directory")
        ]
        
        for path, description in required_paths:
            if not path.exists():
                runner.logger.error(f"{description} not found: {path}")
                sys.exit(1)
        
        if args.dry_run:
            runner.logger.info("DRY RUN MODE: Validating configurations...")
            config_dirs = runner.find_config_directories()
            valid_configs = 0
            for config_dir in config_dirs:
                if runner.validate_config_directory(config_dir):
                    valid_configs += 1
                    runner.logger.info(f" {config_dir.name} is valid")
                else:
                    runner.logger.error(f" {config_dir.name} is invalid")
            runner.logger.info(f"Dry run completed. {valid_configs}/{len(config_dirs)} configurations are valid.")
        else:
            # Run the experiment sequence
            runner.run_experiment_sequence()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()