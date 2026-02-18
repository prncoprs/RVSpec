#!/usr/bin/env python3
"""
cFS Simulation Runner for 42 Data Collection
Modified version to collect 42 simulator logs instead of COSMOS telemetry
"""

import argparse
import logging
import signal
import subprocess
import time
import shutil
import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import re


class CFSSimulationRunner:
    def __init__(self, parameter_dir: str, nos3_root: str, 
                 data_output_dir: str, wait_time: int = 60, 
                 start_config: int = 0, end_config: Optional[int] = None):
        """
        Initialize the cFS simulation runner for 42 data collection
        
        Args:
            parameter_dir: Path to parameter_sets directory
            nos3_root: Path to NOS3 root directory
            data_output_dir: Path to output data directory for 42 logs
            wait_time: Wait time in seconds for simulation
            start_config: Starting configuration number
            end_config: Ending configuration number (optional)
        """
        # Path setup using pathlib
        self.parameter_dir = Path(parameter_dir)
        self.nos3_root = Path(nos3_root)
        self.config_target_dir = Path(nos3_root) / "cfg" / "build" / "InOut"
        self.data_output_dir = Path(data_output_dir)
        self.sim_log_dir = Path("./sim_log")
        
        # Docker container name for 42 simulator
        self.container_name = "sc01-fortytwo"
        self.container_42_path = "/home/nos3/.nos3/42/NOS3InOut"
        
        # Simulation parameters
        self.wait_time = wait_time
        self.start_config = start_config
        self.end_config = end_config
        
        # Runtime state
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
        log_filename = self.sim_log_dir / f"simulation_42_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    def find_config_directories(self) -> List[Path]:
        """Find all configuration directories"""
        if not self.parameter_dir.exists():
            raise FileNotFoundError(f"Parameter directory not found: {self.parameter_dir}")
        
        config_dirs = []
        pattern = re.compile(r'config_(\d+)')
        
        for item in self.parameter_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    config_num = int(match.group(1))
                    if config_num >= self.start_config:
                        if self.end_config is None or config_num <= self.end_config:
                            config_dirs.append(item)
        
        # Sort by configuration number
        config_dirs.sort(key=lambda x: int(re.search(r'config_(\d+)', x.name).group(1)))
        
        self.logger.info(f"Found {len(config_dirs)} configuration directories")
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
    
    def copy_config_files_to_nos3(self, config_dir: Path) -> bool:
        """Copy configuration files to NOS3 target directory"""
        try:
            self.logger.info(f"Copying configuration files from {config_dir.name} to NOS3")
            
            # Ensure target directory exists
            self.config_target_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy each required file
            for config_file in self.required_config_files:
                src_file = config_dir / config_file
                dst_file = self.config_target_dir / config_file
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    self.logger.debug(f"Copied {config_file} -> {dst_file}")
                else:
                    self.logger.error(f"Source file not found: {src_file}")
                    return False
            
            self.logger.info(f"Successfully copied configuration files for {config_dir.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy configuration files: {e}")
            return False
    
    def save_config_to_output(self, config_dir: Path, output_dir: Path) -> bool:
        """Save configuration files to output directory for record keeping"""
        try:
            config_output_dir = output_dir / "config"
            config_output_dir.mkdir(exist_ok=True, parents=True)
            
            for config_file in self.required_config_files:
                src_file = config_dir / config_file
                dst_file = config_output_dir / config_file
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    
            self.logger.info(f"Saved configuration files to {config_output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration files: {e}")
            return False
    
    
    def atomic_copy_42_data(self, config_id: str, output_dir: Path) -> bool:
        """Copy 42 simulator data from host machine (not container)"""
        try:
            # 42
            host_42_path = Path.home() / ".nos3" / "42" / "NOS3InOut"
            
            self.logger.info(f"Copying 42 data from host: {host_42_path}")
            
            if not host_42_path.exists():
                self.logger.error(f"42 data directory not found: {host_42_path}")
                return False
            
            # 
            data_files = list(host_42_path.glob("*.42"))
            if not data_files:
                self.logger.error("No .42 files found in directory")
                return False
            
            self.logger.info(f"Found {len(data_files)} .42 files")
            
            # 
            nos3_output_dir = output_dir / "NOS3InOut"
            
            # 
            if nos3_output_dir.exists():
                shutil.rmtree(nos3_output_dir)
            
            # 
            shutil.copytree(host_42_path, nos3_output_dir)
            
            # 
            key_files = ["time.42", "PosN.42", "Albedo.42", "Illum.42", "svn.42", "svb.42"]
            missing_files = []
            for key_file in key_files:
                if not (nos3_output_dir / key_file).exists():
                    missing_files.append(key_file)
            
            if missing_files:
                self.logger.warning(f"Missing key files: {missing_files}")
            
            self.logger.info(f"Successfully copied 42 data to {nos3_output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error copying 42 data: {e}")
            return False
        
        
    # def atomic_copy_42_data(self, config_id: str, output_dir: Path) -> bool:
    #     """Atomically copy 42 simulator data from Docker container"""
    #     try:
    #         self.logger.info(f"Copying 42 data from container {self.container_name}")
            
    #         # Create temp archive in container
    #         temp_archive = "/tmp/nos3_42_data.tgz"
            
    #         # Package data in container
    #         pack_cmd = [
    #             "docker", "exec", self.container_name,
    #             "bash", "-c",
    #             f"cd {self.container_42_path} && tar -czf {temp_archive} ."
    #         ]
            
    #         result = subprocess.run(pack_cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             self.logger.error(f"Failed to package 42 data: {result.stderr}")
    #             return False
            
    #         # Copy archive to host
    #         local_archive = output_dir / "nos3_42_data.tgz"
    #         copy_cmd = [
    #             "docker", "cp",
    #             f"{self.container_name}:{temp_archive}",
    #             str(local_archive)
    #         ]
            
    #         result = subprocess.run(copy_cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             self.logger.error(f"Failed to copy archive: {result.stderr}")
    #             return False
            
    #         # Extract archive
    #         nos3_output_dir = output_dir / "NOS3InOut"
    #         nos3_output_dir.mkdir(exist_ok=True, parents=True)
            
    #         extract_cmd = [
    #             "tar", "-xzf", str(local_archive),
    #             "-C", str(nos3_output_dir)
    #         ]
            
    #         result = subprocess.run(extract_cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             self.logger.error(f"Failed to extract archive: {result.stderr}")
    #             return False
            
    #         # Remove archive file
    #         local_archive.unlink()
            
    #         # Verify key files exist
    #         key_files = ["time.42", "PosN.42", "Albedo.42", "Illum.42"]
    #         for key_file in key_files:
    #             if not (nos3_output_dir / key_file).exists():
    #                 self.logger.warning(f"Key file missing: {key_file}")
            
    #         self.logger.info(f"Successfully copied 42 data to {nos3_output_dir}")
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"Error copying 42 data: {e}")
    #         return False
    
    def extract_parameters_from_config(self, config_dir: Path) -> Dict:
        """Extract key parameters from configuration files"""
        params = {}
        
        try:
            # Extract from SC_NOS3.txt
            sc_file = config_dir / "SC_NOS3.txt"
            if sc_file.exists():
                content = sc_file.read_text()
                lines = content.splitlines()
                
                for i, line in enumerate(lines):
                    if "Mass" in line and i > 0:
                        # Previous line should contain the mass value
                        try:
                            params['mass'] = float(lines[i-1].split()[0])
                        except:
                            pass
                    elif "Moments of Inertia" in line and i > 0:
                        try:
                            moi = lines[i-1].split()[:3]
                            params['moi_xx'] = float(moi[0])
                            params['moi_yy'] = float(moi[1]) if len(moi) > 1 else None
                            params['moi_zz'] = float(moi[2]) if len(moi) > 2 else None
                        except:
                            pass
            
            # Extract from Orb_LEO.txt
            orb_file = config_dir / "Orb_LEO.txt"
            if orb_file.exists():
                content = orb_file.read_text()
                lines = content.splitlines()
                
                for i, line in enumerate(lines):
                    if "Periapsis & Apoapsis Altitude" in line and i > 0:
                        try:
                            alts = lines[i-1].split()[:2]
                            params['periapsis_alt_km'] = float(alts[0])
                            params['apoapsis_alt_km'] = float(alts[1]) if len(alts) > 1 else None
                        except:
                            pass
                    elif "Inclination" in line and i > 0:
                        try:
                            params['inclination_deg'] = float(lines[i-1].split()[0])
                        except:
                            pass
                            
        except Exception as e:
            self.logger.warning(f"Error extracting parameters: {e}")
            
        return params
    
    def create_metadata(self, config_id: str, config_dir: Path, 
                       start_time: datetime, end_time: datetime) -> Dict:
        """Create metadata for the simulation run"""
        params = self.extract_parameters_from_config(config_dir)
        
        metadata = {
            'config_id': config_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'wait_time_configured': self.wait_time,
            'nos3_root': str(self.nos3_root),
            'container_name': self.container_name,
            'container_42_path': self.container_42_path,
            'parameters': params,
            'timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def run_nos3_simulation(self, config_id: str, config_dir: Path) -> bool:
        """Run NOS3 simulation for a specific configuration"""
        try:
            self.logger.info(f"Starting NOS3 simulation for {config_id}")
            
            # Create output directory for this config
            config_output_dir = self.data_output_dir / config_id
            config_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Save configuration files to output
            if not self.save_config_to_output(config_dir, config_output_dir):
                self.logger.warning("Failed to save config files, but continuing...")
            
            # Record start time
            start_time = datetime.now()
            
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
                self.stop_simulation()
                return False
            
            # Wait for simulation to run
            self.logger.info(f"Simulation launched. Running for {self.wait_time} seconds...")
            self.simulation_running = True
            
            # Precise timing
            sim_start = time.time()
            while time.time() - sim_start < self.wait_time:
                if not self.simulation_running:  # Check for interruption
                    break
                time.sleep(0.5)
            
            elapsed = time.time() - sim_start
            self.logger.info(f"Simulation ran for {elapsed:.2f} seconds")
            
            # CRITICAL: Copy 42 data BEFORE stopping simulation
            self.logger.info("Copying 42 simulator data...")
            copy_success = self.atomic_copy_42_data(config_id, config_output_dir)
            
            if not copy_success:
                self.logger.error("Failed to copy 42 data!")
            
            # Record end time
            end_time = datetime.now()
            
            # Now stop simulation
            self.logger.info("Stopping NOS3 simulation...")
            stop_success = self.stop_simulation()
            
            if not stop_success:
                self.logger.warning("Issues stopping simulation, but data was collected")
            
            # Save metadata
            metadata = self.create_metadata(config_id, config_dir, start_time, end_time)
            metadata_file = config_output_dir / "metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            self.simulation_running = False
            return copy_success  # Success based on data collection
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {e}")
            try:
                self.stop_simulation()
            except:
                pass
            self.simulation_running = False
            return False
    
    def monitor_launch_completion(self, process: subprocess.Popen) -> bool:
        """Monitor make launch output for completion"""
        self.logger.info("Monitoring launch process...")
        
        completion_marker = "Docker launch script completed!"
        timeout_seconds = 300  # 5 minutes
        start_time = time.time()
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line_stripped = line.strip()
                    # Log key lines
                    if any(keyword in line_stripped.lower() 
                          for keyword in ['error', 'failed', 'docker', 'completed']):
                        self.logger.debug(f"Launch: {line_stripped}")
                    
                    if completion_marker in line:
                        self.logger.info("Launch completion detected!")
                        return True
                
                if time.time() - start_time > timeout_seconds:
                    self.logger.error(f"Launch timeout after {timeout_seconds} seconds")
                    process.terminate()
                    return False
                
                if process.poll() is not None:
                    break
            
            return_code = process.wait()
            if return_code == 0:
                self.logger.info("Launch process completed")
                return True
            else:
                self.logger.error(f"Launch failed with code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error monitoring launch: {e}")
            return False
    
    def stop_simulation(self) -> bool:
        """Stop the NOS3 simulation"""
        try:
            self.logger.info("Executing 'make stop' command...")
            
            stop_process = subprocess.Popen(
                ['make', 'stop'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.nos3_root
            )
            
            timeout_seconds = 120
            start_time = time.time()
            
            while stop_process.poll() is None:
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning("Stop timeout, forcing termination")
                    stop_process.terminate()
                    time.sleep(5)
                    if stop_process.poll() is None:
                        stop_process.kill()
                    break
                time.sleep(1)
            
            return_code = stop_process.wait()
            
            if return_code == 0:
                self.logger.info("Simulation stopped successfully")
                return True
            else:
                self.logger.warning(f"Stop returned code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return False
    
    def run_experiment_sequence(self) -> None:
        """Run the complete sequence of experiments"""
        try:
            # Find all configuration directories
            config_dirs = self.find_config_directories()
            
            if not config_dirs:
                self.logger.error("No configuration directories found!")
                return
            
            total_configs = len(config_dirs)
            self.logger.info(f"Starting experiment sequence with {total_configs} configurations")
            
            successful = 0
            failed = 0
            
            for i, config_dir in enumerate(config_dirs, 1):
                config_id = config_dir.name
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"EXPERIMENT {i}/{total_configs}: {config_id}")
                self.logger.info(f"{'='*80}")
                
                # Validate configuration
                if not self.validate_config_directory(config_dir):
                    self.logger.error(f"Skipping {config_id} due to missing files")
                    failed += 1
                    continue
                
                # Copy config to NOS3
                if not self.copy_config_files_to_nos3(config_dir):
                    self.logger.error(f"Failed to copy config files for {config_id}")
                    failed += 1
                    continue
                
                # Run simulation and collect data
                sim_success = self.run_nos3_simulation(config_id, config_dir)
                
                if sim_success:
                    successful += 1
                    self.logger.info(f" Experiment {config_id} completed successfully")
                else:
                    failed += 1
                    self.logger.error(f" Experiment {config_id} failed")
                
                # Brief pause between experiments
                if i < total_configs:
                    self.logger.info("Pausing 5 seconds before next experiment...")
                    time.sleep(5)
            
            # Final summary
            self.logger.info(f"\n{'='*80}")
            self.logger.info("EXPERIMENT SEQUENCE COMPLETED")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Total experiments: {total_configs}")
            self.logger.info(f"Successful: {successful}")
            self.logger.info(f"Failed: {failed}")
            
            if total_configs > 0:
                self.logger.info(f"Success rate: {successful/total_configs*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Fatal error in experiment sequence: {e}")
            raise
    
    def cleanup_and_exit(self, signal_num, frame):
        """Handle graceful shutdown on signal"""
        self.logger.info(f"\nReceived signal {signal_num}. Initiating graceful shutdown...")
        
        if self.simulation_running:
            self.logger.info("Stopping current simulation...")
            self.simulation_running = False
            self.stop_simulation()
        
        self.logger.info("Cleanup completed. Exiting...")
        sys.exit(0)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run cFS/NOS3 simulations and collect 42 simulator data'
    )
    
    # Path arguments with defaults
    parser.add_argument('--param-dir', 
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/parameter_generation/parameter_output/parameter_sets',
                       help='Path to parameter_sets directory')
    
    parser.add_argument('--nos3-root',
                       default='<RVSPEC_ROOT>/UAV/nos3',
                       help='Path to NOS3 root directory')
    
    parser.add_argument('--output-dir',
                       default='<DATA_DIR>',
                       help='Path to output directory for 42 data')
    
    # Simulation parameters
    parser.add_argument('--wait-time', type=int, default=120,
                       help='Simulation run time in seconds (default: 120)')
    
    parser.add_argument('--start-config', type=int, default=0,
                       help='Start from specific config number (default: 0)')
    
    parser.add_argument('--end-config', type=int,
                       help='End at specific config number (optional)')
    
    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configs without running simulations')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = CFSSimulationRunner(
            parameter_dir=args.param_dir,
            nos3_root=args.nos3_root,
            data_output_dir=args.output_dir,
            wait_time=args.wait_time,
            start_config=args.start_config,
            end_config=args.end_config
        )
        
        if args.verbose:
            runner.logger.setLevel(logging.DEBUG)
        
        # Validate paths
        required_paths = [
            (Path(args.param_dir), "Parameter directory"),
            (Path(args.nos3_root), "NOS3 root directory")
        ]
        
        for path, description in required_paths:
            if not path.exists():
                runner.logger.error(f"{description} not found: {path}")
                sys.exit(1)
        
        if args.dry_run:
            runner.logger.info("DRY RUN MODE: Validating configurations...")
            config_dirs = runner.find_config_directories()
            valid = 0
            for config_dir in config_dirs:
                if runner.validate_config_directory(config_dir):
                    valid += 1
                    runner.logger.info(f" {config_dir.name} is valid")
                else:
                    runner.logger.error(f" {config_dir.name} is invalid")
            runner.logger.info(f"Validation complete: {valid}/{len(config_dirs)} valid")
        else:
            # Run experiments
            runner.run_experiment_sequence()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()