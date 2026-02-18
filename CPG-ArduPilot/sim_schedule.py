#!/usr/bin/env python

import os
import sys
import subprocess
import logging
import time
import shlex

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number of times to run each sim_task.py with the same parameter file
num_runs = 1  # Change this to your desired number of repetitions

def main():
    # Define the directory where your .parm files are stored.
    parm_dir = "./Params"
    if not os.path.exists(parm_dir):
        logger.error(f"Parameter directory '{parm_dir}' does not exist.")
        return

    # List all .parm files in the directory.
    parm_files = [os.path.join(parm_dir, f) for f in os.listdir(parm_dir) if f.endswith(".parm")]
    if not parm_files:
        logger.error("No .parm files found in the parameter directory.")
        return

    # Convert each parameter file to its absolute path and sort.
    parm_files = sorted([os.path.abspath(pf) for pf in parm_files])

    print(parm_files)
    # sys.exit(0)

    # Loop over each .parm file and run sim_task.py multiple times.
    for parm_file in parm_files:
        for i in range(num_runs):
            logger.info(f"=== Running simulation {i+1}/{num_runs} with parameter file: {parm_file} ===")
            
            # Build the command to run sim_task.py.
            cmd = ["python3", "sim_task.py", 
                "--mission", "<RVSPEC_ROOT>/UAV/ardupilot/Tools/autotest/ArduCopter_Tests/CopterMission/copter_mission.txt",
                "--parm_file", parm_file]
            
            print("!!!  Executing command:")
            print(" ".join(shlex.quote(c) for c in cmd))
            print("")
            
            try:
                # Run the simulation as a subprocess.
                result = subprocess.run(cmd, check=True)
                logger.info(f"Run {i+1}/{num_runs} with {parm_file} completed (return code {result.returncode}).")
            except subprocess.CalledProcessError as e:
                logger.error(f"Run {i+1}/{num_runs} with {parm_file} failed: {e}")
            
            # Optional: wait a few seconds before starting the next run.
            time.sleep(5)
    print("------------------------------------------- parm_files -------------------------------------------------")
    print(parm_files)

if __name__ == '__main__':
    main()