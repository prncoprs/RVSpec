import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

# Define paths
bin_directory = "../../Data/DefaultNoise"  # Directory containing .BIN files
output_directory = "./data"  # Output directory
mavtypes = ["ESC", "XKF5", "PSCD", "DSF", "ARM", "FMT", "MODE", "RCI2", "XKF3", "RCO2", "XKY0",
            "MSG", "EV", "MOTB", "VER", "VIBE", "MAG", "GPS", "MAV", "XKV1", "BARO", "AHR2",
            "DU32", "MAVC", "BAT", "TERR", "XKQ", "GPA", "SIM", "EAHR", "XKY1", "PM", "MULT",
            "IMU", "PSCN", "XKF4", "RCIN", "ATT", "POS", "CTRL", "RATE", "UNIT", "FILE", "FMTU",
            "XKV2", "PARM", "XKFS", "XKTV", "SRTL", "CMD", "XKF2", "RCOU", "PSCE", "SIM2",
            "DCM", "XKT", "ORGN", "XKF1", "CTUN"]

# Define the new BIN files with SIM_WIND_SPD = 0
bin_files = ["00000001.BIN", "00000002.BIN", "00000003.BIN"]

# Expected environmental factor
env_factor = "SIM_WIND_SPD"
env_value = "0"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

def extract_mav_type(bin_file, mavtype, bin_output_dir):
    """Extract MAVLink data from a .BIN file and save it as a CSV."""
    bin_path = os.path.join(bin_directory, bin_file)
    output_csv = os.path.join(bin_output_dir, f"{mavtype}.csv")

    # Run mavlogdump.py to extract logs
    command = f"mavlogdump.py --types {mavtype} --format csv {bin_path} > {output_csv}"

    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted {mavtype} from {bin_file} -> {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract {mavtype} from {bin_file}: {e}")

def process_bin_file(bin_file):
    """Processes a .BIN file: extracts MAVLink data and stores it properly."""
    bin_basename = os.path.splitext(bin_file)[0]  # Extract filename without extension
    env_folder = os.path.join(output_directory, f"{env_factor}-{env_value}")  # Create env factor folder
    bin_output_dir = os.path.join(env_folder, f"{bin_basename}-{env_factor}-{env_value}")

    # Create directories
    os.makedirs(bin_output_dir, exist_ok=True)

    # Extract all MAV types using parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(extract_mav_type, bin_file, mavtype, bin_output_dir) for mavtype in mavtypes]
        for future in futures:
            future.result()  # Ensure all tasks are completed

    print(f"Extraction complete for {bin_file}, stored in {bin_output_dir}")

# Process each BIN file
for bin_file in bin_files:
    process_bin_file(bin_file)

print("All BIN files extracted successfully.")

