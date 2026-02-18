import os
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Define paths
bin_directory = "./raw_data"
output_directory = "./data"
output_type = "gazebo-single"

# MAVLink message types to extract
mavtypes = [
    "ESC", "XKF5", "PSCD", "DSF", "ARM", "FMT", "MODE", "RCI2", "XKF3", "RCO2", "XKY0",
    "MSG", "EV", "MOTB", "VER", "VIBE", "MAG", "GPS", "MAV", "XKV1", "BARO", "AHR2",
    "DU32", "MAVC", "BAT", "TERR", "XKQ", "GPA", "SIM", "EAHR", "XKY1", "PM", "MULT",
    "IMU", "PSCN", "XKF4", "RCIN", "ATT", "POS", "CTRL", "RATE", "UNIT", "FILE", "FMTU",
    "XKV2", "PARM", "XKFS", "XKTV", "SRTL", "CMD", "XKF2", "RCOU", "PSCE", "SIM2",
    "DCM", "XKT", "ORGN", "XKF1", "CTUN"
]

# Get today’s date
today = datetime.now().strftime("%Y%m%d")

# Create base output directory
os.makedirs(output_directory, exist_ok=True)

def extract_all_mavtypes(bin_file):
    """Extracts all mavtypes from a .BIN file and stores them in a dated folder."""
    bin_basename = os.path.splitext(bin_file)[0]
    bin_path = os.path.join(bin_directory, bin_file)

    # Build the output directory name
    out_folder_name = f"{today}-{output_type}-{bin_basename}"
    bin_output_dir = os.path.join(output_directory, out_folder_name)
    os.makedirs(bin_output_dir, exist_ok=True)

    # Extract each MAVLink type into separate CSVs
    for mavtype in mavtypes:
        output_csv = os.path.join(bin_output_dir, f"{mavtype}.csv")
        command = f"mavlogdump.py --types {mavtype} --format csv {bin_path} > {output_csv}"

        try:
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[] {bin_file} -> {mavtype}.csv")
        except subprocess.CalledProcessError:
            print(f"[×] Failed to extract {mavtype} from {bin_file}")

# Process all .BIN files
bin_files = sorted([f for f in os.listdir(bin_directory) if f.endswith(".BIN")])
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    executor.map(extract_all_mavtypes, bin_files)
