"""
Drone Simulation Data Preprocessing Module
==========================================

This module handles all data extraction, processing, and feature computation
from raw drone simulation logs. It produces clean, filtered datasets ready
for machine learning analysis.

"""

import pandas as pd
import numpy as np
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from geopy.distance import geodesic


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Environmental factors used in simulations
ENV_FACTORS = [
    "SIM_ACC1_RND", "SIM_ACC2_RND",
    "SIM_BAR2_DRIFT", "SIM_BAR2_GLITCH", "SIM_BAR2_RND",
    "SIM_BARO_DRIFT", "SIM_BARO_GLITCH", "SIM_BARO_RND",
    "SIM_BARO_WCF_BAK", "SIM_BARO_WCF_DN", "SIM_BARO_WCF_FWD", 
    "SIM_BARO_WCF_LFT", "SIM_BARO_WCF_RGT", "SIM_BARO_WCF_UP",
    "SIM_GPS_DRIFTALT", "SIM_GPS_GLITCH_X", "SIM_GPS_GLITCH_Y", 
    "SIM_GPS_GLITCH_Z", "SIM_GPS_NOISE",
    "SIM_GPS2_DRFTALT", "SIM_GPS2_GLTCH_X", "SIM_GPS2_GLTCH_Y", 
    "SIM_GPS2_GLTCH_Z", "SIM_GPS2_NOISE",
    "SIM_GYR1_RND", "SIM_GYR2_RND",
    "SIM_MAG_RND",
    "SIM_TEMP_BFACTOR", "SIM_TEMP_BRD_OFF", "SIM_TEMP_START", "SIM_TEMP_TCONST",
    "SIM_WIND_DIR", "SIM_WIND_DIR_Z", "SIM_WIND_SPD", "SIM_WIND_T", 
    "SIM_WIND_T_ALT", "SIM_WIND_T_COEF", "SIM_WIND_TURB"
]

# CSV files to include in analysis
SELECTED_FILES = [
    'AHR2', 'ATT', 'BARO', 'BAT', 'CTUN', 'DCM', 'ESC', 'GPA', 'GPS', 
    'IMU', 'MAG', 'PM', 'POS', 'PSCD', 'PSCE', 'PSCN', 'RATE', 'SIM', 
    'SIM2', 'SRTL', 'TERR', 'VIBE', 'XKF4'
]

# Flight mode mapping for ArduCopter
MODE_MAPPING_ACM = {
    0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO', 4: 'GUIDED',
    5: 'LOITER', 6: 'RTL', 7: 'CIRCLE', 8: 'POSITION', 9: 'LAND',
    10: 'OF_LOITER', 11: 'DRIFT', 13: 'SPORT', 14: 'FLIP', 15: 'AUTOTUNE',
    16: 'POSHOLD', 17: 'BRAKE', 18: 'THROW', 19: 'AVOID_ADSB',
    20: 'GUIDED_NOGPS', 21: 'SMART_RTL', 22: 'FLOWHOLD', 23: 'FOLLOW',
    24: 'ZIGZAG', 25: 'SYSTEMID', 26: 'AUTOROTATE', 27: 'AUTO_RTL',
}

# Replication column identifier
REPLICATION_COLUMN = "Rep"

# Simulator configurations
SIMULATOR_CONFIGS = {
    "SITL": {
        "data_root": "./data_sitl",
        "analysis_output": "./analysis_output_sitl",
        "display_name": "SITL Simulator"
    },
    "Gazebo": {
        "data_root": "./data_gazebo",
        "analysis_output": "./analysis_output_gazebo",
        "display_name": "Gazebo Simulator"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_directory(output_dir):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance (meters) between two GPS coordinates."""
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def get_timeus_bounds(bin_folder_path):
    """
    Find the minimum and maximum TimeUS values from all CSV files in the bin folder.
    
    Args:
        bin_folder_path: Path to a single simulation's bin folder
        
    Returns:
        tuple: (min_timeus, max_timeus) or (None, None) if not found
    """
    min_timeus = float('inf')
    max_timeus = float('-inf')
    found_any = False

    for filename in os.listdir(bin_folder_path):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(bin_folder_path, filename)

        try:
            df = pd.read_csv(file_path, usecols=["TimeUS"], float_precision='round_trip')
            if not df.empty:
                file_min = df["TimeUS"].min()
                file_max = df["TimeUS"].max()
                if pd.notna(file_min) and pd.notna(file_max):
                    min_timeus = min(min_timeus, file_min)
                    max_timeus = max(max_timeus, file_max)
                    found_any = True
        except (ValueError, KeyError):
            continue
        except Exception as e:
            print(f"[Warning] Failed to read {file_path}: {e}")
            continue

    return (min_timeus, max_timeus) if found_any else (None, None)


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def extract_env_factors(parm_csv_path):
    """
    Extract environmental factors from a PARM.csv file.
    Uses the last row's Value if multiple entries exist for a factor.
    
    Args:
        parm_csv_path: Path to the parameter CSV file
        
    Returns:
        dict: {env_factor: value}, with None for missing ones
    """
    if not os.path.exists(parm_csv_path):
        print(f"[Error] File not found: {parm_csv_path}")
        return None
        
    env_factors = dict.fromkeys(ENV_FACTORS, None)
    
    try:
        df_parm = pd.read_csv(parm_csv_path, float_precision='round_trip')
        if "Name" not in df_parm.columns or "Value" not in df_parm.columns:
            print(f"[Error] Missing required columns in {parm_csv_path}")
            return None
            
        for factor in ENV_FACTORS:
            rows = df_parm[df_parm["Name"] == factor]
            if not rows.empty:
                try:
                    env_factors[factor] = float(rows.iloc[-1]["Value"])
                except Exception as e:
                    print(f"[Warning] Could not parse value for {factor} in {parm_csv_path}: {e}")
        return env_factors
    except Exception as e:
        print(f"[Error] Failed to read {parm_csv_path}: {e}")
        return None


def get_hit_ground_speed(msg_csv_path):
    """Extract Hit Ground Speed from MSG.csv."""
    if not os.path.exists(msg_csv_path):
        print(f"Missing MSG.csv: {msg_csv_path}")
        return None
        
    try:
        df_msg = pd.read_csv(msg_csv_path, float_precision='round_trip')
        for _, row in df_msg.iterrows():
            message = str(row.get("Message", ""))
            if "SIM Hit ground at" in message:
                parts = message.split(" ")
                return float(parts[-2])
    except Exception as e:
        print(f"Error reading {msg_csv_path}: {e}")
    return None


def get_whole_mission_duration(msg_csv_path):
    """Extract whole mission duration from MSG.csv using first and last TimeUS."""
    if not os.path.exists(msg_csv_path):
        print(f"Missing MSG.csv: {msg_csv_path}")
        return None
        
    try:
        df_msg = pd.read_csv(msg_csv_path, float_precision='round_trip')
        if "TimeUS" not in df_msg.columns or df_msg.empty:
            print(f"Invalid MSG.csv format or empty file: {msg_csv_path}")
            return None
        start_time = df_msg["TimeUS"].iloc[0]
        end_time = df_msg["TimeUS"].iloc[-1]
        return end_time - start_time
    except Exception as e:
        print(f"Error reading MSG.csv: {e}")
        return None


def extract_flight_mode_phases(mode_csv_path, bin_folder_path):
    """
    Extract mission phases based on Mode changes using actual min/max TimeUS across the bin folder.
    
    Args:
        mode_csv_path: Path to MODE.csv file
        bin_folder_path: Path to the bin folder containing MODE.csv and other logs
        
    Returns:
        list: List of tuples (start_TimeUS, end_TimeUS, phase_idx)
    """
    phase_ranges = []
    min_time, max_time = get_timeus_bounds(bin_folder_path)
    
    if min_time is None or max_time is None:
        print(f"[Error] Cannot determine time bounds from {bin_folder_path}")
        return phase_ranges
        
    try:
        df_mode = pd.read_csv(mode_csv_path, float_precision='round_trip')
        df_mode = df_mode.sort_values("TimeUS").reset_index(drop=True)
        
        if df_mode.empty or "ModeNum" not in df_mode.columns or "TimeUS" not in df_mode.columns:
            print(f"[Error] Invalid or empty MODE.csv: {mode_csv_path}")
            return phase_ranges
            
        # Phase 0: From min_time to first mode change
        first_mode_time = df_mode.loc[0, "TimeUS"]
        phase_ranges.append((min_time, first_mode_time, 0))
        
        # Intermediate phases: between mode switches
        phase_idx = 1
        for i in range(1, len(df_mode)):
            start_time = df_mode.loc[i - 1, "TimeUS"]
            end_time = df_mode.loc[i, "TimeUS"]
            phase_ranges.append((start_time, end_time, phase_idx))
            phase_idx += 1
            
        # Final phase: from last mode change to end of log
        last_mode_time = df_mode["TimeUS"].iloc[-1]
        phase_ranges.append((last_mode_time, max_time, phase_idx))
        
    except Exception as e:
        print(f"[Error] Failed to read {mode_csv_path}: {e}")
        
    return phase_ranges


# ============================================================================
# STATISTICAL COMPUTATION FUNCTIONS
# ============================================================================

def compute_phase_durations(mission_phases):
    """Compute the duration for each mission phase."""
    phase_durations = {}
    
    for start_time, end_time, phase_num in mission_phases:
        duration = end_time - start_time
        phase_durations[f"Mission_Duration_Phase{phase_num}"] = duration
    
    return phase_durations


def compute_whole_mission_stats(df, csv_file, folder_data):
    """Compute mission-wide statistics."""
    df_numeric = df.select_dtypes(include=[np.number]).iloc[:, 2:] # Remove timestamp and TimeUS columns
    for col in df_numeric.columns:
        folder_data[f"{csv_file}_{col}_mean"] = df_numeric[col].mean()
        folder_data[f"{csv_file}_{col}_min"] = df_numeric[col].min()
        folder_data[f"{csv_file}_{col}_max"] = df_numeric[col].max()
        folder_data[f"{csv_file}_{col}_std"] = df_numeric[col].std()
        folder_data[f"{csv_file}_{col}_median"] = df_numeric[col].median()


def compute_per_phase_stats(df, csv_file, folder_data, mission_phases):
    """Compute per-phase statistics for each mission phase."""
    for start, end, phase_num in mission_phases:
        df_phase = df[(df["TimeUS"] >= start) & (df["TimeUS"] < end)]
        if df_phase.empty:
            continue

        df_phase_numeric = df_phase.select_dtypes(include=[np.number]).iloc[:, 2:]
        for col in df_phase_numeric.columns:
            folder_data[f"{csv_file}_{col}_mean_Phase{phase_num}"] = df_phase_numeric[col].mean()
            folder_data[f"{csv_file}_{col}_min_Phase{phase_num}"] = df_phase_numeric[col].min()
            folder_data[f"{csv_file}_{col}_max_Phase{phase_num}"] = df_phase_numeric[col].max()
            folder_data[f"{csv_file}_{col}_std_Phase{phase_num}"] = df_phase_numeric[col].std()
            folder_data[f"{csv_file}_{col}_median_Phase{phase_num}"] = df_phase_numeric[col].median()


def compute_vibration_stats(folder_path, folder_data, mission_phases):
    """Compute vibration statistics for each IMU in VIBE.csv."""
    vibe_csv_path = os.path.join(folder_path, "VIBE.csv")
    if not os.path.exists(vibe_csv_path):
        return
    
    df_vibe = pd.read_csv(vibe_csv_path)

    # Ensure required columns exist
    required_columns = {"IMU", "VibeX", "VibeY", "VibeZ", "TimeUS"}
    if not required_columns.issubset(df_vibe.columns):
        return
    
    # Compute total vibration for each IMU
    df_vibe["Vibration"] = df_vibe["VibeX"] + df_vibe["VibeY"] + df_vibe["VibeZ"]
    
    # Ensure IMUs are processed in numerical order (IMU0, IMU1, ...)
    imu_list = sorted(df_vibe["IMU"].unique())
    
    # Compute whole mission vibration stats for each IMU
    for imu in imu_list:
        df_imu = df_vibe[df_vibe["IMU"] == imu]

        folder_data[f"IMU{imu}_Vibration_mean"] = df_imu["Vibration"].mean()
        folder_data[f"IMU{imu}_Vibration_max"] = df_imu["Vibration"].max()
        folder_data[f"IMU{imu}_Vibration_min"] = df_imu["Vibration"].min()
        folder_data[f"IMU{imu}_Vibration_std"] = df_imu["Vibration"].std()
        folder_data[f"IMU{imu}_Vibration_median"] = df_imu["Vibration"].median()

        # Compute per-phase vibration stats for each IMU
        for start, end, phase_num in mission_phases:
            df_phase = df_vibe[(df_vibe["TimeUS"] >= start) & (df_vibe["TimeUS"] < end)]
            if df_phase.empty:
                continue

            df_imu_phase = df_phase[df_phase["IMU"] == imu]
            if df_imu_phase.empty:
                continue

            folder_data[f"IMU{imu}_Vibration_mean_Phase{phase_num}"] = df_imu_phase["Vibration"].mean()
            folder_data[f"IMU{imu}_Vibration_max_Phase{phase_num}"] = df_imu_phase["Vibration"].max()
            folder_data[f"IMU{imu}_Vibration_min_Phase{phase_num}"] = df_imu_phase["Vibration"].min()
            folder_data[f"IMU{imu}_Vibration_std_Phase{phase_num}"] = df_imu_phase["Vibration"].std()
            folder_data[f"IMU{imu}_Vibration_median_Phase{phase_num}"] = df_imu_phase["Vibration"].median()


def compute_battery_stats(folder_path, folder_data, mission_phases):
    """Compute battery consumption using CurrTot from BAT.csv."""
    bat_path = os.path.join(folder_path, "BAT.csv")
    if not os.path.exists(bat_path):
        return
    
    df_bat = pd.read_csv(bat_path, float_precision='round_trip')
    if "CurrTot" not in df_bat.columns:
        return

    # Whole mission battery consumption
    folder_data["Battery_Consumption"] = df_bat["CurrTot"].iloc[-1] - df_bat["CurrTot"].iloc[0]

    # Phase-wise battery consumption
    for start, end, phase_num in mission_phases:
        df_phase = df_bat[(df_bat["TimeUS"] >= start) & (df_bat["TimeUS"] < end)]
        if df_phase.empty:
            continue

        folder_data[f"Battery_Consumption_Phase{phase_num}"] = df_phase["CurrTot"].iloc[-1] - df_phase["CurrTot"].iloc[0]


def compute_rpy_error(folder_path, folder_data, mission_phases):
    """
    Compute the Roll-Pitch-Yaw (RPY) error using ATT.csv.
    RPY Error = |DesRoll - Roll| + |DesPitch - Pitch| + |DesYaw - Yaw|
    """
    att_path = os.path.join(folder_path, "ATT.csv")
    if not os.path.exists(att_path):
        return

    # Read ATT.csv with full precision
    df_att = pd.read_csv(att_path, float_precision='round_trip')

    # Ensure necessary columns exist
    required_cols = {"DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw", "TimeUS"}
    if not required_cols.issubset(df_att.columns):
        return

    # Compute individual Roll, Pitch, and Yaw errors
    df_att["Roll_Error"] = np.abs(df_att["DesRoll"] - df_att["Roll"])
    df_att["Pitch_Error"] = np.abs(df_att["DesPitch"] - df_att["Pitch"])
    df_att["Yaw_Error"] = np.abs(df_att["DesYaw"] - df_att["Yaw"])

    # Compute total RPY error
    df_att["RPY_Error"] = df_att["Roll_Error"] + df_att["Pitch_Error"] + df_att["Yaw_Error"]

    # Store whole mission RPY errors
    folder_data["Roll_Error"] = df_att["Roll_Error"].mean()
    folder_data["Pitch_Error"] = df_att["Pitch_Error"].mean()
    folder_data["Yaw_Error"] = df_att["Yaw_Error"].mean()
    folder_data["RPY_Error"] = df_att["RPY_Error"].mean()

    # Compute errors for each mission phase
    for start, end, phase_num in mission_phases:
        df_phase = df_att[(df_att["TimeUS"] >= start) & (df_att["TimeUS"] < end)]
        if df_phase.empty:
            continue

        folder_data[f"Roll_Error_Phase{phase_num}"] = df_phase["Roll_Error"].mean()
        folder_data[f"Pitch_Error_Phase{phase_num}"] = df_phase["Pitch_Error"].mean()
        folder_data[f"Yaw_Error_Phase{phase_num}"] = df_phase["Yaw_Error"].mean()
        folder_data[f"RPY_Error_Phase{phase_num}"] = df_phase["RPY_Error"].mean()


def compute_brake_time(bin_folder, folder_data):
    """Compute brake time from BRAKE mode initiation to full stop."""
    sim2_path = os.path.join(bin_folder, "SIM2.csv")
    mode_path = os.path.join(bin_folder, "MODE.csv")
    
    if not (os.path.exists(sim2_path) and os.path.exists(mode_path)):
        return
    
    try:
        mode_df = pd.read_csv(mode_path, float_precision='round_trip')
        sim2_df = pd.read_csv(sim2_path, float_precision='round_trip')
        
        # Compute 3D speed from velocity components
        sim2_df["speed"] = np.sqrt(sim2_df["VN"]**2 + sim2_df["VE"]**2 + sim2_df["VD"]**2)
        
        # Find BRAKE mode (ModeNum == 17)
        brake_rows = mode_df[mode_df["ModeNum"] == 17].reset_index()
        if brake_rows.empty:
            return  # No BRAKE mode found
        
        start_time = brake_rows.loc[0, "TimeUS"]
        
        # Extract data after BRAKE begins
        post_brake = sim2_df[sim2_df["TimeUS"] >= start_time].copy()
        if post_brake.empty:
            return
        
        t_start = post_brake["TimeUS"].iloc[0] * 1e-6
        STOP_THRESHOLD = 0.1  # m/s
        
        # Find first point when speed < STOP_THRESHOLD
        stopped = post_brake[post_brake["speed"] < STOP_THRESHOLD]
        if stopped.empty:
            return
        
        t_stop = stopped["TimeUS"].iloc[0] * 1e-6
        brake_time = t_stop - t_start
        folder_data["Brake_Time"] = brake_time
        
    except Exception as e:
        print(f"Error computing brake time for {bin_folder}: {e}")


# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================

def process_bin_folder(bin_folder):
    """
    Process a single bin folder and extract all relevant features.
    
    Args:
        bin_folder: Path to the bin folder containing CSV files
        
    Returns:
        dict: Dictionary containing all extracted features, or None if processing failed
    """
    if not os.path.isdir(bin_folder):
        return None

    msg_path = os.path.join(bin_folder, "MSG.csv")
    mode_path = os.path.join(bin_folder, "MODE.csv")
    parm_path = os.path.join(bin_folder, "PARM.csv")

    if not os.path.exists(parm_path):
        return None

    # Extract environmental factors
    env_factors = extract_env_factors(parm_path)
    if env_factors is None or all(v is None for v in env_factors.values()):
        return None

    # Extract mission phases and basic metrics
    mission_phases = extract_flight_mode_phases(mode_path, bin_folder)
    hit_ground_speed = get_hit_ground_speed(msg_path)
    whole_mission_duration = get_whole_mission_duration(msg_path)

    # Initialize folder data with basic metrics
    folder_data = {
        **env_factors,
        "Hit_Ground_Speed": hit_ground_speed,
        "Mission_Duration": whole_mission_duration
    }

    # Compute phase durations and specialized statistics
    folder_data.update(compute_phase_durations(mission_phases))
    compute_battery_stats(bin_folder, folder_data, mission_phases)
    compute_rpy_error(bin_folder, folder_data, mission_phases)
    compute_vibration_stats(bin_folder, folder_data, mission_phases)
    compute_brake_time(bin_folder, folder_data)

    # Process each selected CSV file - NO OUTPUT during parallel processing
    for csv_file in SELECTED_FILES:
        csv_path = os.path.join(bin_folder, csv_file + ".csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path, float_precision='round_trip')
            if "TimeUS" not in df.columns:
                continue

            compute_whole_mission_stats(df, csv_file, folder_data)
            compute_per_phase_stats(df, csv_file, folder_data, mission_phases)

        except Exception as e:
            # Store errors to report later, don't print during parallel execution
            if 'errors' not in folder_data:
                folder_data['errors'] = []
            folder_data['errors'].append(f"Error in {csv_file}: {str(e)}")
            
    return folder_data


def process_all_data(data_root, analysis_output):
    """
    Process all bin folders and merge the results for a specific data directory.
    
    Args:
        data_root: Root directory containing the simulation data
        analysis_output: Output directory for results
        
    Returns:
        pd.DataFrame: Merged dataframe with all processed data
    """
    all_data = []

    if not os.path.exists(data_root):
        print(f"Error: Data root directory does not exist: {data_root}")
        return None

    print(f"Processing data from: {data_root}")
    
    # Count total folders for progress tracking
    folders = [f for f in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, f))]
    total_folders = len(folders)
    print(f"Found {total_folders} simulation folders to process")
    
    # Process all folders in parallel with clean progress output
    with ProcessPoolExecutor() as executor:
        # Submit all jobs
        future_to_folder = {}
        for folder in folders:
            full_path = os.path.join(data_root, folder)
            future = executor.submit(process_bin_folder, full_path)
            future_to_folder[future] = folder

        # Process completed jobs with single progress line
        processed_count = 0
        error_count = 0
        
        for future in as_completed(future_to_folder):
            folder_name = future_to_folder[future]
            result = future.result()
            
            if result:
                # Check for and report errors without cluttering output
                if 'errors' in result:
                    error_count += len(result['errors'])
                    # Remove errors from data before adding to results
                    del result['errors']
                all_data.append(result)
            
            processed_count += 1
            
            # Single clean progress line that overwrites itself
            progress_pct = (processed_count / total_folders) * 100
            success_count = len(all_data)
            status = f"Progress: {processed_count}/{total_folders} folders ({progress_pct:.1f}%) | "
            status += f"Successful: {success_count} | "
            if error_count > 0:
                status += f"Errors: {error_count} | "
            status += f"Current: {folder_name}"
            
            # Ensure the line is properly padded and overwrites completely
            print(f"\r{status:<120}", end='', flush=True)
    
    # Clear progress line with success message
    success_rate = (len(all_data) / total_folders) * 100 if total_folders > 0 else 0
    final_msg = f" Processing complete: {len(all_data)}/{total_folders} folders successful ({success_rate:.1f}%)"
    if error_count > 0:
        final_msg += f" | {error_count} processing errors encountered"
    print(f"\r{final_msg:<120}")

    # Merge and organize results
    if all_data:
        print("Merging and organizing data...")
        df_merged = pd.DataFrame(all_data)

        # Sort by all ENV_FACTORS
        print("Sorting data...")
        df_merged = df_merged.sort_values(by=ENV_FACTORS)

        # Group by ENV_FACTORS and assign rep1, rep2, ...
        print("Assigning replication labels...")
        df_merged["Rep"] = (
            df_merged.groupby(ENV_FACTORS)
            .cumcount() + 1
        ).apply(lambda x: f"rep{x}")

        # Reorder columns: ENV_FACTORS + ["Rep"] + rest
        print("Reorganizing columns...")
        all_columns = list(df_merged.columns)
        other_columns = [col for col in all_columns if col not in ENV_FACTORS and col != "Rep"]
        ordered_columns = ENV_FACTORS + ["Rep"] + other_columns
        df_merged = df_merged[ordered_columns]

        # Save to CSV
        ensure_output_directory(analysis_output)
        output_path = os.path.join(analysis_output, "merged_data.csv")
        print(f"Saving merged data to {output_path}...")
        df_merged.to_csv(output_path, index=False)
        print(f" Merged data saved ({df_merged.shape[0]} rows, {df_merged.shape[1]} columns)")
        
        return df_merged
    else:
        print(" No valid data was processed.")
        return None


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_dataframe(df):
    """
    Preprocess the dataframe by:
    - Ensuring environmental factor columns exist
    - Keeping environmental factors even if they have NaN values
    - Dropping non-environmental columns that are entirely NaN
    - Removing near-constant (low-variance) columns
    - Averaging replicated experiments if REPLICATION_COLUMN is present
    """
    df_cleaned = df.copy()

    # Ensure environmental factor columns exist
    for col in ENV_FACTORS:
        if col not in df_cleaned.columns:
            df_cleaned[col] = np.nan

    # Handle replicated experiments if 'Rep' column is present
    if REPLICATION_COLUMN in df_cleaned.columns:
        # Temporarily fill missing env factors with placeholder for grouping
        df_cleaned[ENV_FACTORS] = df_cleaned[ENV_FACTORS].fillna("Unknown")

        # Identify numeric columns to average (excluding env factors and Rep)
        numeric_cols = [
            col for col in df_cleaned.select_dtypes(include=[np.number]).columns
            if col not in ENV_FACTORS and col != REPLICATION_COLUMN
        ]

        # Group by environmental factors and average
        df_grouped = df_cleaned.groupby(ENV_FACTORS, as_index=False)[numeric_cols].mean(numeric_only=True)

        # Restore NaNs in env factors
        df_grouped[ENV_FACTORS] = df_grouped[ENV_FACTORS].replace("Unknown", np.nan)

        df_cleaned = df_grouped.copy(deep=True)

    # Drop non-environmental columns that are fully NaN
    other_cols = [col for col in df_cleaned.columns if col not in ENV_FACTORS]
    drop_cols = [col for col in other_cols if df_cleaned[col].isna().all()]
    df_cleaned = df_cleaned.drop(columns=drop_cols)

    # Drop near-constant (low-variance) columns, excluding ENV_FACTORS
    near_constant_threshold = 1e-6
    variance = df_cleaned.drop(columns=ENV_FACTORS, errors='ignore').var(numeric_only=True)
    low_variance_cols = variance[variance < near_constant_threshold].index.tolist()
    df_cleaned = df_cleaned.drop(columns=low_variance_cols)

    return df_cleaned


def drop_unwanted_columns(df):
    """
    Drop all columns containing specific keywords in their names.
    
    Args:
        df: Input Pandas DataFrame
        
    Returns:
        DataFrame with specified columns removed
    """
    keywords_to_drop = [
        "_max", "_min", "_median", "_Q1", "_Q2", "_Q3", "_Q4", "IMU1_", "BAT_", 
        "AHR2_", "DCM_", "CTUN_", "PSCD_", "PSCE_", "PSCN_", "SRTL_", "CTRL_", 
        "TERR_", "PM_", "IMU0_Vibration_", "RATE_", "GPS_GMS_"
    ]
    columns_to_drop = [col for col in df.columns if any(keyword in col for keyword in keywords_to_drop)]
    return df.drop(columns=columns_to_drop, errors="ignore")


def check_file_exists(filepath, description):
    """Check if a file exists and return the result with a message."""
    if os.path.exists(filepath):
        print(f" Found existing {description}: {filepath}")
        return True
    else:
        print(f" {description} not found: {filepath}")
        return False


def load_or_process_data(data_root, analysis_output):
    """
    Load existing data or process raw data if not available for a specific data directory.
    
    Args:
        data_root: Root directory containing the simulation data
        analysis_output: Output directory for results
        
    Returns:
        pd.DataFrame: Processed and filtered data
    """
    ensure_output_directory(analysis_output)
    
    merged_path = os.path.join(analysis_output, "merged_data.csv")
    filtered_path = os.path.join(analysis_output, "filtered_data.csv")
    
    print(f"\n--- Processing Data from {data_root} ---")
    
    # Check for filtered data first (most processed)
    if check_file_exists(filtered_path, "filtered data"):
        print("Loading existing filtered data...")
        return pd.read_csv(filtered_path)
    
    # Check for merged data
    if check_file_exists(merged_path, "merged data"):
        print("Loading existing merged data and applying filtering...")
        df_merged = pd.read_csv(merged_path)
        
        # Apply preprocessing and filtering with progress
        print("Applying preprocessing filters...")
        df_filtered = preprocess_dataframe(df_merged)
        print("Dropping unwanted columns...")
        df_filtered = drop_unwanted_columns(df_filtered)
        
        # Save filtered data for future use
        print(f"Saving filtered data...")
        df_filtered.to_csv(filtered_path, index=False)
        print(f" Filtered data saved to {filtered_path}")
        return df_filtered
    
    # Process raw data from scratch
    print("Processing raw data from scratch...")
    df_merged = process_all_data(data_root, analysis_output)
    if df_merged is None:
        raise ValueError(f"Failed to process raw data from {data_root}.")
    
    # Apply preprocessing and filtering
    print("Applying preprocessing filters...")
    df_filtered = preprocess_dataframe(df_merged)
    print("Dropping unwanted columns...")
    df_filtered = drop_unwanted_columns(df_filtered)
    
    # Save filtered data
    print(f"Saving filtered data...")
    df_filtered.to_csv(filtered_path, index=False)
    print(f" Filtered data saved to {filtered_path}")
    
    return df_filtered


# ============================================================================
# MAIN EXECUTION FOR PREPROCESSING
# ============================================================================

def main(data_root, output_dir=None):
    """
    Main function to process simulation data and create filtered dataset.
    
    Args:
        data_root: Path to the root directory containing simulation data
        output_dir: Output directory (defaults to analysis_output_<basename>)
        
    Returns:
        pd.DataFrame: Filtered and processed data
    """
    if output_dir is None:
        basename = os.path.basename(data_root.rstrip('/\\'))
        output_dir = f"./analysis_output_{basename}"
    
    print(f"Data Preprocessing Pipeline")
    print(f"Input: {data_root}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    try:
        df_filtered = load_or_process_data(data_root, output_dir)
        
        print(f"\n Preprocessing complete!")
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Environmental factors: {len([col for col in df_filtered.columns if col in ENV_FACTORS])}")
        print(f"Physical states: {len([col for col in df_filtered.columns if col not in ENV_FACTORS])}")
        print(f"Output saved to: {output_dir}")
        
        return df_filtered
        
    except Exception as e:
        print(f" Preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_preprocessor.py <data_root_path> [output_dir]")
        print("Example: python data_preprocessor.py ./data_sitl ./analysis_output_sitl")
        sys.exit(1)
    
    data_root = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    df_filtered = main(data_root, output_dir)
    
    if df_filtered is not None:
        print("\n Sample of processed data:")
        print(df_filtered.head())
        print(f"\nColumns: {list(df_filtered.columns[:10])}...")  # Show first 10 columns