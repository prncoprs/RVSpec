"""
PX4 Drone Simulation Data Preprocessing Module
==============================================

This module handles all data extraction, processing, and feature computation
from raw PX4 drone simulation logs. It produces clean, filtered datasets ready
for machine learning analysis.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Environmental factors from parameter_summary.csv (excluding metadata columns)
ENV_FACTORS = [
    "gyroscopeNoiseDensity", "gyroscopeRandomWalk", "gyroscopeTurnOnBiasSigma",
    "accelerometerNoiseDensity", "accelerometerRandomWalk", "accelerometerTurnOnBiasSigma",
    "noiseDensity", "randomWalk", 
    "gpsXYRandomWalk", "gpsZRandomWalk", "gpsXYNoiseDensity", "gpsZNoiseDensity",
    "gpsVXYNoiseDensity", "gpsVZNoiseDensity",
    "windVelocityMean", "windVelocityVariance", 
    "windDirectionMeanX", "windDirectionMeanY", "windDirectionMeanZ", "windDirectionVariance",
    "temperature", "pressure", "magnetic_field_x", "magnetic_field_y", "magnetic_field_z"
]

# Ground truth topics to process (4 topics)
GROUND_TRUTH_TOPICS = [
    'vehicle_local_position_groundtruth',
    'vehicle_attitude_groundtruth', 
    'vehicle_angular_velocity_groundtruth',
    'vehicle_global_position_groundtruth'
]

# Columns to drop for each topic
COLUMNS_TO_DROP = {
    'vehicle_attitude_groundtruth': [
        'delta_q_reset[0]', 'delta_q_reset[1]', 'delta_q_reset[2]', 'delta_q_reset[3]', 
        'quat_reset_counter'
    ],
    'vehicle_angular_velocity_groundtruth': [
        'xyz_derivative[0]', 'xyz_derivative[1]', 'xyz_derivative[2]'
    ],
    'vehicle_global_position_groundtruth': [
        'alt_ellipsoid', 'delta_alt', 'eph', 'epv', 'terrain_alt', 
        'lat_lon_reset_counter', 'alt_reset_counter', 'terrain_alt_valid', 'dead_reckoning'
    ],
    'vehicle_local_position_groundtruth': []  # Will remove constant columns dynamically
}

# PX4 nav_state values
NAV_STATES = {
    'POSCTL': 2,
    'AUTO_TAKEOFF': 17,
    'AUTO_LOITER': 4,
    'AUTO_RTL': 5
}

# Statistics to calculate
STATISTICS = ['mean', 'std', 'max', 'min', 'median']

# PX4 data paths
PX4_CONFIG = {
    "data_root": Path("<DATA_DIR>/PGFuzzVMShared"),
    "parameter_summary": Path("<DATA_DIR>/PGFuzzVMShared"),
    "csv_output": Path("<DATA_DIR>/PGFuzzVMShared"),
    "analysis_output": Path("<DATA_DIR>/PGFuzzVMShared")
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_directory(output_dir):
    """Ensure the output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def extract_experiment_info(folder_name):
    """
    Extract experiment information from folder name.
    Expected format: exp_XXXX_iris_YYYY_windy_ZZZZ_runWW
    
    Returns:
        dict: {exp_id, iris_id, windy_id, run_id} or None if parsing fails
    """
    pattern = r'exp_(\d+)_iris_(\d+)_windy_(\d+)_run(\d+)'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'exp_id': int(match.group(1)),
            'iris_id': int(match.group(2)), 
            'windy_id': int(match.group(3)),
            'run_id': int(match.group(4))
        }
    return None


def convert_ulg_to_csv(ulg_path, csv_output_dir):
    """
    Convert ULG file to CSV files using ulog2csv.
    
    Args:
        ulg_path: Path to the ULG file
        csv_output_dir: Directory to save CSV files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        ensure_output_directory(csv_output_dir)
        
        # Run ulog2csv command
        cmd = ['ulog2csv', str(ulg_path), '-o', str(csv_output_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            print(f"Error converting {ulg_path}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout converting {ulg_path}")
        return False
    except Exception as e:
        print(f"Exception converting {ulg_path}: {e}")
        return False


def get_csv_files_for_topics(csv_dir, topics):
    """
    Get CSV file paths for specified topics.
    
    Args:
        csv_dir: Directory containing CSV files
        topics: List of topic names to find
        
    Returns:
        dict: {topic: csv_path} for found topics
    """
    csv_files = {}
    csv_dir = Path(csv_dir)
    
    if not csv_dir.exists():
        return csv_files
        
    for topic in topics:
        # PX4 CSV files are named like: exp_XXXX_iris_YYYY_windy_ZZZZ_runWW_TIMESTAMP_topic_0.csv
        for csv_file in csv_dir.glob('*.csv'):
            if f"_{topic}_0.csv" in csv_file.name:
                csv_files[topic] = csv_file
                break
                
    return csv_files


def quaternion_to_euler(q0, q1, q2, q3):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        q0, q1, q2, q3: Quaternion components (w, x, y, z)
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


# ============================================================================
# PHASE DETECTION
# ============================================================================

def extract_flight_phases(vehicle_status_csv):
    """
    Extract 4 flight phases from PX4 vehicle_status topic.
    Phases: POSCTL -> AUTO_TAKEOFF -> AUTO_LOITER -> AUTO_RTL
    
    Args:
        vehicle_status_csv: Path to vehicle_status CSV file
        
    Returns:
        list: List of tuples (start_timestamp, end_timestamp, phase_idx) or None if incomplete
    """
    vehicle_status_csv = Path(vehicle_status_csv)
    
    if not vehicle_status_csv.exists():
        return None
    
    try:
        df = pd.read_csv(vehicle_status_csv)
        
        if 'timestamp' not in df.columns or 'nav_state' not in df.columns:
            return None
            
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if df.empty:
            return None
        
        # Drop initial nav_state = 4 (AUTO_LOITER before POSCTL)
        first_posctl_idx = None
        for i, nav_state in enumerate(df['nav_state']):
            if nav_state == NAV_STATES['POSCTL']:
                first_posctl_idx = i
                break
        
        if first_posctl_idx is None:
            return None
            
        # Use data from first POSCTL onwards
        df = df[first_posctl_idx:].reset_index(drop=True)
        
        # Find phase transitions
        phase_transitions = {}
        
        # Phase 0: POSCTL start
        phase_transitions['phase0_start'] = df.iloc[0]['timestamp']
        
        # Phase 1: AUTO_TAKEOFF start
        takeoff_idx = None
        for i, nav_state in enumerate(df['nav_state']):
            if nav_state == NAV_STATES['AUTO_TAKEOFF']:
                takeoff_idx = i
                break
        
        if takeoff_idx is None:
            return None
        phase_transitions['phase1_start'] = df.iloc[takeoff_idx]['timestamp']
        
        # Phase 2: AUTO_LOITER start (after takeoff)
        loiter_idx = None
        for i in range(takeoff_idx + 1, len(df)):
            if df.iloc[i]['nav_state'] == NAV_STATES['AUTO_LOITER']:
                loiter_idx = i
                break
        
        if loiter_idx is None:
            return None
        phase_transitions['phase2_start'] = df.iloc[loiter_idx]['timestamp']
        
        # Phase 3: AUTO_RTL start
        rtl_idx = None
        for i in range(loiter_idx + 1, len(df)):
            if df.iloc[i]['nav_state'] == NAV_STATES['AUTO_RTL']:
                rtl_idx = i
                break
        
        if rtl_idx is None:
            return None
        phase_transitions['phase3_start'] = df.iloc[rtl_idx]['timestamp']
        
        # Phase 3: AUTO_RTL end (when nav_state changes from RTL)
        rtl_end_idx = None
        for i in range(rtl_idx + 1, len(df)):
            if df.iloc[i]['nav_state'] != NAV_STATES['AUTO_RTL']:
                rtl_end_idx = i
                break
        
        if rtl_end_idx is None:
            phase_transitions['phase3_end'] = df.iloc[-1]['timestamp']
        else:
            phase_transitions['phase3_end'] = df.iloc[rtl_end_idx]['timestamp']
        
        # Create phase ranges
        phase_ranges = [
            (phase_transitions['phase0_start'], phase_transitions['phase1_start'], 0),
            (phase_transitions['phase1_start'], phase_transitions['phase2_start'], 1),
            (phase_transitions['phase2_start'], phase_transitions['phase3_start'], 2),
            (phase_transitions['phase3_start'], phase_transitions['phase3_end'], 3)
        ]
        
        return phase_ranges
        
    except Exception as e:
        print(f"Error extracting flight phases: {e}")
        return None


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def load_parameter_summary(parameter_csv_path):
    """Load the parameter summary CSV file."""
    try:
        df = pd.read_csv(parameter_csv_path)
        return df
    except Exception as e:
        print(f"Error loading parameter summary: {e}")
        return None


def extract_env_factors_for_experiment(exp_info, parameter_df):
    """
    Extract environmental factors for a specific experiment from parameter summary.
    
    Args:
        exp_info: Dictionary with experiment info (exp_id, iris_id, windy_id, run_id)
        parameter_df: Parameter summary dataframe
        
    Returns:
        dict: Environmental factors for this experiment
    """
    env_factors = dict.fromkeys(ENV_FACTORS, None)
    
    if parameter_df is None or exp_info is None:
        return env_factors
    
    try:
        # Match based on iris_file and world_file
        iris_file_expected = f"iris_{exp_info['iris_id']:04d}.sdf"
        world_file_expected = f"windy_{exp_info['windy_id']:04d}.world"
        
        # Find the row matching iris_file and world_file
        exp_rows = parameter_df[
            (parameter_df['iris_file'] == iris_file_expected) &
            (parameter_df['world_file'] == world_file_expected)
        ]
        
        if not exp_rows.empty:
            # Use the first matching row
            row = exp_rows.iloc[0]
            for factor in ENV_FACTORS:
                if factor in parameter_df.columns:
                    try:
                        env_factors[factor] = float(row[factor])
                    except (ValueError, TypeError):
                        env_factors[factor] = None
        
        return env_factors
        
    except Exception as e:
        print(f"Error extracting env factors for {exp_info}: {e}")
        return env_factors


def remove_constant_columns(df):
    """Remove columns that have all identical values."""
    constant_cols = []
    for col in df.columns:
        if col not in ['timestamp', 'timestamp_sample']:
            try:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            except:
                pass
    return df.drop(columns=constant_cols)


def calculate_statistics_for_data(data, prefix):
    """
    Calculate statistics for all numeric columns in the data.
    
    Args:
        data: DataFrame with numeric data (no timestamp columns)
        prefix: Prefix for column names (e.g., 'topic_column' or 'topic_column_Phase0')
        
    Returns:
        dict: Dictionary with statistic values
    """
    stats = {}
    
    if data.empty:
        return stats
    
    for col in data.columns:
        if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            for stat in STATISTICS:
                col_name = f"{prefix}_{stat}"
                try:
                    if stat == 'mean':
                        stats[col_name] = data[col].mean()
                    elif stat == 'std':
                        stats[col_name] = data[col].std()
                    elif stat == 'max':
                        stats[col_name] = data[col].max()
                    elif stat == 'min':
                        stats[col_name] = data[col].min()
                    elif stat == 'median':
                        stats[col_name] = data[col].median()
                except Exception:
                    stats[col_name] = np.nan
    
    return stats


def process_ground_truth_topic(csv_path, topic_name, phase_ranges):
    """
    Process a single ground truth topic and calculate all statistics.
    
    Args:
        csv_path: Path to the topic CSV file
        topic_name: Name of the topic
        phase_ranges: List of (start_time, end_time, phase_idx) tuples
        
    Returns:
        dict: All statistics for this topic
    """
    topic_stats = {}
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        if df.empty or 'timestamp' not in df.columns:
            return topic_stats
        
        # Special handling for vehicle_attitude_groundtruth - add roll, pitch, yaw
        if topic_name == 'vehicle_attitude_groundtruth':
            # Check if quaternion columns exist
            quat_cols = ['q[0]', 'q[1]', 'q[2]', 'q[3]']
            if all(col in df.columns for col in quat_cols):
                # Convert quaternions to Euler angles
                rolls, pitches, yaws = [], [], []
                for _, row in df.iterrows():
                    roll, pitch, yaw = quaternion_to_euler(
                        row['q[0]'], row['q[1]'], row['q[2]'], row['q[3]']
                    )
                    rolls.append(roll)
                    pitches.append(pitch)
                    yaws.append(yaw)
                
                # Add new columns
                df['roll'] = rolls
                df['pitch'] = pitches
                df['yaw'] = yaws
        
        # Remove unwanted columns
        columns_to_drop = COLUMNS_TO_DROP.get(topic_name, [])
        columns_to_drop.extend(['timestamp_sample'])  # Always remove timestamp_sample
        
        # Drop columns that exist
        existing_drop_cols = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
        # Remove constant columns for vehicle_local_position_groundtruth
        if topic_name == 'vehicle_local_position_groundtruth':
            df = remove_constant_columns(df)
        
        # Calculate whole mission statistics
        whole_mission_data = df.drop(columns=['timestamp'])
        for col in whole_mission_data.columns:
            prefix = f"{topic_name}_{col}"
            col_stats = calculate_statistics_for_data(whole_mission_data[[col]], prefix)
            topic_stats.update(col_stats)
        
        # Calculate per-phase statistics
        if phase_ranges:
            for start_time, end_time, phase_idx in phase_ranges:
                phase_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
                if not phase_data.empty:
                    phase_data_no_timestamp = phase_data.drop(columns=['timestamp'])
                    for col in phase_data_no_timestamp.columns:
                        prefix = f"{topic_name}_{col}_Phase{phase_idx}"
                        col_stats = calculate_statistics_for_data(phase_data_no_timestamp[[col]], prefix)
                        topic_stats.update(col_stats)
        
    except Exception as e:
        print(f"Error processing {topic_name}: {e}")
    
    return topic_stats


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_experiment_run(folder_path, parameter_df, csv_output_base):
    """
    Process a single experiment run (e.g., run01 or run02).
    
    Args:
        folder_path: Path to experiment folder
        parameter_df: Parameter summary dataframe
        csv_output_base: Base directory for CSV output
        
    Returns:
        dict: Extracted features or None if processing failed
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    
    # Extract experiment info from folder name
    exp_info = extract_experiment_info(folder_name)
    if exp_info is None:
        return None
    
    # Set up CSV output directory for this experiment
    csv_output_dir = Path(csv_output_base) / folder_name
    
    # Check if this experiment has been processed (CSV directory exists)
    if not csv_output_dir.exists():
        # Find ULG file in the folder
        ulg_files = list(folder_path.glob('*.ulg'))
        if not ulg_files:
            return None
        
        ulg_path = ulg_files[0]
        
        # Convert ULG to CSV
        print(f"Converting ULG to CSV for {folder_name}")
        if not convert_ulg_to_csv(ulg_path, csv_output_dir):
            return None
    
    # Get CSV files for required topics
    required_topics = ['vehicle_status'] + GROUND_TRUTH_TOPICS
    csv_files = get_csv_files_for_topics(csv_output_dir, required_topics)
    
    # Extract flight phases
    if 'vehicle_status' not in csv_files:
        print(f"Missing vehicle_status CSV for {folder_name}")
        return None
    
    phase_ranges = extract_flight_phases(csv_files['vehicle_status'])
    if phase_ranges is None:
        print(f"Incomplete phases in {folder_name}, skipping experiment")
        return None
    
    # Extract environmental factors
    env_factors = extract_env_factors_for_experiment(exp_info, parameter_df)
    
    # Initialize experiment data
    experiment_data = {
        'exp_id': exp_info['exp_id'],
        **env_factors
    }
    
    # Process each ground truth topic
    for topic in GROUND_TRUTH_TOPICS:
        if topic in csv_files:
            topic_stats = process_ground_truth_topic(csv_files[topic], topic, phase_ranges)
            experiment_data.update(topic_stats)
    
    return experiment_data


def process_all_experiments():
    """
    Process all PX4 experiment data.
    
    Returns:
        pd.DataFrame: Merged dataframe with all processed data
    """
    data_root = PX4_CONFIG["data_root"]
    parameter_csv = PX4_CONFIG["parameter_summary"]
    csv_output_base = PX4_CONFIG["csv_output"]
    analysis_output = PX4_CONFIG["analysis_output"]
    
    print(f"Processing PX4 data from: {data_root}")
    
    # Ensure output directories exist
    ensure_output_directory(csv_output_base)
    ensure_output_directory(analysis_output)
    
    # Load parameter summary
    parameter_df = load_parameter_summary(parameter_csv)
    if parameter_df is None:
        print("Failed to load parameter summary. Continuing without environmental factors.")
    
    # Find all experiment folders
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        return None
    
    folders = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and item.name.startswith('exp_'):
            folders.append(item)
    
    total_folders = len(folders)
    print(f"Found {total_folders} experiment folders to process")
    
    if total_folders == 0:
        print("No experiment folders found.")
        return None
    
    all_experiment_data = []
    
    # Use limited number of processes to avoid system freeze
    import os
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if can't detect
    max_workers = cpu_count // 2  # Use half of available CPUs
    print(f"Using {max_workers} worker processes (CPU count: {cpu_count})")
    
    # Process folders in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_folder = {}
        for folder_path in folders:
            future = executor.submit(process_single_experiment_run, folder_path, parameter_df, csv_output_base)
            future_to_folder[future] = folder_path.name
        
        # Process completed jobs
        processed_count = 0
        error_count = 0
        
        for future in as_completed(future_to_folder):
            folder_name = future_to_folder[future]
            
            try:
                result = future.result()
                if result:
                    all_experiment_data.append(result)
                else:
                    error_count += 1
            except Exception as e:
                print(f"\nError processing {folder_name}: {e}")
                error_count += 1
            
            processed_count += 1
            
            # Progress update
            progress_pct = (processed_count / total_folders) * 100
            success_count = len(all_experiment_data)
            status = f"Progress: {processed_count}/{total_folders} ({progress_pct:.1f}%) | "
            status += f"Success: {success_count} | "
            if error_count > 0:
                status += f"Errors: {error_count} | "
            status += f"Current: {folder_name}"
            
            print(f"\r{status:<120}", end='', flush=True)
    
    # Final status
    success_rate = (len(all_experiment_data) / total_folders) * 100 if total_folders > 0 else 0
    final_msg = f" Processing complete: {len(all_experiment_data)}/{total_folders} folders successful ({success_rate:.1f}%)"
    if error_count > 0:
        final_msg += f" | {error_count} processing errors"
    print(f"\r{final_msg:<120}")
    
    if not all_experiment_data:
        print(" No valid data was processed.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_experiment_data)
    
    # Group by (exp_id, iris_id, windy_id) and average run01/run02
    print("Averaging replications...")
    
    # Extract iris_id and windy_id for grouping (if they exist in env factors)
    grouping_cols = ['exp_id']
    
    # Find iris and windy identifiers in the data
    for col in df.columns:
        if any(env_factor in col for env_factor in ENV_FACTORS):
            # We'll use exp_id as primary grouping, but we need to handle replication
            break
    
    # Group by exp_id (assuming consecutive exp_ids with run01/run02 should be averaged)
    # We need to map exp_ids to their base experiment
    df['base_exp_id'] = df['exp_id'].apply(lambda x: (x - 1) // 2 + 1)  # Map pairs to same base
    
    # Group by base_exp_id and average
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'base_exp_id' in numeric_cols:
        numeric_cols.remove('base_exp_id')
    if 'exp_id' in numeric_cols:
        numeric_cols.remove('exp_id')
    
    # Average all numeric columns by base_exp_id
    df_averaged = df.groupby('base_exp_id')[numeric_cols].mean().reset_index()
    
    # Keep the base_exp_id as exp_id
    df_averaged['exp_id'] = df_averaged['base_exp_id']
    df_averaged = df_averaged.drop(columns=['base_exp_id'])
    
    # Reorder columns: exp_id first, then env factors, then physical states
    all_columns = list(df_averaged.columns)
    env_factor_cols = [col for col in all_columns if col in ENV_FACTORS]
    physical_state_cols = [col for col in all_columns if col not in ['exp_id'] + ENV_FACTORS]
    
    ordered_columns = ['exp_id'] + ENV_FACTORS + physical_state_cols
    df_final = df_averaged[[col for col in ordered_columns if col in df_averaged.columns]]
    
    # Sort by exp_id
    df_final = df_final.sort_values('exp_id').reset_index(drop=True)
    
    # Save merged data
    output_path = analysis_output / "merged_data.csv"
    print(f"Saving merged data to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print(f" Merged data saved ({df_final.shape[0]} rows, {df_final.shape[1]} columns)")
    
    return df_final


def check_for_new_experiments():
    """
    Check if there are new experiment folders that haven't been processed.
    
    Returns:
        bool: True if new experiments are found, False otherwise
    """
    data_root = PX4_CONFIG["data_root"]
    csv_output_base = PX4_CONFIG["csv_output"]
    
    if not data_root.exists():
        return False
    
    # Get all experiment folders sorted by name
    experiment_folders = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and item.name.startswith('exp_'):
            experiment_folders.append(item.name)
    
    if not experiment_folders:
        return False
    
    # Check CSV output base directory
    if not csv_output_base.exists():
        print(f"CSV output directory doesn't exist. Will process all {len(experiment_folders)} experiments.")
        return True
    
    # Get the last processed experiment folder
    processed_folders = []
    for item in sorted(csv_output_base.iterdir()):
        if item.is_dir() and item.name.startswith('exp_'):
            processed_folders.append(item.name)
    
    if not processed_folders:
        print(f"No processed experiments found. Will process all {len(experiment_folders)} experiments.")
        return True
    
    # Find experiments that haven't been processed
    last_processed = processed_folders[-1] if processed_folders else ""
    new_experiments = []
    
    for exp_folder in experiment_folders:
        if exp_folder > last_processed:  # String comparison works for exp_XXXX format
            new_experiments.append(exp_folder)
    
    if new_experiments:
        print(f"Found {len(new_experiments)} new experiments after {last_processed}:")
        for exp in new_experiments[:5]:  # Show first 5
            print(f"  - {exp}")
        if len(new_experiments) > 5:
            print(f"  ... and {len(new_experiments) - 5} more")
        return True
    
    print(f"All experiments up to {experiment_folders[-1]} have been processed.")
    return False


def get_experiment_count():
    """
    Get the total number of experiment folders.
    
    Returns:
        int: Number of experiment folders
    """
    data_root = PX4_CONFIG["data_root"]
    
    if not data_root.exists():
        return 0
    
    count = 0
    for item in data_root.iterdir():
        if item.is_dir() and item.name.startswith('exp_'):
            count += 1
    
    return count


def get_last_processed_experiment():
    """
    Get the name of the last processed experiment folder.
    
    Returns:
        str: Name of the last processed experiment, or empty string if none
    """
    csv_output_base = PX4_CONFIG["csv_output"]
    
    if not csv_output_base.exists():
        return ""
    
    processed_folders = []
    for item in sorted(csv_output_base.iterdir()):
        if item.is_dir() and item.name.startswith('exp_'):
            processed_folders.append(item.name)
    
    return processed_folders[-1] if processed_folders else ""


def check_processed_data_validity():
    """
    Check if existing processed data is still valid (no new experiments).
    
    Returns:
        bool: True if existing data is valid, False if needs reprocessing
    """
    # Always return False to force regeneration of merged and filtered CSV files
    # This ensures we always recalculate even if no new experiments are found
    return False


def create_filtered_data(merged_df):
    """
    Create filtered dataset by removing columns with NaN values and constant columns.
    
    Args:
        merged_df: Merged DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if merged_df is None:
        return None
    
    print("Creating filtered dataset...")
    
    # Keep exp_id and environmental factors (even with NaN)
    protected_cols = ['exp_id'] + ENV_FACTORS
    protected_data = merged_df[protected_cols]
    
    # Filter physical state columns (remove any with NaN)
    physical_cols = [col for col in merged_df.columns if col not in protected_cols]
    physical_data = merged_df[physical_cols]
    
    # Remove columns that are entirely NaN
    physical_data = physical_data.dropna(axis=1, how='all')
    
    # Remove columns that contain any NaN
    physical_data = physical_data.dropna(axis=1, how='any')
    
    # Remove constant columns (columns with only one unique value)
    constant_cols = []
    for col in physical_data.columns:
        try:
            if physical_data[col].nunique() <= 1:
                constant_cols.append(col)
        except:
            pass
    
    if constant_cols:
        physical_data = physical_data.drop(columns=constant_cols)
        print(f"Removed {len(constant_cols)} constant columns")
    
    # Combine protected and filtered physical data
    filtered_df = pd.concat([protected_data, physical_data], axis=1)
    
    print(f"Filtered data: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
    print(f"Removed {merged_df.shape[1] - filtered_df.shape[1]} columns total (NaN + constant)")
    
    return filtered_df
    """
    Create filtered dataset by removing columns with NaN values.
    
    Args:
        merged_df: Merged DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if merged_df is None:
        return None
    
    print("Creating filtered dataset...")
    
    # Keep exp_id and environmental factors (even with NaN)
    protected_cols = ['exp_id'] + ENV_FACTORS
    protected_data = merged_df[protected_cols]
    
    # Filter physical state columns (remove any with NaN)
    physical_cols = [col for col in merged_df.columns if col not in protected_cols]
    physical_data = merged_df[physical_cols]
    
    # Remove columns that are entirely NaN
    physical_data = physical_data.dropna(axis=1, how='all')
    
    # Remove columns that contain any NaN
    physical_data = physical_data.dropna(axis=1, how='any')
    
    # Combine protected and filtered physical data
    filtered_df = pd.concat([protected_data, physical_data], axis=1)
    
    print(f"Filtered data: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
    print(f"Removed {merged_df.shape[1] - filtered_df.shape[1]} columns with NaN values")
    
    return filtered_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_or_process_px4_data():
    """
    Load existing processed PX4 data or process raw data if not available.
    
    Returns:
        pd.DataFrame: Processed and filtered data
    """
    analysis_output = PX4_CONFIG["analysis_output"]
    ensure_output_directory(analysis_output)
    
    merged_path = analysis_output / "merged_data.csv"
    filtered_path = analysis_output / "filtered_data.csv"
    
    print(f"\n--- Processing PX4 Data ---")
    
    # Check if existing processed data is still valid
    data_is_valid = check_processed_data_validity()
    
    if not data_is_valid:
        print(" Processing data from scratch (new experiments detected or invalid existing data)...")
        
        # Remove existing files to force reprocessing
        if merged_path.exists():
            merged_path.unlink()
            print(f"Removed outdated merged_data.csv")
        if filtered_path.exists():
            filtered_path.unlink()
            print(f"Removed outdated filtered_data.csv")
        
        # Process raw data from scratch
        df_merged = process_all_experiments()
        
        if df_merged is None:
            raise ValueError("Failed to process raw PX4 data.")
        
        # Apply filtering
        print("Applying filtering...")
        df_filtered = create_filtered_data(df_merged)
        
        # Save filtered data
        print(f"Saving filtered data...")
        df_filtered.to_csv(filtered_path, index=False)
        print(f" Filtered data saved to {filtered_path}")
        return df_filtered
    
    # Check for filtered data first
    if filtered_path.exists():
        print(f" Found existing valid filtered data: {filtered_path}")
        print("Loading existing filtered data...")
        return pd.read_csv(filtered_path)
    
    # Check for merged data
    if merged_path.exists():
        print(f" Found existing merged data: {merged_path}")
        print("Loading existing merged data and applying filtering...")
        df_merged = pd.read_csv(merged_path)
        
        # Apply filtering
        print("Applying filtering...")
        df_filtered = create_filtered_data(df_merged)
        
        # Save filtered data
        print(f"Saving filtered data...")
        df_filtered.to_csv(filtered_path, index=False)
        print(f" Filtered data saved to {filtered_path}")
        return df_filtered
    
    # This shouldn't happen if check_processed_data_validity works correctly
    print(" No existing processed data found")
    print("Processing raw data from scratch...")
    df_merged = process_all_experiments()
    
    if df_merged is None:
        raise ValueError("Failed to process raw PX4 data.")
    
    # Apply filtering
    print("Applying filtering...")
    df_filtered = create_filtered_data(df_merged)
    
    # Save filtered data
    print(f"Saving filtered data...")
    df_filtered.to_csv(filtered_path, index=False)
    print(f" Filtered data saved to {filtered_path}")
    
    return df_filtered


def main():
    """
    Main function to process PX4 simulation data and create filtered dataset.
    
    Returns:
        pd.DataFrame: Filtered and processed data
    """
    print("PX4 Data Preprocessing Pipeline")
    print("="*60)
    
    try:
        df_filtered = load_or_process_px4_data()
        
        print(f"\n PX4 preprocessing complete!")
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Environmental factors: {len([col for col in df_filtered.columns if col in ENV_FACTORS])}")
        print(f"Physical states: {len([col for col in df_filtered.columns if col not in ['exp_id'] + ENV_FACTORS])}")
        print(f"Output saved to: {PX4_CONFIG['analysis_output']}")
        
        return df_filtered
        
    except Exception as e:
        print(f" PX4 preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("PX4 Drone Simulation Data Preprocessing")
    print("Processing data from configured paths...")
    
    df_filtered = main()
    
    if df_filtered is not None:
        print("\n Sample of processed data:")
        print(df_filtered.head())
        print(f"\nColumns: {list(df_filtered.columns[:10])}...")  # Show first 10 columns
        
        # Show column breakdown
        env_cols = [col for col in df_filtered.columns if col in ENV_FACTORS]
        physical_cols = [col for col in df_filtered.columns if col not in ['exp_id'] + ENV_FACTORS]
        print(f"\nEnvironmental factors ({len(env_cols)}): {env_cols[:5]}...")
        print(f"Physical state columns ({len(physical_cols)}): {physical_cols[:5]}...")
    else:
        print("Failed to process PX4 data.")
        sys.exit(1)