"""
Drone Simulation Data Analysis Pipeline
=====================================

This script processes drone simulation logs, extracts features, and fits multiple
machine learning models to analyze the relationship between environmental factors
and physical flight states across different mission phases.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
import json
import os
import re
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from geopy.distance import geodesic

# Machine Learning Libraries
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib

# Specialized Libraries
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
import networkx as nx
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from xgboost import XGBRegressor
from mapie.regression import MapieRegressor
import lightgbm as lgb


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Directory Configuration
ROOT_DIR = "."
# Data paths for different simulators (to be configured by user)
SIMULATOR_CONFIGS = {
    "SITL": {
        "data_root": "./data",  # Update this path
        "analysis_output": "./analysis_output_sitl",
        "display_name": "SITL Simulator"
    },
    "Gazebo": {
        "data_root": "./data-gazebo",  # Update this path  
        "analysis_output": "./analysis_output_gazebo",
        "display_name": "Gazebo Simulator"
    }
}

# Data Processing Configuration
SELECTED_FILES = [
    'AHR2', 'ATT', 'BARO', 'BAT', 'CTUN', 'DCM', 'ESC', 'GPA', 'GPS', 
    'IMU', 'MAG', 'PM', 'POS', 'PSCD', 'PSCE', 'PSCN', 'RATE', 'SIM', 
    'SIM2', 'SRTL', 'TERR', 'VIBE', 'XKF4'
]

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

# Flight mode mapping for ArduCopter
MODE_MAPPING_ACM = {
    0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO', 4: 'GUIDED',
    5: 'LOITER', 6: 'RTL', 7: 'CIRCLE', 8: 'POSITION', 9: 'LAND',
    10: 'OF_LOITER', 11: 'DRIFT', 13: 'SPORT', 14: 'FLIP', 15: 'AUTOTUNE',
    16: 'POSHOLD', 17: 'BRAKE', 18: 'THROW', 19: 'AVOID_ADSB',
    20: 'GUIDED_NOGPS', 21: 'SMART_RTL', 22: 'FLOWHOLD', 23: 'FOLLOW',
    24: 'ZIGZAG', 25: 'SYSTEMID', 26: 'AUTOROTATE', 27: 'AUTO_RTL',
}

# Analysis Configuration
N_TOP = 200  # Number of top correlated factors
REPLICATION_COLUMN = "Rep"  # Column that identifies replicates


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_directory():
    """Ensure the analysis output directory exists."""
    os.makedirs(ANALYSIS_OUTPUT, exist_ok=True)


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


def get_experiment_rep(folder_name):
    """Determine which of the three repeated experiments the folder belongs to (Rep1, Rep2, Rep3)."""
    bin_number = folder_name.split("-")[0]  # Extract BIN file number (e.g., 00000001)
    try:
        bin_index = int(bin_number)  # Convert to integer
        return f"Rep{(bin_index - 1) % 3 + 1}"  # Assign Rep1, Rep2, or Rep3
    except ValueError:
        return None  # If parsing fails, return None


def get_sim_wind_spd(folder_name):
    """Extract SIM_WIND_SPD from the folder name."""
    parts = folder_name.split("-")
    if "SIM_WIND_SPD" in parts:
        idx = parts.index("SIM_WIND_SPD") + 1
        return float(parts[idx]) if idx < len(parts) else None
    return None


def get_hit_ground_speed(msg_csv_path):
    """Extract Hit Ground Speed from MSG.csv."""
    if not os.path.exists(msg_csv_path):
        print(f"Missing MSG.csv: {msg_csv_path}")
        return None
        
    try:
        df_msg = pd.read_csv(msg_csv_path, float_precision='round_trip')
        for _, row in df_msg.iterrows():
            message = str(row.get("Message", ""))  # Ensure message is string
            if "SIM Hit ground at" in message:
                parts = message.split(" ")
                return float(parts[-2])  # Extract speed value
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
        # Ensure column exists and dataframe isn't empty
        if "TimeUS" not in df_msg.columns or df_msg.empty:
            print(f"Invalid MSG.csv format or empty file: {msg_csv_path}")
            return None
        start_time = df_msg["TimeUS"].iloc[0]
        end_time = df_msg["TimeUS"].iloc[-1]
        duration = end_time - start_time
        return duration
    except Exception as e:
        print(f"Error reading MSG.csv: {e}")
        return None


def extract_mission_phases(msg_csv_path):
    """Extract mission phases from MSG.csv."""
    mission_phases = []
    disarm_time = None  # Time when motors are disarmed
    mission_num = 0
    
    try:
        df_msg = pd.read_csv(msg_csv_path, float_precision='round_trip')
        first_TimeUS = df_msg["TimeUS"].iloc[0]  # First recorded TimeUS
        last_TimeUS = df_msg["TimeUS"].iloc[-1]  # Last recorded TimeUS
        
        # Extract mission start TimeUSs
        for _, row in df_msg.iterrows():
            message = row.get("Message", "")
            TimeUS = row.get("TimeUS", None)
            if isinstance(message, str):
                if "Mission:" in message:
                    mission_phases.append(TimeUS)
                elif "Disarming motors" in message:
                    disarm_time = TimeUS  # Motors disarmed
        
        # Sort TimeUSs & create phase ranges
        mission_phases.sort()
        phase_ranges = [(first_TimeUS, mission_phases[0], mission_num)]  # Phase 0: Before first mission
        mission_num += 1
        
        for i in range(len(mission_phases) - 1):
            phase_ranges.append((mission_phases[i], mission_phases[i+1], mission_num))
            mission_num += 1
            
        # Explicitly define Phase 12 (RTL) → Ends at "Disarming motors"
        if disarm_time is not None:
            phase_ranges.append((mission_phases[-1], disarm_time, mission_num))  # Phase 12
            mission_num += 1
            
    except Exception as e:
        print(f"Error reading MSG.csv: {e}")
        
    return phase_ranges


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
                continue  # Skip if no data in the phase

            df_imu_phase = df_phase[df_phase["IMU"] == imu]
            if df_imu_phase.empty:
                continue  # Skip if this IMU has no data in this phase

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
        return  # Skip if ATT.csv is missing

    # Read ATT.csv with full precision
    df_att = pd.read_csv(att_path, float_precision='round_trip')

    # Ensure necessary columns exist
    required_cols = {"DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw", "TimeUS"}
    if not required_cols.issubset(df_att.columns):
        return  # Skip if required columns are missing

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
            continue  # Skip if no data in the phase

        folder_data[f"Roll_Error_Phase{phase_num}"] = df_phase["Roll_Error"].mean()
        folder_data[f"Pitch_Error_Phase{phase_num}"] = df_phase["Pitch_Error"].mean()
        folder_data[f"Yaw_Error_Phase{phase_num}"] = df_phase["Yaw_Error"].mean()
        folder_data[f"RPY_Error_Phase{phase_num}"] = df_phase["RPY_Error"].mean()


def compute_brake_time(bin_folder, folder_data):
    """Compute brake time from BRAKE mode initiation to full stop."""
    sim2_path = os.path.join(bin_folder, "SIM2.csv")
    mode_path = os.path.join(bin_folder, "MODE.csv")
    
    mode_df = pd.read_csv(mode_path, float_precision='round_trip')
    sim2_df = pd.read_csv(sim2_path, float_precision='round_trip')
    
    # Compute 3D speed from velocity components
    sim2_df["speed"] = np.sqrt(sim2_df["VN"]**2 + sim2_df["VE"]**2 + sim2_df["VD"]**2)
    
    # Find BRAKE mode (ModeNum == 17)
    brake_rows = mode_df[mode_df["ModeNum"] == 17].reset_index()
    if brake_rows.empty:
        raise ValueError(f"No BRAKE mode found in {bin_folder}")
    
    start_time = brake_rows.loc[0, "TimeUS"]
    
    # Extract data after BRAKE begins
    post_brake = sim2_df[sim2_df["TimeUS"] >= start_time].copy()
    if post_brake.empty:
        raise ValueError(f"No SIM2 data after BRAKE mode start in {bin_folder}")
    
    t_start = post_brake["TimeUS"].iloc[0] * 1e-6
    initial_speed = post_brake["speed"].iloc[0]
    STOP_THRESHOLD = 0.1  # m/s
    
    # Find first point when speed < STOP_THRESHOLD
    stopped = post_brake[post_brake["speed"] < STOP_THRESHOLD]
    if stopped.empty:
        raise ValueError(f"Drone never stopped (speed < {STOP_THRESHOLD}) in {bin_folder}")
    
    t_stop = stopped["TimeUS"].iloc[0] * 1e-6
    brake_time = t_stop - t_start
    folder_data["Brake_Time"] = brake_time


def point_line_distance(p, p1, p2):
    """
    Compute perpendicular distance of point `p` from line segment `p1` to `p2`.
    Computes both:
      - X-Y Plane Crosstrack Error (meters)
      - 3D Crosstrack Error (includes Altitude)
    """
    lat_p, lon_p, alt_p = p
    lat1, lon1, alt1 = p1
    lat2, lon2, alt2 = p2

    # Convert lat/lon to meters using haversine
    ref_point = (lat1, lon1)  # Use wp_start as reference point
    xy_a = np.array([0, 0])  # Reference point is (0,0)
    xy_b = np.array([haversine_distance(lat1, lon1, lat2, lon1), haversine_distance(lat1, lon1, lat1, lon2)])
    xy_p = np.array([haversine_distance(lat1, lon1, lat_p, lon1), haversine_distance(lat1, lon1, lat1, lon_p)])

    # Compute 2D (X-Y Plane) vector projections
    xy_ab = xy_b - xy_a
    xy_ap = xy_p - xy_a
    xy_ab_norm = np.dot(xy_ab, xy_ab)

    if xy_ab_norm == 0:
        xy_error = np.linalg.norm(xy_ap)  # If p1 == p2, return distance from p to p1
    else:
        xy_proj = np.dot(xy_ap, xy_ab) / xy_ab_norm
        xy_proj_point = xy_a + xy_proj * xy_ab
        xy_error = np.linalg.norm(xy_p - xy_proj_point)  # X-Y plane crosstrack error

    # Compute altitude error separately
    xy_proj = np.clip(xy_proj, 0, 1)  # Ensure projection stays in valid range
    alt_error = abs(alt_p - (alt1 + (alt2 - alt1) * xy_proj))

    # Compute full 3D crosstrack error
    full_3d_error = np.sqrt(xy_error ** 2 + alt_error ** 2)

    return full_3d_error, xy_error, alt_error


def compute_crosstrack_error(folder_path, folder_data, mission_phases, phase_waypoints):
    """
    Compute 3D Crosstrack Error, X-Y Crosstrack Error, and Altitude Error.
    Updates the `folder_data` dictionary with total and phase-wise values.
    """
    sim_path = os.path.join(folder_path, "SIM.csv")  # Actual UAV positions

    if not os.path.exists(sim_path):
        return  # Skip if required files are missing
    
    df_sim = pd.read_csv(sim_path, float_precision="round_trip")

    if not all(col in df_sim.columns for col in ["TimeUS", "Lat", "Lng", "Alt"]):
        print("SIM.csv is missing required columns: TimeUS, Lat, Lng, Alt")
        return

    total_xtrack_error = 0
    total_xy_error = 0
    total_alt_error = 0
    total_points = 0  # To normalize errors
    phase_errors = {}  # Temporary storage for phase errors
    
    for start_time, end_time, phase_num in mission_phases:
        if phase_num not in phase_waypoints:
            continue  # Skip phases that don't have defined waypoints

        wp_start, wp_end = phase_waypoints[phase_num]  # Start & End waypoints of the phase

        # Extract the relevant time range for this phase
        df_phase = df_sim[(df_sim["TimeUS"] >= start_time) & (df_sim["TimeUS"] <= end_time)]

        if df_phase.empty:
            continue  # Skip phases with no data

        # Compute XTE for each point in the phase
        errors = df_phase.apply(lambda row: point_line_distance(
            (row["Lat"], row["Lng"], row["Alt"]),
            wp_start, wp_end
        ), axis=1)

        # Extract individual error components
        df_phase = df_phase.copy()  # Ensure it's a modifiable DataFrame
        df_phase["Crosstrack_Error"], df_phase["XY_Crosstrack_Error"], df_phase["Altitude_Error"] = zip(*errors)

        # Store mean phase errors
        phase_errors[f"Crosstrack_Error_Phase{phase_num}"] = df_phase["Crosstrack_Error"].mean()
        phase_errors[f"XY_Crosstrack_Error_Phase{phase_num}"] = df_phase["XY_Crosstrack_Error"].mean()
        phase_errors[f"Altitude_Error_Phase{phase_num}"] = df_phase["Altitude_Error"].mean()

        # Accumulate total errors
        total_xtrack_error += df_phase["Crosstrack_Error"].sum()
        total_xy_error += df_phase["XY_Crosstrack_Error"].sum()
        total_alt_error += df_phase["Altitude_Error"].sum()
        total_points += len(df_phase)

    # Normalize total errors
    folder_data["Crosstrack_Error"] = total_xtrack_error / total_points if total_points else None
    folder_data["XY_Crosstrack_Error"] = total_xy_error / total_points if total_points else None
    folder_data["Altitude_Error"] = total_alt_error / total_points if total_points else None

    # Store each phase error afterward
    folder_data.update(phase_errors)


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
        print(f"PARM.csv not found in {bin_folder}")
        return None

    # Extract environmental factors
    env_factors = extract_env_factors(parm_path)
    if all(v is None for v in env_factors.values()):
        print(f"No ENV_FACTORS found in {bin_folder}")
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

    # Process each selected CSV file
    for csv_file in SELECTED_FILES:
        csv_path = os.path.join(bin_folder, csv_file + ".csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path, float_precision='round_trip')
            if "TimeUS" not in df.columns:
                continue

            print(f"Processing {csv_path}...")
            compute_whole_mission_stats(df, csv_file, folder_data)
            compute_per_phase_stats(df, csv_file, folder_data, mission_phases)

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            
    return folder_data


def process_all_data():
    """
    Process all bin folders and merge the results.
    
    Returns:
        pd.DataFrame: Merged dataframe with all processed data
    """
    all_data = []

    # Process all folders in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for folder in sorted(os.listdir(DATA_ROOT)):
            full_path = os.path.join(DATA_ROOT, folder)
            if os.path.isdir(full_path):
                futures.append(executor.submit(process_bin_folder, full_path))

        for future in as_completed(futures):
            result = future.result()
            if result:
                all_data.append(result)

    # Merge and organize results
    if all_data:
        df_merged = pd.DataFrame(all_data)

        # Sort by all ENV_FACTORS
        df_merged = df_merged.sort_values(by=ENV_FACTORS)

        # Group by ENV_FACTORS and assign rep1, rep2, ...
        df_merged["Rep"] = (
            df_merged.groupby(ENV_FACTORS)
            .cumcount() + 1
        ).apply(lambda x: f"rep{x}")

        # Reorder columns: ENV_FACTORS + ["Rep"] + rest
        all_columns = list(df_merged.columns)
        env_set = set(ENV_FACTORS)
        other_columns = [col for col in all_columns if col not in ENV_FACTORS and col != "Rep"]
        ordered_columns = ENV_FACTORS + ["Rep"] + other_columns
        df_merged = df_merged[ordered_columns]

        # Save to CSV
        ensure_output_directory()
        output_path = os.path.join(ANALYSIS_OUTPUT, "merged_data.csv")
        df_merged.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}!")
        
        return df_merged
    else:
        print("No valid data was processed.")
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
    variance = df_cleaned.drop(columns=ENV_FACTORS).var(numeric_only=True)
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


# ============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# ============================================================================

def compute_top_correlations(df, env_factor, n=N_TOP):
    """
    Compute the top `n` correlated non-environmental factors for a given environmental factor.

    Args:
        df: The dataframe
        env_factor: The environmental factor to analyze
        n: The number of top correlated factors to find
        
    Returns:
        Series of top correlated factors
    """
    if env_factor not in df.columns:
        print(f"Warning: {env_factor} not found in dataframe. Skipping...")
        return None

    corr_matrix = df.corr(numeric_only=True)  # Compute correlation matrix

    # Get absolute correlation values excluding itself and other env factors
    excluded = set(ENV_FACTORS)  # Exclude all env factors including self
    correlation_values = corr_matrix[env_factor].drop(labels=excluded, errors="ignore").abs()

    # Get the top N most correlated non-env factors
    return correlation_values.nlargest(n)


def plot_correlation_heatmap(df, env_factor, top_n_correlations):
    """
    Plot a heatmap showing the correlation of the top N features with the environmental factor.
    
    Args:
        df: Processed DataFrame
        env_factor: Environmental factor for correlation
        top_n_correlations: The top correlated features
    """
    # Ensure only existing features are selected
    valid_features = [f for f in top_n_correlations.index if f in df.columns]

    if not valid_features:
        print(f"Warning: No valid features found for {env_factor}. Skipping heatmap.")
        return

    # Create a valid DataFrame for heatmap
    heatmap_df = df[valid_features + [env_factor]]

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df.corr()[[env_factor]], annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Top {len(valid_features)} Correlated Factors with {env_factor}")
    plt.show()


def analyze_top_correlations(df, env_factors=ENV_FACTORS, n=N_TOP, output_csv="Top_Correlated_Factors.csv"):
    """
    Analyze the top N correlated factors for each environmental factor.

    Args:
        df: The dataframe
        env_factors: List of environmental factors
        n: Number of top correlated factors to compute
        output_csv: Output CSV file path
    """
    df_preprocessed = preprocess_dataframe(df)
    
    # Ensure environmental factors exist before computing correlations
    for env_factor in env_factors:
        if env_factor not in df_preprocessed.columns:
            print(f"Warning: {env_factor} not found in the preprocessed dataframe.")

    top_correlations = {}

    for env_factor in env_factors:
        top_n_correlations = compute_top_correlations(df_preprocessed, env_factor, n)
        if top_n_correlations is not None:
            top_correlations[env_factor] = top_n_correlations

            # Print results
            print(f"\nTop {n} most correlated factors with {env_factor}:")
            print(top_n_correlations)

    # Save results to CSV
    df_top_correlations = pd.DataFrame.from_dict(top_correlations, orient='index').T
    output_path = os.path.join(ANALYSIS_OUTPUT, output_csv)
    df_top_correlations.to_csv(output_path, index=True)
    print(f"Top correlated factors saved to {output_path}")


# ============================================================================
# MACHINE LEARNING MODEL FUNCTIONS
# ============================================================================

class ModelTrainer:
    """Base class for training different types of models."""
    
    def __init__(self, env_factors, output_dir):
        self.env_factors = env_factors
        self.output_dir = output_dir
        
    def prepare_data(self, df_filtered, phys_state):
        """Prepare data for model training."""
        df_valid = df_filtered[self.env_factors + [phys_state]].dropna()
        if df_valid.empty:
            return None, None
        
        X = df_valid[self.env_factors].values
        y = df_valid[phys_state].values
        return X, y
    
    def evaluate_model(self, model, X, y, cv=5):
        """Evaluate model using cross-validation."""
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        return scores.mean()


class PolynomialRidgeTrainer(ModelTrainer):
    """Trainer for Polynomial + Ridge regression models."""
    
    def __init__(self, env_factors, output_dir, degree=2):
        super().__init__(env_factors, output_dir)
        self.degree = degree
        
    def train_model(self, df_filtered, phys_state, phase):
        """Train a polynomial ridge model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Use a pipeline with polynomial features and Ridge regression
        pipeline = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            Ridge(alpha=1.0)
        )

        # Cross-validation to get average R²
        r2_cv_mean = self.evaluate_model(pipeline, X, y)

        # Refit model on full data
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = Ridge(alpha=1.0).fit(X_poly, y)

        # Save model and transformer
        model_dir = os.path.join(self.output_dir, f"models_phase{phase}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": model,
            "poly": poly,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path,
            **{f"coef_{i}": v for i, v in enumerate(model.coef_)}
        }


class GaussianProcessTrainer(ModelTrainer):
    """Trainer for Gaussian Process Regression models."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train a Gaussian Process model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Define kernel and model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(X.shape[1]), 
            length_scale_bounds=(1e-2, 1e2)
        )
        model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-2, 
            normalize_y=True
        )

        # Cross-validation for generalization performance
        r2_cv_mean = self.evaluate_model(model, X, y)

        # Refit on full data
        model.fit(X, y)

        # Save model
        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_gpr")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": model,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path
        }


class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost models."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train an XGBoost model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        model = XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            verbosity=0, 
            random_state=0
        )
        
        r2_cv_mean = self.evaluate_model(model, X, y)
        model.fit(X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_xgb")
        os.makedirs(model_dir, exist_ok=True)

        # Save in JSON format for cross-version compatibility
        model_path = os.path.join(model_dir, f"{phys_state}.json")
        model.get_booster().save_model(model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path
        }


class LightGBMTrainer(ModelTrainer):
    """Trainer for LightGBM models with quantile regression."""
    
    def train_quantile_model(self, X, y, quantile_alpha=None):
        """Train a quantile regression model."""
        if quantile_alpha is None:
            model = lgb.LGBMRegressor(
                objective='regression', 
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=0
            )
        else:
            model = lgb.LGBMRegressor(
                objective='quantile', 
                alpha=quantile_alpha, 
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=0
            )
        model.fit(X, y)
        return model
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train LightGBM models for median and quantile predictions."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Train three models: median, lower quantile, upper quantile
        model_median = self.train_quantile_model(X, y, quantile_alpha=None)
        model_lower = self.train_quantile_model(X, y, quantile_alpha=0.025)
        model_upper = self.train_quantile_model(X, y, quantile_alpha=0.975)

        r2_cv_mean = self.evaluate_model(model_median, X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_lgb")
        os.makedirs(model_dir, exist_ok=True)

        # Save models in txt format (LightGBM saves as .txt)
        model_median_path = os.path.join(model_dir, f"{phys_state}_median.txt")
        model_lower_path = os.path.join(model_dir, f"{phys_state}_lower.txt")
        model_upper_path = os.path.join(model_dir, f"{phys_state}_upper.txt")

        model_median.booster_.save_model(model_median_path)
        model_lower.booster_.save_model(model_lower_path)
        model_upper.booster_.save_model(model_upper_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_median_path": model_median_path,
            "model_lower_path": model_lower_path,
            "model_upper_path": model_upper_path
        }


def process_one_state_with_trainer(args):
    """Process a single physical state with a given trainer."""
    trainer, df_filtered, phys_state, phase = args
    return trainer.train_model(df_filtered, phys_state, phase)


def process_multivariate_analysis_with_trainer(
    df_filtered, 
    trainer_class, 
    analysis_output,
    phase_range=range(1, 13), 
    suffix="",
    **trainer_kwargs
):
    """
    Process multivariate analysis using a specific trainer class.
    
    Args:
        df_filtered: Preprocessed DataFrame
        trainer_class: Class to use for training (e.g., PolynomialRidgeTrainer)
        analysis_output: Output directory for this simulator
        phase_range: Range of phases to process
        suffix: Suffix for output files
        **trainer_kwargs: Additional arguments for trainer initialization
    """
    all_phase_r2_summary = []
    trainer = trainer_class(ENV_FACTORS, analysis_output, **trainer_kwargs)

    for phase in phase_range:
        phase_suffix = f"_Phase{phase}"
        phys_states = [
            col for col in df_filtered.columns
            if col not in ENV_FACTORS and col.endswith(phase_suffix)
        ]

        if not phys_states:
            print(f"Phase {phase}: No physical states found, skipping...")
            continue

        print(f"Processing Phase {phase} with {trainer_class.__name__} on {len(phys_states)} physical states...")

        records = []
        r2_values = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            args_list = [
                (trainer, df_filtered, state, phase)
                for state in phys_states
            ]
            futures = [executor.submit(process_one_state_with_trainer, args) for args in args_list]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    r2_values.append(result["r2_cv"])

        if records:
            # Save per-model performance
            coef_df = pd.DataFrame(records)
            csv_path = os.path.join(analysis_output, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            coef_df.to_csv(csv_path, index=False)
            print(f"Saved {trainer_class.__name__} results for Phase {phase} to {csv_path}")

            # Generate histogram
            plot_r2_histogram(r2_values, phase, analysis_output, suffix)

            # Save summary
            summary = {
                "Phase": phase,
                "Num_Models": len(r2_values),
                "Num_R2_GTE_0.9": sum(r >= 0.9 for r in r2_values),
                "Num_R2_GTE_0.8": sum(r >= 0.8 for r in r2_values),
                "Num_R2_GTE_0.7": sum(r >= 0.7 for r in r2_values),
                "Mean_R2_CV": np.mean(r2_values),
                "Median_R2_CV": np.median(r2_values),
                "Min_R2_CV": np.min(r2_values),
                "Max_R2_CV": np.max(r2_values),
            }
            all_phase_r2_summary.append(summary)
        else:
            print(f"No {trainer_class.__name__} models passed R² threshold for Phase {phase}")

    if all_phase_r2_summary:
        summary_df = pd.DataFrame(all_phase_r2_summary)
        summary_csv_path = os.path.join(analysis_output, f"multivariate_r2_summary{suffix}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved {trainer_class.__name__} R² summary to {summary_csv_path}")dir = os.path.join(self.output_dir, f"models_phase{phase}_lgb")
        os.makedirs(model_dir, exist_ok=True)

        # Save models in txt format (LightGBM saves as .txt)
        model_median_path = os.path.join(model_dir, f"{phys_state}_median.txt")
        model_lower_path = os.path.join(model_dir, f"{phys_state}_lower.txt")
        model_upper_path = os.path.join(model_dir, f"{phys_state}_upper.txt")

        model_median.booster_.save_model(model_median_path)
        model_lower.booster_.save_model(model_lower_path)
        model_upper.booster_.save_model(model_upper_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_median_path": model_median_path,
            "model_lower_path": model_lower_path,
            "model_upper_path": model_upper_path
        }


def process_one_state_with_trainer(args):
    """Process a single physical state with a given trainer."""
    trainer, df_filtered, phys_state, phase = args
    return trainer.train_model(df_filtered, phys_state, phase)


def process_multivariate_analysis_with_trainer(
    df_filtered, 
    trainer_class, 
    phase_range=range(1, 13), 
    suffix="",
    **trainer_kwargs
):
    """
    Process multivariate analysis using a specific trainer class.
    
    Args:
        df_filtered: Preprocessed DataFrame
        trainer_class: Class to use for training (e.g., PolynomialRidgeTrainer)
        phase_range: Range of phases to process
        suffix: Suffix for output files
        **trainer_kwargs: Additional arguments for trainer initialization
    """
    all_phase_r2_summary = []
    trainer = trainer_class(ENV_FACTORS, ANALYSIS_OUTPUT, **trainer_kwargs)

    for phase in phase_range:
        phase_suffix = f"_Phase{phase}"
        phys_states = [
            col for col in df_filtered.columns
            if col not in ENV_FACTORS and col.endswith(phase_suffix)
        ]

        if not phys_states:
            print(f"Phase {phase}: No physical states found, skipping...")
            continue

        print(f"Processing Phase {phase} with {trainer_class.__name__} on {len(phys_states)} physical states...")

        records = []
        r2_values = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            args_list = [
                (trainer, df_filtered, state, phase)
                for state in phys_states
            ]
            futures = [executor.submit(process_one_state_with_trainer, args) for args in args_list]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    r2_values.append(result["r2_cv"])

        if records:
            # Save per-model performance
            coef_df = pd.DataFrame(records)
            csv_path = os.path.join(ANALYSIS_OUTPUT, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            coef_df.to_csv(csv_path, index=False)
            print(f"Saved {trainer_class.__name__} results for Phase {phase} to {csv_path}")

            # Generate histogram
            plot_r2_histogram(r2_values, phase, ANALYSIS_OUTPUT, suffix)

            # Save summary
            summary = {
                "Phase": phase,
                "Num_Models": len(r2_values),
                "Num_R2_GTE_0.9": sum(r >= 0.9 for r in r2_values),
                "Num_R2_GTE_0.8": sum(r >= 0.8 for r in r2_values),
                "Num_R2_GTE_0.7": sum(r >= 0.7 for r in r2_values),
                "Mean_R2_CV": np.mean(r2_values),
                "Median_R2_CV": np.median(r2_values),
                "Min_R2_CV": np.min(r2_values),
                "Max_R2_CV": np.max(r2_values),
            }
            all_phase_r2_summary.append(summary)
        else:
            print(f"No {trainer_class.__name__} models passed R² threshold for Phase {phase}")

    if all_phase_r2_summary:
        summary_df = pd.DataFrame(all_phase_r2_summary)
        summary_csv_path = os.path.join(ANALYSIS_OUTPUT, f"multivariate_r2_summary{suffix}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved {trainer_class.__name__} R² summary to {summary_csv_path}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_r2_histogram(r2_values, phase, output_dir, suffix=""):
    """Plot R² distribution histogram for a given phase."""
    plt.figure(figsize=(6, 4))
    plt.hist(r2_values, bins=np.linspace(0, 1, 21), color="skyblue", edgecolor="black")
    plt.title(f"R² Distribution for Phase {phase}")
    plt.xlabel("R² Score")
    plt.ylabel("Number of Models")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"r2_histogram_phase{phase}{suffix}.pdf"
    plt.savefig(os.path.join(output_dir, filename), format="pdf")
    plt.close()


def plot_phase_model_fits(
    df_filtered,
    env_factors,
    phase=3,
    output_dir="./analysis_output",
    max_plots=12,
    suffix=""
):
    """Plot model fit quality for the best performing models in a phase."""
    os.makedirs(output_dir, exist_ok=True)

    # Load coefficient CSV
    coef_path = os.path.join(output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
    if not os.path.exists(coef_path):
        print(f"Coefficient file not found: {coef_path}")
        return
        
    coef_df = pd.read_csv(coef_path)

    # Sort by R² from cross-validation
    coef_df = coef_df.sort_values(by="r2_cv", ascending=False).head(max_plots)

    num_models = len(coef_df)
    cols = 4
    rows = int(np.ceil(num_models / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axs = axs.flatten()

    for idx, row in enumerate(coef_df.itertuples()):
        state = row.state
        model_path = os.path.join(output_dir, f"models_phase{phase}", f"{state}.joblib")

        # Load model
        if not os.path.exists(model_path):
            continue
            
        model_bundle = joblib.load(model_path)
        model = model_bundle["model"]
        poly = model_bundle["poly"]

        # Prepare data
        if state not in df_filtered.columns:
            continue
        df_valid = df_filtered[env_factors + [state]].dropna()
        X = df_valid[env_factors].values
        y_true = df_valid[state].values
        y_pred = model.predict(poly.transform(X))

        # Plot
        ax = axs[idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{state}\nCV R² = {row.r2_cv:.3f}")

    # Hide unused subplots
    for j in range(num_models, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"phase{phase}_model_fit_overview{suffix}.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()
    print(f"Saved Phase {phase} model fit overview to {output_path}")


def plot_all_r2_histograms(analysis_output_dir="./analysis_output", phase_range=range(1, 13), suffix=""):
    """Plot R² histograms for all phases in a single figure."""
    cols = 4
    rows = 3
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()

    for idx, phase in enumerate(phase_range):
        coef_path = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
        if not os.path.exists(coef_path):
            print(f"[Warning] File not found for Phase {phase}: {coef_path}")
            continue

        df = pd.read_csv(coef_path)
        if "r2_cv" not in df.columns:
            print(f"[Warning] r2_cv column not found in {coef_path}")
            continue

        r2_values = df["r2_cv"].values
        ax = axs[idx]
        ax.hist(r2_values, bins=np.linspace(0, 1, 21), color="skyblue", edgecolor="black")
        ax.set_title(f"Phase {phase}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(5, int(1.1 * np.max(np.histogram(r2_values, bins=21)[0]))))
        ax.set_xlabel("R²")
        ax.set_ylabel("Count")
        ax.grid(True)

    # Hide unused subplots if any
    for j in range(len(phase_range), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    output_pdf = os.path.join(analysis_output_dir, f"r2_histograms_all_phases{suffix}.png")
    plt.savefig(output_pdf, format="png")
    plt.close()
    print(f"Saved R² histograms to {output_pdf}")


def plot_model_performance_summary(summary_csv_path, output_dir="./analysis_output", model_name=""):
    """
    Visualize model R² performance summary per phase.

    Args:
        summary_csv_path: Path to the summary CSV
        output_dir: Directory to save the plot
        model_name: Name of the model for the title
    """
    os.makedirs(output_dir, exist_ok=True)
    df_summary = pd.read_csv(summary_csv_path)

    # Sort by phase number if needed
    if "Phase" in df_summary.columns:
        df_summary["PhaseNum"] = df_summary["Phase"]
        if df_summary["Phase"].dtype == object:
            df_summary["PhaseNum"] = df_summary["Phase"].str.extract(r"(\d+)").astype(int)
        df_summary = df_summary.sort_values("PhaseNum")

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_summary, x="PhaseNum", y="Mean_R2_CV", color="skyblue", label="Mean R² (CV)")

    plt.plot(df_summary["PhaseNum"], df_summary["Max_R2_CV"], label="Max R²", marker="o", linestyle="--", color="green")
    plt.plot(df_summary["PhaseNum"], df_summary["Min_R2_CV"], label="Min R²", marker="x", linestyle="--", color="red")
    plt.plot(df_summary["PhaseNum"], df_summary["Median_R2_CV"], label="Median R²", marker="^", linestyle="--", color="orange")

    plt.xticks(rotation=45)
    plt.title(f"{model_name} Model Performance (R² CV per Phase)")
    plt.xlabel("Phase")
    plt.ylabel("R² Score")
    plt.ylim(-0.5, 1.05)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"r2_summary_plot_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f" Saved {model_name} R² summary plot to: {plot_path}")
    return plot_path


# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================================================

def generate_phase_state_table(analysis_output_dir, r2_threshold=0.7, phase_range=range(1, 13), suffix=""):
    """Generate a table mapping phases to physical states with good model fit."""
    all_states = set()
    phase_to_states = {}

    # Pass 1: collect all good physical states
    for phase in phase_range:
        coef_path = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
        if not os.path.exists(coef_path):
            print(f"[Warning] Missing file: {coef_path}")
            continue

        df = pd.read_csv(coef_path)
        if "state" not in df.columns or "r2_cv" not in df.columns:
            print(f"[Warning] Invalid format in {coef_path}")
            continue

        df_good = df[df["r2_cv"] >= r2_threshold]
        if df_good.empty:
            continue

        clean_states = [re.sub(r"_Phase\d+$", "", s) for s in df_good["state"]]
        phase_to_states[phase] = set(clean_states)
        all_states.update(clean_states)

    # Sort for consistent column order
    all_states = sorted(all_states)
    rows = []

    for phase in phase_range:
        phase_name = f"Phase{phase}"
        row = {"Phase": phase_name}
        active_states = phase_to_states.get(phase, set())
        for state in all_states:
            row[state] = 1 if state in active_states else 0
        rows.append(row)

    df_matrix = pd.DataFrame(rows)
    output_path = os.path.join(analysis_output_dir, f"phase_to_states_matrix{suffix}.csv")
    df_matrix.to_csv(output_path, index=False)
    print(f"Saved binary phase-to-state matrix to {output_path}")
    return df_matrix


def plot_phase_state_heatmap(matrix_csv_path, output_dir="./analysis_output", suffix=""):
    """Plot heatmap showing which physical states are well-modeled in each phase."""
    os.makedirs(output_dir, exist_ok=True)

    # Load binary matrix (assumes 'Phase' column is present)
    df = pd.read_csv(matrix_csv_path)
    df = df.set_index("Phase")

    # Create heatmap
    plt.figure(figsize=(min(30, df.shape[1] * 0.4), min(10, df.shape[0] * 0.5)))
    sns.heatmap(df, annot=False, cmap="Blues", cbar=True, linewidths=0.5, linecolor='gray')

    plt.title("Phase-to-Physical State Coverage (R² ≥ 0.7)")
    plt.xlabel("Physical States")
    plt.ylabel("Phases")

    heatmap_path = os.path.join(output_dir, f"phase_state_heatmap{suffix}.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, format="png")
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")


def analyze_env_factor_influence_via_pca(
    df_filtered,
    env_factors,
    phase_range=range(1, 13),
    n_components=3,
    output_dir="./analysis_output",
    top_k=5
):
    """
    Analyze environmental factor influence using PCA on physical states.
    
    Args:
        df_filtered: Preprocessed DataFrame
        env_factors: List of environmental factors
        phase_range: Range of phases to analyze
        n_components: Number of PCA components
        output_dir: Output directory for results
        top_k: Number of top factors to show in heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    phase_records = []

    for phase in phase_range:
        suffix = f"_Phase{phase}"
        phys_state_cols = [col for col in df_filtered.columns if col.endswith(suffix) and col not in env_factors]

        if not phys_state_cols:
            print(f"[Phase {phase}] No physical state columns found.")
            continue

        cols_needed = env_factors + phys_state_cols
        df_phase = df_filtered[cols_needed].dropna()
        if df_phase.empty:
            print(f"[Phase {phase}] No valid data after dropping NaNs.")
            continue

        # Normalize physical states
        X_phys = df_phase[phys_state_cols]
        X_phys_std = (X_phys - X_phys.mean()) / X_phys.std()

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_phys_std)

        # Correlate ENV_FACTORS with each PC
        for comp_idx in range(n_components):
            pc_values = X_pca[:, comp_idx]
            for factor in env_factors:
                corr = np.corrcoef(df_phase[factor], pc_values)[0, 1]
                phase_records.append({
                    "Phase": f"Phase{phase}",
                    "PC": f"PC{comp_idx+1}",
                    "ENV_FACTOR": factor,
                    "Correlation": corr,
                    "AbsCorr": abs(corr),
                    "ExplainedVar": pca.explained_variance_ratio_[comp_idx]
                })

    # Save all correlation results
    df_result = pd.DataFrame(phase_records)
    result_path = os.path.join(output_dir, "pca_env_factor_influence.csv")
    df_result.to_csv(result_path, index=False)
    print(f" Saved PCA influence table to {result_path}")

    # Build heatmap for PC1 only
    df_pc1 = df_result[df_result["PC"] == "PC1"]
    df_pc1["PhaseNum"] = df_pc1["Phase"].str.extract(r"Phase(\d+)").astype(int)

    # Select top_k factors by AbsCorr per phase
    heatmap_df = (
        df_pc1.sort_values(["PhaseNum", "AbsCorr"], ascending=[True, False])
        .groupby("PhaseNum")
        .head(top_k)
        .pivot(index="PhaseNum", columns="ENV_FACTOR", values="AbsCorr")
        .fillna(0)
    )

    heatmap_df = heatmap_df.sort_index()  # ensure numerical order

    plt.figure(figsize=(min(heatmap_df.shape[1]*0.6, 20), 6))
    sns.heatmap(heatmap_df, annot=True, cmap="viridis", cbar=True, linewidths=0.5, linecolor='gray')
    plt.title(f"Top {top_k} ENV_FACTORS by Abs(Correlation) with PC1 (Per Phase)", fontsize=12)
    plt.ylabel("Phase")
    plt.xlabel("ENV Factor")
    plt.yticks(ticks=np.arange(len(heatmap_df.index)) + 0.5, labels=[f"Phase{p}" for p in heatmap_df.index], rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "pca_env_factor_pc1_top_heatmap.png")
    plt.savefig(heatmap_path, format="png")
    plt.close()
    print(f" Saved PC1 ENV_FACTOR heatmap to {heatmap_path}")

    return df_result


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def check_file_exists(filepath, description):
    """Check if a file exists and return the result with a message."""
    if os.path.exists(filepath):
        print(f" Found existing {description}: {filepath}")
        return True
    else:
        print(f" {description} not found: {filepath}")
        return False


def load_or_process_data():
    """Load existing data or process raw data if not available."""
    merged_path = os.path.join(ANALYSIS_OUTPUT, "merged_data.csv")
    filtered_path = os.path.join(ANALYSIS_OUTPUT, "filtered_data.csv")
    
    # Check for filtered data first (most processed)
    if check_file_exists(filtered_path, "filtered data"):
        print("Loading existing filtered data...")
        return pd.read_csv(filtered_path)
    
    # Check for merged data
    if check_file_exists(merged_path, "merged data"):
        print("Loading existing merged data and applying filtering...")
        df_merged = pd.read_csv(merged_path)
        
        # Apply preprocessing and filtering
        df_filtered = preprocess_dataframe(df_merged)
        df_filtered = drop_unwanted_columns(df_filtered)
        
        # Save filtered data for future use
        df_filtered.to_csv(filtered_path, index=False)
        print(f"Filtered data saved to {filtered_path}")
        return df_filtered
    
    # Process raw data from scratch
    print("Processing raw data from scratch...")
    df_merged = process_all_data()
    if df_merged is None:
        raise ValueError("Failed to process raw data.")
    
    # Apply preprocessing and filtering
    df_filtered = preprocess_dataframe(df_merged)
    df_filtered = drop_unwanted_columns(df_filtered)
    
    # Save filtered data
    df_filtered.to_csv(filtered_path, index=False)
    print(f"Filtered data saved to {filtered_path}")
    
    return df_filtered


def generate_model_performance_comparison_table(output_dir, phase_range=range(1, 13)):
    """
    Generate a comprehensive comparison table of all model performances.
    
    Args:
        output_dir: Directory containing model results
        phase_range: Range of phases to analyze
        
    Returns:
        pd.DataFrame: Comparison table with model performances
    """
    model_configs = [
        {"name": "Polynomial_Ridge", "suffix": "", "display_name": "Polynomial + Ridge"},
        {"name": "Gaussian_Process", "suffix": "_gpr", "display_name": "Gaussian Process"},
        {"name": "XGBoost", "suffix": "_xgb", "display_name": "XGBoost"},
        {"name": "LightGBM", "suffix": "_lgb", "display_name": "LightGBM"}
    ]
    
    comparison_results = []
    
    for model_config in model_configs:
        model_name = model_config["name"]
        suffix = model_config["suffix"]
        display_name = model_config["display_name"]
        
        summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}.csv")
        
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found for {display_name}: {summary_path}")
            continue
            
        df_summary = pd.read_csv(summary_path)
        
        # Calculate overall statistics across all phases
        overall_stats = {
            "Model": display_name,
            "Total_Models": df_summary["Num_Models"].sum(),
            "Total_R2_GTE_0.9": df_summary["Num_R2_GTE_0.9"].sum(),
            "Total_R2_GTE_0.8": df_summary["Num_R2_GTE_0.8"].sum(),
            "Total_R2_GTE_0.7": df_summary["Num_R2_GTE_0.7"].sum(),
            "Overall_Mean_R2": df_summary["Mean_R2_CV"].mean(),
            "Overall_Median_R2": df_summary["Median_R2_CV"].median(),
            "Best_Phase_R2": df_summary["Mean_R2_CV"].max(),
            "Worst_Phase_R2": df_summary["Mean_R2_CV"].min(),
            "R2_Std_Across_Phases": df_summary["Mean_R2_CV"].std()
        }
        
        # Calculate percentages
        if overall_stats["Total_Models"] > 0:
            overall_stats["Pct_R2_GTE_0.9"] = (overall_stats["Total_R2_GTE_0.9"] / overall_stats["Total_Models"]) * 100
            overall_stats["Pct_R2_GTE_0.8"] = (overall_stats["Total_R2_GTE_0.8"] / overall_stats["Total_Models"]) * 100
            overall_stats["Pct_R2_GTE_0.7"] = (overall_stats["Total_R2_GTE_0.7"] / overall_stats["Total_Models"]) * 100
        else:
            overall_stats["Pct_R2_GTE_0.9"] = 0
            overall_stats["Pct_R2_GTE_0.8"] = 0
            overall_stats["Pct_R2_GTE_0.7"] = 0
            
        comparison_results.append(overall_stats)
    
    # Convert to DataFrame and sort by overall mean R²
    df_comparison = pd.DataFrame(comparison_results)
    if not df_comparison.empty:
        df_comparison = df_comparison.sort_values("Overall_Mean_R2", ascending=False)
        
        # Round numerical columns for better readability
        numerical_cols = df_comparison.select_dtypes(include=[np.number]).columns
        df_comparison[numerical_cols] = df_comparison[numerical_cols].round(4)
        
        # Save to CSV
        output_path = os.path.join(output_dir, "model_performance_comparison.csv")
        df_comparison.to_csv(output_path, index=False)
        print(f" Model performance comparison saved to: {output_path}")
        
        # Create a summary table for paper
        paper_table = df_comparison[[
            "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", 
            "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7", 
            "Best_Phase_R2", "R2_Std_Across_Phases"
        ]].copy()
        
        # Rename columns for paper
        paper_table.columns = [
            "Model", "Total Models", "Mean R²", "Median R²", 
            "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7", 
            "Best Phase R²", "R² Std Dev"
        ]
        
        paper_output_path = os.path.join(output_dir, "model_performance_paper_table.csv")
        paper_table.to_csv(paper_output_path, index=False)
        print(f" Paper-ready table saved to: {paper_output_path}")
        
        # Print summary to console
        simulator_name = os.path.basename(output_dir).replace("analysis_output_", "").upper()
        print(f"\n" + "="*80)
        print(f"MODEL PERFORMANCE COMPARISON SUMMARY - {simulator_name}")
        print("="*80)
        print(paper_table.to_string(index=False))
        print("="*80)
        
    return df_comparison


def generate_cross_simulator_comparison_table():
    """
    Generate a comparison table between SITL and Gazebo simulators.
    
    Returns:
        pd.DataFrame: Cross-simulator comparison table
    """
    cross_comparison_results = []
    
    for sim_name, config in SIMULATOR_CONFIGS.items():
        analysis_output = config["analysis_output"]
        display_name = config["display_name"]
        
        # Load the model performance comparison for this simulator
        comparison_path = os.path.join(analysis_output, "model_performance_comparison.csv")
        if not os.path.exists(comparison_path):
            print(f"Warning: No performance comparison found for {display_name}: {comparison_path}")
            continue
            
        df_sim_comparison = pd.read_csv(comparison_path)
        
        # Add simulator information to each row
        df_sim_comparison["Simulator"] = sim_name
        cross_comparison_results.append(df_sim_comparison)
    
    if not cross_comparison_results:
        print("No simulator comparisons found.")
        return None
    
    # Combine all simulator results
    df_cross_comparison = pd.concat(cross_comparison_results, ignore_index=True)
    
    # Reorder columns to have Simulator first
    cols = df_cross_comparison.columns.tolist()
    cols = ["Simulator"] + [col for col in cols if col != "Simulator"]
    df_cross_comparison = df_cross_comparison[cols]
    
    # Save cross-simulator comparison
    cross_output_dir = "./analysis_output_combined"
    os.makedirs(cross_output_dir, exist_ok=True)
    
    output_path = os.path.join(cross_output_dir, "cross_simulator_model_comparison.csv")
    df_cross_comparison.to_csv(output_path, index=False)
    print(f" Cross-simulator comparison saved to: {output_path}")
    
    # Create paper-ready cross-simulator table
    paper_cross_table = df_cross_comparison[[
        "Simulator", "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", 
        "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7"
    ]].copy()
    
    # Rename columns for paper
    paper_cross_table.columns = [
        "Simulator", "Model", "Total Models", "Mean R²", "Median R²", 
        "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7"
    ]
    
    paper_cross_output_path = os.path.join(cross_output_dir, "cross_simulator_paper_table.csv")
    paper_cross_table.to_csv(paper_cross_output_path, index=False)
    print(f" Cross-simulator paper table saved to: {paper_cross_output_path}")
    
    # Print cross-simulator summary
    print(f"\n" + "="*100)
    print("CROSS-SIMULATOR MODEL PERFORMANCE COMPARISON")
    print("="*100)
    print(paper_cross_table.to_string(index=False))
    print("="*100)
    
    return df_cross_comparison


def plot_cross_simulator_comparison(cross_comparison_df=None):
    """
    Create visualizations comparing model performance across simulators.
    
    Args:
        cross_comparison_df: DataFrame with cross-simulator comparison data
    """
    if cross_comparison_df is None:
        # Try to load existing cross-comparison data
        cross_output_dir = "./analysis_output_combined"
        cross_path = os.path.join(cross_output_dir, "cross_simulator_model_comparison.csv")
        if os.path.exists(cross_path):
            cross_comparison_df = pd.read_csv(cross_path)
        else:
            print("No cross-simulator comparison data found.")
            return
    
    # Create output directory
    cross_output_dir = "./analysis_output_combined"
    os.makedirs(cross_output_dir, exist_ok=True)
    
    # 1. Mean R² comparison bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cross_comparison_df, x="Model", y="Overall_Mean_R2", hue="Simulator")
    plt.title("Model Performance Comparison: SITL vs Gazebo\n(Mean R² Score)", fontsize=14)
    plt.xlabel("Model Type")
    plt.ylabel("Mean R² Score")
    plt.xticks(rotation=45)
    plt.legend(title="Simulator")
    plt.tight_layout()
    
    plot_path = os.path.join(cross_output_dir, "cross_simulator_mean_r2_comparison.png")
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()
    print(f" Cross-simulator mean R² comparison saved to: {plot_path}")
    
    # 2. Percentage of high-performing models comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    thresholds = ["Pct_R2_GTE_0.7", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.9"]
    threshold_labels = ["% R² ≥ 0.7", "% R² ≥ 0.8", "% R² ≥ 0.9"]
    
    for idx, (threshold, label) in enumerate(zip(thresholds, threshold_labels)):
        sns.barplot(data=cross_comparison_df, x="Model", y=threshold, hue="Simulator", ax=axes[idx])
        axes[idx].set_title(f"{label}")
        axes[idx].set_xlabel("Model Type")
        axes[idx].set_ylabel("Percentage (%)")
        axes[idx].tick_params(axis='x', rotation=45)
        if idx == 0:
            axes[idx].legend(title="Simulator")
        else:
            axes[idx].legend().remove()
    
    plt.suptitle("Model Performance Distribution: SITL vs Gazebo", fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(cross_output_dir, "cross_simulator_performance_distribution.png")
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()
    print(f" Cross-simulator performance distribution saved to: {plot_path}")


def process_simulator_analysis(simulator_name):
    """
    Process analysis for a single simulator.
    
    Args:
        simulator_name: Name of the simulator ("SITL" or "Gazebo")
        
    Returns:
        pd.DataFrame: Filtered data for this simulator
    """
    config = SIMULATOR_CONFIGS[simulator_name]
    analysis_output = config["analysis_output"]
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {config['display_name'].upper()}")
    print(f"{'='*60}")
    
    # Step 1: Load or process data
    try:
        df_filtered = load_or_process_data(simulator_name)
    except Exception as e:
        print(f"Failed to load/process data for {simulator_name}: {e}")
        return None
    
    # Step 2: Correlation Analysis (skip if already done)
    print(f"\n--- Correlation Analysis for {simulator_name} ---")
    correlation_file = os.path.join(analysis_output, "Top_Correlated_Factors.csv")
    if check_file_exists(correlation_file, "correlation analysis"):
        print("Skipping correlation analysis...")
    else:
        analyze_top_correlations(df_filtered, output_csv=os.path.join(analysis_output, "Top_Correlated_Factors.csv"))
    
    # Step 3: Train Multiple Model Types
    print(f"\n--- Training Models for {simulator_name} ---")
    
    models_to_train = [
        {"trainer": PolynomialRidgeTrainer, "suffix": "", "name": "Polynomial + Ridge", "kwargs": {"degree": 2}},
        {"trainer": GaussianProcessTrainer, "suffix": "_gpr", "name": "Gaussian Process", "kwargs": {}},
        {"trainer": XGBoostTrainer, "suffix": "_xgb", "name": "XGBoost", "kwargs": {}},
        {"trainer": LightGBMTrainer, "suffix": "_lgb", "name": "LightGBM", "kwargs": {}}
    ]
    
    for model_config in models_to_train:
        print(f"\n--- Training {model_config['name']} Models for {simulator_name} ---")
        summary_file = os.path.join(analysis_output, f"multivariate_r2_summary{model_config['suffix']}.csv")
        
        if check_file_exists(summary_file, f"{model_config['name']} results"):
            print(f"Skipping {model_config['name']} training...")
        else:
            process_multivariate_analysis_with_trainer(
                df_filtered, 
                model_config["trainer"],
                analysis_output,
                suffix=model_config["suffix"],
                **model_config["kwargs"]
            )
    
    # Step 4: Generate Performance Comparison Tables
    print(f"\n--- Performance Analysis for {simulator_name} ---")
    generate_model_performance_comparison_table(analysis_output)
    generate_phase_wise_comparison_table(analysis_output)
    plot_model_comparison_heatmap(analysis_output)
    
    # Step 5: Generate Visualizations
    print(f"\n--- Generating Visualizations for {simulator_name} ---")
    
    # Plot R² histograms for all phases
    histogram_files = [
        "r2_histograms_all_phases.png",
        "r2_histograms_all_phases_xgb.png", 
        "r2_histograms_all_phases_lgb.png"
    ]
    
    suffixes = ["", "_xgb", "_lgb"]
    for suffix, hist_file in zip(suffixes, histogram_files):
        if not os.path.exists(os.path.join(analysis_output, hist_file)):
            plot_all_r2_histograms(analysis_output_dir=analysis_output, phase_range=range(1, 13), suffix=suffix)
    
    # Plot performance summaries
    performance_plots = [
        {"file": "multivariate_r2_summary.csv", "name": "Polynomial Ridge"},
        {"file": "multivariate_r2_summary_gpr.csv", "name": "Gaussian Process"},
        {"file": "multivariate_r2_summary_xgb.csv", "name": "XGBoost"},
        {"file": "multivariate_r2_summary_lgb.csv", "name": "LightGBM"}
    ]
    
    for plot_config in performance_plots:
        summary_path = os.path.join(analysis_output, plot_config["file"])
        if os.path.exists(summary_path):
            plot_model_performance_summary(summary_path, analysis_output, plot_config["name"])
    
    print(f" {simulator_name} analysis complete!")
    return df_filtered


def main(sitl_data_path=None, gazebo_data_path=None):
    """
    Main execution pipeline for the drone data analysis with dual simulator support.
    
    Args:
        sitl_data_path: Path to SITL data directory (optional, can be set in config)
        gazebo_data_path: Path to Gazebo data directory (optional, can be set in config)
    """
    print("Starting Dual-Simulator Drone Data Analysis Pipeline...")
    print("="*80)
    
    # Configure simulator paths
    configure_simulator_paths(sitl_data_path, gazebo_data_path)
    
    # Process each simulator
    simulator_results = {}
    
    for sim_name in ["SITL", "Gazebo"]:
        config = SIMULATOR_CONFIGS[sim_name]
        if not os.path.exists(config["data_root"]):
            print(f"  Skipping {sim_name}: Data directory not found - {config['data_root']}")
            continue
            
        df_filtered = process_simulator_analysis(sim_name)
        if df_filtered is not None:
            simulator_results[sim_name] = df_filtered
    
    # Cross-simulator comparison
    if len(simulator_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-SIMULATOR COMPARISON")
        print(f"{'='*80}")
        
        cross_comparison_df = generate_cross_simulator_comparison_table()
        if cross_comparison_df is not None:
            plot_cross_simulator_comparison(cross_comparison_df)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    for sim_name, config in SIMULATOR_CONFIGS.items():
        if sim_name in simulator_results:
            print(f" {config['display_name']} results: {config['analysis_output']}")
    
    if len(simulator_results) > 1:
        print(f" Cross-simulator comparison: ./analysis_output_combined")
    
    print(f"\n Key Output Files for Paper:")
    for sim_name, config in SIMULATOR_CONFIGS.items():
        if sim_name in simulator_results:
            analysis_output = config['analysis_output']
            print(f"\n{config['display_name']}:")
            print(f"  • Model Comparison: {os.path.join(analysis_output, 'model_performance_paper_table.csv')}")
            print(f"  • Phase-wise Analysis: {os.path.join(analysis_output, 'phase_wise_model_comparison.csv')}")
            print(f"  • Performance Heatmap: {os.path.join(analysis_output, 'model_comparison_heatmap.png')}")
    
    if len(simulator_results) > 1:
        print(f"\nCross-Simulator:")
        print(f"  • Combined Comparison: ./analysis_output_combined/cross_simulator_paper_table.csv")
        print(f"  • Performance Plots: ./analysis_output_combined/")


if __name__ == "__main__":
    # Example usage - update these paths to your actual data directories
    main(
        sitl_data_path="./data_sitl",      # Update this path
        gazebo_data_path="./data_gazebo"   # Update this path
    )0)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "model_comparison_heatmap.png")
    plt.savefig(heatmap_path, format="png", dpi=300)
    plt.close()
    print(f" Model comparison heatmap saved to: {heatmap_path}")


def main():
    """Main execution pipeline for the drone data analysis."""
    print("Starting Drone Simulation Data Analysis Pipeline...")
    ensure_output_directory()
    
    # Step 1: Load or process data (with skip logic)
    print("\n=== Step 1: Loading/Processing Data ===")
    try:
        df_filtered = load_or_process_data()
    except Exception as e:
        print(f"Failed to load/process data: {e}")
        return
    
    # Remove non-numeric columns before computing correlations
    df_numeric = df_filtered.select_dtypes(include=[np.number])
    
    # Step 2: Correlation Analysis (skip if already done)
    print("\n=== Step 2: Correlation Analysis ===")
    correlation_file = os.path.join(ANALYSIS_OUTPUT, "Top_Correlated_Factors.csv")
    if check_file_exists(correlation_file, "correlation analysis"):
        print("Skipping correlation analysis...")
    else:
        analyze_top_correlations(df_filtered)
    
    # Step 3: Train Multiple Model Types
    print("\n=== Step 3: Training Multiple Model Types ===")
    
    models_to_train = [
        {"trainer": PolynomialRidgeTrainer, "suffix": "", "name": "Polynomial + Ridge", "kwargs": {"degree": 2}},
        {"trainer": GaussianProcessTrainer, "suffix": "_gpr", "name": "Gaussian Process", "kwargs": {}},
        {"trainer": XGBoostTrainer, "suffix": "_xgb", "name": "XGBoost", "kwargs": {}},
        {"trainer": LightGBMTrainer, "suffix": "_lgb", "name": "LightGBM", "kwargs": {}}
    ]
    
    for model_config in models_to_train:
        print(f"\n--- Training {model_config['name']} Models ---")
        summary_file = os.path.join(ANALYSIS_OUTPUT, f"multivariate_r2_summary{model_config['suffix']}.csv")
        
        if check_file_exists(summary_file, f"{model_config['name']} results"):
            print(f"Skipping {model_config['name']} training...")
        else:
            process_multivariate_analysis_with_trainer(
                df_filtered, 
                model_config["trainer"], 
                suffix=model_config["suffix"],
                **model_config["kwargs"]
            )
    
    # Step 4: Generate Performance Comparison Tables
    print("\n=== Step 4: Performance Comparison Analysis ===")
    generate_model_performance_comparison_table(ANALYSIS_OUTPUT)
    generate_phase_wise_comparison_table(ANALYSIS_OUTPUT)
    plot_model_comparison_heatmap(ANALYSIS_OUTPUT)
    
    # Step 5: Generate Visualizations
    print("\n=== Step 5: Generating Visualizations ===")
    
    # Plot model fit overviews for selected phases (skip if exists)
    for phase in range(1, 13):
        overview_file = os.path.join(ANALYSIS_OUTPUT, f"phase{phase}_model_fit_overview.pdf")
        if not os.path.exists(overview_file):
            plot_phase_model_fits(df_filtered, ENV_FACTORS, phase=phase, output_dir=ANALYSIS_OUTPUT)
    
    # Plot R² histograms for all phases
    histogram_files = [
        "r2_histograms_all_phases.png",
        "r2_histograms_all_phases_xgb.png", 
        "r2_histograms_all_phases_lgb.png"
    ]
    
    suffixes = ["", "_xgb", "_lgb"]
    for suffix, hist_file in zip(suffixes, histogram_files):
        if not os.path.exists(os.path.join(ANALYSIS_OUTPUT, hist_file)):
            plot_all_r2_histograms(analysis_output_dir=ANALYSIS_OUTPUT, phase_range=range(1, 13), suffix=suffix)
    
    # Plot performance summaries
    performance_plots = [
        {"file": "multivariate_r2_summary.csv", "name": "Polynomial Ridge"},
        {"file": "multivariate_r2_summary_gpr.csv", "name": "Gaussian Process"},
        {"file": "multivariate_r2_summary_xgb.csv", "name": "XGBoost"},
        {"file": "multivariate_r2_summary_lgb.csv", "name": "LightGBM"}
    ]
    
    for plot_config in performance_plots:
        summary_path = os.path.join(ANALYSIS_OUTPUT, plot_config["file"])
        if os.path.exists(summary_path):
            plot_model_performance_summary(summary_path, ANALYSIS_OUTPUT, plot_config["name"])
    
    # Step 6: Advanced Analysis
    print("\n=== Step 6: Advanced Analysis ===")
    
    # Generate phase-to-state mapping tables (skip if exists)
    matrix_file = os.path.join(ANALYSIS_OUTPUT, "phase_to_states_matrix.csv")
    if not check_file_exists(matrix_file, "phase-to-state mapping"):
        generate_phase_state_table(ANALYSIS_OUTPUT)
        plot_phase_state_heatmap(matrix_file)
    
    # PCA analysis (skip if exists)
    pca_file = os.path.join(ANALYSIS_OUTPUT, "pca_env_factor_influence.csv")
    if not check_file_exists(pca_file, "PCA analysis"):
        analyze_env_factor_influence_via_pca(
            df_filtered,
            ENV_FACTORS,
            output_dir=ANALYSIS_OUTPUT,
            top_k=10
        )
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to: {ANALYSIS_OUTPUT}")
    print("\n Key Output Files for Paper:")
    print(f"  • Model Comparison Table: {os.path.join(ANALYSIS_OUTPUT, 'model_performance_paper_table.csv')}")
    print(f"  • Phase-wise Comparison: {os.path.join(ANALYSIS_OUTPUT, 'phase_wise_model_comparison.csv')}")
    print(f"  • Comparison Heatmap: {os.path.join(ANALYSIS_OUTPUT, 'model_comparison_heatmap.png')}")


if __name__ == "__main__":
    main()