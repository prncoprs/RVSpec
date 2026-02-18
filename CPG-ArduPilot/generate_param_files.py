import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import qmc

# === CONFIG ===
INPUT_CSV = "parameter_range.csv"
SINGLE_OUTPUT_DIR = "tmp_single_parm_files"
MULTI_OUTPUT_DIR = "tmp_multi_parm_files"
NUM_LHS_SAMPLES = 400
SCALE_FACTOR = 2  # Shrink factor for multi-mutation

# === UTILITIES ===
def frange(start, stop, step):
    vals = []
    while start <= stop + 1e-8:
        vals.append(round(start, 6))
        start += step
    return vals

def parse_range(range_str):
    if "or" in range_str.lower():
        return 0.0, 1.0
    range_str = range_str.replace("~", "–").replace("–", "-").replace("—", "-").strip()
    parts = range_str.rsplit("-", 1)
    return float(parts[0].strip()), float(parts[1].strip())

# === LOAD PARAMETER TABLE ===
df = pd.read_csv(INPUT_CSV, encoding='latin1')
param_names = df["Parameter"].tolist()
param_defaults = dict(zip(df["Parameter"], df["Default"]))

param_values = {}
param_ranges = []
for _, row in df.iterrows():
    p = row["Parameter"]
    rmin, rmax = parse_range(row["Range"])
    step = float(row["Step"])
    vals = frange(rmin, rmax, step)
    if p == "SIM_WIND_DIR" and vals[-1] == 360.0:
        vals = vals[:-1]
    param_values[p] = vals
    param_ranges.append((rmin, rmax))

# === SINGLE MUTATION GENERATION ===
def is_valid_single(param, value):
    wind_dependents = [p for p in param_names if "WIND_DIR" in p or "BARO_WCF" in p]
    if param in wind_dependents or param in ["SIM_WIND_TURB", "SIM_WIND_T_ALT", "SIM_WIND_T_COEF", "SIM_WIND_T"]:
        return False
    if param in ["SIM_ACC2_RND", "SIM_GYR2_RND", "SIM_BAR2_RND", "SIM_BARO_DRIFT", "SIM_BAR2_DRIFT", "SIM_GPS_DRIFTALT",
                 "SIM_GPS2_NOISE", "SIM_GPS2_GLTCH_X", "SIM_GPS2_GLTCH_Y", "SIM_GPS2_GLTCH_Z",
                 "SIM_GPS2_DRFTALT"]:
        return False
    return True

def generate_single_mutation_files():
    os.makedirs(SINGLE_OUTPUT_DIR, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_count = 0
    default_config = {p: float(param_defaults[p]) for p in param_names}

    print(" Generating single-mutation parameter files...")
    for param in param_names:
        for value in param_values[param]:
            if not is_valid_single(param, value):
                continue
            if float(value) == float(param_defaults[param]):
                continue

            config = default_config.copy()
            config[param] = value

            # Mirror primaries into secondaries
            if param == "SIM_ACC1_RND": config["SIM_ACC2_RND"] = 0.0
            if param == "SIM_GYR1_RND": config["SIM_GYR2_RND"] = 0.0
            if param == "SIM_BARO_RND": config["SIM_BAR2_RND"] = 0.0
            if param == "SIM_GPS_NOISE": config["SIM_GPS2_NOISE"] = 0.0
            if param == "SIM_GPS_GLITCH_X": config["SIM_GPS2_GLTCH_X"] = 0.0
            if param == "SIM_GPS_GLITCH_Y": config["SIM_GPS2_GLTCH_Y"] = 0.0
            if param == "SIM_GPS_GLITCH_Z": config["SIM_GPS2_GLTCH_Z"] = 0.0
            if param == "SIM_GPS_DRIFTALT": config["SIM_GPS2_DRFTALT"] = 0.0

            if all(float(v) == float(param_defaults[k]) for k, v in config.items()):
                continue

            file_count += 1
            fname = f"{now_str}-single-{file_count:06d}.parm"
            with open(os.path.join(SINGLE_OUTPUT_DIR, fname), "w") as f:
                for k in param_names:
                    f.write(f"{k}\t{config[k]:.6f}\n")

    print(f" {file_count} single-mutation files saved in '{SINGLE_OUTPUT_DIR}'")

# === MULTI-MUTATION (LHS) GENERATION ===
def apply_constraints(sample):
    s = dict(zip(param_names, sample))
    
    # Apply shrinkage toward default
    for p in param_names:
        s[p] = float(param_defaults[p]) + (s[p] - float(param_defaults[p])) / SCALE_FACTOR

    # --- Constraint 1: Wind off — use default values
    if s.get("SIM_WIND_SPD", 0) == 0:
        for p in param_names:
            if "WIND_DIR" in p or "BARO_WCF" in p or p == "SIM_WIND_T":
                s[p] = float(param_defaults[p])

    # --- Constraint 2: Wind profile logic — use default if invalid
    if round(s.get("SIM_WIND_T", 0)) not in [0, 2]:
        s["SIM_WIND_T_ALT"] = float(param_defaults["SIM_WIND_T_ALT"])
        s["SIM_WIND_T_COEF"] = float(param_defaults["SIM_WIND_T_COEF"])

    # --- Constraint 3: Mirror sensors
    s["SIM_ACC2_RND"] = s["SIM_ACC1_RND"]
    s["SIM_GYR2_RND"] = s["SIM_GYR1_RND"]
    s["SIM_BAR2_RND"] = s["SIM_BARO_RND"]
    s["SIM_GPS2_NOISE"] = s["SIM_GPS_NOISE"]
    s["SIM_GPS2_GLTCH_X"] = s["SIM_GPS_GLITCH_X"]
    s["SIM_GPS2_GLTCH_Y"] = s["SIM_GPS_GLITCH_Y"]
    s["SIM_GPS2_GLTCH_Z"] = s["SIM_GPS_GLITCH_Z"]
    s["SIM_GPS2_DRFTALT"] = s["SIM_GPS_DRIFTALT"]

    # --- Constraint 4: Clamp wind direction to [0, 360)
    if s.get("SIM_WIND_DIR", 0) >= 360.0:
        s["SIM_WIND_DIR"] = 0.0

    # --- Constraint 5: Round enum for wind type
    s["SIM_WIND_T"] = round(s.get("SIM_WIND_T", 0))

    # --- Constraint 6: If RND or GLITCH mutated, force DRIFT back to default
    for param in param_names:
        if any(key in param for key in ["RND", "GLITCH"]):
            if abs(s[param] - float(param_defaults.get(param, 0))) > 1e-6:
                # Determine related DRIFT param
                if "ACC" in param:
                    s["SIM_BARO_DRIFT"] = float(param_defaults["SIM_BARO_DRIFT"])
                if "GYR" in param:
                    s["SIM_BARO_DRIFT"] = float(param_defaults["SIM_BARO_DRIFT"])
                if "BARO" in param:
                    s["SIM_BARO_DRIFT"] = float(param_defaults["SIM_BARO_DRIFT"])
                    s["SIM_BAR2_DRIFT"] = float(param_defaults["SIM_BAR2_DRIFT"])
                if "GPS" in param:
                    s["SIM_GPS_DRIFTALT"] = float(param_defaults["SIM_GPS_DRIFTALT"])
                    s["SIM_GPS2_DRFTALT"] = float(param_defaults["SIM_GPS2_DRFTALT"])
    
    return s

def generate_multi_mutation_files():
    os.makedirs(MULTI_OUTPUT_DIR, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sampler = qmc.LatinHypercube(d=len(param_ranges))
    lhs_samples = sampler.random(n=NUM_LHS_SAMPLES)
    scaled_samples = qmc.scale(lhs_samples, [r[0] for r in param_ranges], [r[1] for r in param_ranges])
    
    print(" Generating multi-mutation parameter files with LHS...")
    count = 0
    for i in range(NUM_LHS_SAMPLES):
        sample = scaled_samples[i]
        config = apply_constraints(sample)
        fname = f"{now_str}-multi-{i+1:06d}.parm"
        with open(os.path.join(MULTI_OUTPUT_DIR, fname), "w") as f:
            for k in param_names:
                f.write(f"{k}\t{config[k]:.6f}\n")
        count += 1

    print(f" {count} multi-mutation LHS files saved in '{MULTI_OUTPUT_DIR}'")

# === RUN ===
if __name__ == "__main__":
    generate_single_mutation_files()
    generate_multi_mutation_files()
