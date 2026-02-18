import pandas as pd
import numpy as np

INPUT_CSV = "parameter_range.csv"

# Load parameter range file
df = pd.read_csv(INPUT_CSV, encoding='latin1')

# Parse range
def parse_range(range_str):
    if "or" in range_str.lower():
        return 0.0, 1.0
    range_str = range_str.replace("~", "–").replace("–", "-").replace("—", "-").strip()
    parts = range_str.rsplit("-", 1)
    return float(parts[0].strip()), float(parts[1].strip())

# Full factorial: multiply all Num Steps
full_factorial = np.prod(df["Num Steps"].astype(np.float64))

# Apply constraints
def is_mutable(param):
    if param in ["SIM_ACC2_RND", "SIM_GYR2_RND", "SIM_BAR2_RND",
                 "SIM_GPS2_NOISE", "SIM_GPS2_GLITCH_X", "SIM_GPS2_GLITCH_Y",
                 "SIM_GPS2_GLITCH_Z", "SIM_GPS2_DRFTALT"]:
        return False
    if "WIND_DIR" in param or "BARO_WCF" in param:
        return False
    if param in ["SIM_WIND_T_ALT", "SIM_WIND_T_COEF"]:
        return False
    if param == "SIM_WIND_TURB":  # even though independent, can limit for initial comparison
        return False
    if param == "SIM_WIND_T":
        return False
    return True

mutable_df = df[df["Parameter"].apply(is_mutable)]
constrained_factorial = np.prod(mutable_df["Num Steps"])

# Suggested LHS sample count
lhs_samples = 400

# Output summary
print(" Simulation Budget Comparison:")
print(f"  1. Full Factorial (no constraints):   {int(full_factorial):,} simulations")
print(f"  2. After Constraint Filtering:        {int(constrained_factorial):,} simulations")
print(f"  3. Latin Hypercube Sampling (LHS):    {lhs_samples:,} simulations")
print("\n Optimization Summary:")
print(f"    Reduction from full factorial to LHS: ~{full_factorial // lhs_samples:,}x fewer simulations")
print(f"    Reduction from constrained factorial to LHS: ~{constrained_factorial // lhs_samples:,}x fewer simulations")
