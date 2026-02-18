#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF Model Evaluation for CSS Sun Vector Error Prediction
Compares actual sun vector errors with Random Forest model predictions
"""

import argparse
import json
import math
import pickle
import re
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ============== File Reading Functions ==============
_FLOAT_RE = re.compile(r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|NAN)')

def read_numeric_cols(path, need=0, offset=0):
    """Read numeric columns from file"""
    rows = []
    try:
        with open(path, "r") as f:
            for ln in f:
                toks = _FLOAT_RE.findall(ln)
                if need == 0:
                    if not toks:
                        continue
                    vals = [float(t) if t.lower()!='nan' else math.nan for t in toks]
                    rows.append(vals)
                else:
                    if len(toks) < offset + need:
                        continue
                    sel = toks[offset:offset+need]
                    vals = [float(t) if t.lower()!='nan' else math.nan for t in sel]
                    rows.append(vals)
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return rows

# ============== Math Helper Functions ==============
def unit(v):
    """Normalize vector to unit length"""
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else [0.0, 0.0, 0.0]

def dot(a, b):
    """Dot product of two 3D vectors"""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def angle_deg(a, b):
    """Angle between two vectors in degrees"""
    aa, bb = unit(a), unit(b)
    c = max(-1.0, min(1.0, dot(aa, bb)))
    return math.degrees(math.acos(c))

def quat_to_cbn(q):
    """Convert quaternion to rotation matrix"""
    q1, q2, q3, q4 = q
    n = math.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
    if n == 0:
        return [[1,0,0], [0,1,0], [0,0,1]]
    q1, q2, q3, q4 = q1/n, q2/n, q3/n, q4/n
    
    C = [[0.0]*3 for _ in range(3)]
    C[0][0] = 1 - 2*(q2*q2 + q3*q3)
    C[0][1] = 2*(q1*q2 - q4*q3)
    C[0][2] = 2*(q1*q3 + q4*q2)
    C[1][0] = 2*(q1*q2 + q4*q3)
    C[1][1] = 1 - 2*(q1*q1 + q3*q3)
    C[1][2] = 2*(q2*q3 - q4*q1)
    C[2][0] = 2*(q1*q3 - q4*q2)
    C[2][1] = 2*(q2*q3 + q4*q1)
    C[2][2] = 1 - 2*(q1*q1 + q2*q2)
    return C

def matvec(C, v):
    """Matrix-vector multiplication"""
    return [C[0][0]*v[0]+C[0][1]*v[1]+C[0][2]*v[2],
            C[1][0]*v[0]+C[1][1]*v[1]+C[1][2]*v[2],
            C[2][0]*v[0]+C[2][1]*v[1]+C[2][2]*v[2]]

# ============== Parameter Parsing Functions ==============
def parse_sc_parameters(sc_path: Path) -> Dict:
    """Parse spacecraft parameters from SC_NOS3.txt"""
    params = {}
    
    with open(sc_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Mass - look for line with just a number followed by ! Mass
        if '!' in line and 'Mass' in line and 'Total' not in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['MASS'] = float(tokens[0])
        
        # MOI
        if 'Moments of Inertia' in line:
            # Look for the next 3 lines with numeric values
            for j in range(i+1, min(i+10, len(lines))):
                tokens = _FLOAT_RE.findall(lines[j])
                if len(tokens) >= 3:
                    params['MOI_XX'] = float(tokens[0])
                    params['MOI_YY'] = float(tokens[1])
                    params['MOI_ZZ'] = float(tokens[2])
                    break
        
        # Magnetic torquers - parse individual MTBs
        if 'MTB Parameters' in line:
            mtb_index = 0
            for j in range(i+1, min(i+30, len(lines))):
                if f'MTB {mtb_index}' in lines[j]:
                    # Look for saturation in the next few lines
                    for k in range(j+1, min(j+5, len(lines))):
                        if 'Saturation' in lines[k]:
                            tokens = _FLOAT_RE.findall(lines[k])
                            if tokens:
                                params[f'MTB_{mtb_index}_SATURATION'] = float(tokens[0])
                                mtb_index += 1
                                break
                    if mtb_index >= 3:  # We only need 3 MTBs
                        break
        
        # Magnetometer
        if 'Magnetometer' in line:
            for j in range(i+1, min(i+10, len(lines))):
                if 'Saturation' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['MAG_SATURATION'] = float(tokens[0])
                elif 'Quantization' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['MAG_QUANTIZATION'] = float(tokens[0])
                elif 'Noise' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['MAG_NOISE'] = float(tokens[0])
        
        # Gyroscope
        if 'Gyroscope' in line or 'Gyro' in line:
            for j in range(i+1, min(i+10, len(lines))):
                if 'Max Rate' in lines[j] or 'Maximum Rate' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['GYRO_MAX_RATE'] = float(tokens[0])
                elif 'Scale Factor Error' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['GYRO_SCALE_FACTOR_ERROR'] = float(tokens[0])
                elif 'Quantization' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['GYRO_QUANTIZATION'] = float(tokens[0])
                elif 'Angle Noise' in lines[j] or 'Noise' in lines[j]:
                    tokens = _FLOAT_RE.findall(lines[j])
                    if tokens:
                        params['GYRO_ANGLE_NOISE'] = float(tokens[0])
    
    # Set defaults for missing parameters
    defaults = {
        'MASS': 1.0,
        'MOI_XX': 0.001, 'MOI_YY': 0.001, 'MOI_ZZ': 0.001,
        'MTB_0_SATURATION': 0.01, 'MTB_1_SATURATION': 0.01, 'MTB_2_SATURATION': 0.01,
        'MAG_SATURATION': 0.0001,
        'MAG_QUANTIZATION': 1e-9,
        'MAG_NOISE': 1e-10,
        'GYRO_MAX_RATE': 1.0,
        'GYRO_SCALE_FACTOR_ERROR': 0.001,
        'GYRO_QUANTIZATION': 0.001,
        'GYRO_ANGLE_NOISE': 0.001
    }
    
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val
            print(f"Warning: {key} not found, using default: {default_val}")
    
    return params

def parse_orbital_parameters(orb_path: Path) -> Dict:
    """Parse orbital parameters from Orb_LEO.txt"""
    params = {}
    
    with open(orb_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Look for Keplerian elements
        if 'Periapsis' in line and 'km' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_PERIAPSIS_ALT'] = float(tokens[0])
        
        if 'Apoapsis' in line and 'km' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_APOAPSIS_ALT'] = float(tokens[0])
        
        if 'Inclination' in line and 'deg' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_INCLINATION'] = float(tokens[0])
        
        if 'RAAN' in line or 'Right Ascension' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_RAAN'] = float(tokens[0])
        
        if 'Arg' in line and 'Periapsis' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_ARG_PERIAPSIS'] = float(tokens[0])
        
        if 'True Anomaly' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['ORB_TRUE_ANOMALY'] = float(tokens[0])
    
    # Set defaults
    defaults = {
        'ORB_PERIAPSIS_ALT': 400.0,
        'ORB_APOAPSIS_ALT': 400.0,
        'ORB_INCLINATION': 51.6,
        'ORB_RAAN': 0.0,
        'ORB_ARG_PERIAPSIS': 0.0,
        'ORB_TRUE_ANOMALY': 0.0
    }
    
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val
            print(f"Warning: {key} not found, using default: {default_val}")
    
    return params

def parse_sim_parameters(inp_sim_path: Path) -> Dict:
    """Parse simulation parameters from Inp_Sim.txt"""
    params = {}
    
    with open(inp_sim_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # F10.7 and Ap indices usually in atmospheric/environment section
        if 'F10.7' in line or 'F107' in line:
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['F10_7_FLUX'] = float(tokens[0])
        
        # Look for Ap index - handle "USER-provided Ap" format
        if ('Ap' in line and 'USER' in line) or ('Ap' in line and 'index' in line.lower()):
            tokens = _FLOAT_RE.findall(line)
            if tokens:
                params['AP_INDEX'] = float(tokens[0])
    
    # Set defaults for space weather
    defaults = {
        'F10_7_FLUX': 150.0,
        'AP_INDEX': 15.0
    }
    
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val
            print(f"Warning: {key} not found, using default: {default_val}")
    
    return params

def parse_css_axes():
    """Parse CSS axes from SC configuration (default 6 CSS)"""
    # Default configuration: ±X, ±Y, ±Z
    axes = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    return [unit(a) for a in axes]

def css_reconstruct_sun_direction(intensities, axes_b):
    """Reconstruct sun direction from CSS measurements"""
    wsum = [0.0, 0.0, 0.0]
    total = 0.0
    
    for intensity, axis in zip(intensities, axes_b):
        if intensity > 0.0:
            wsum[0] += intensity * axis[0]
            wsum[1] += intensity * axis[1]
            wsum[2] += intensity * axis[2]
            total += intensity
    
    if total <= 0.0:
        return None
    return unit(wsum)

# ============== Main Evaluation Class ==============
class RFModelEvaluator:
    def __init__(self, data_dir: str, model_path: str, output_dir: str, model_type: str = 'mean'):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_type = model_type  # Store model type
        self.css_axes = parse_css_axes()
        self.rf_model = None
        self.static_params = {}
        self.results = []
        
        # Feature columns order (must match training)
        self.feature_columns = [
            'MASS', 'MOI_XX', 'MOI_YY', 'MOI_ZZ',
            'MTB_0_SATURATION', 'MTB_1_SATURATION', 'MTB_2_SATURATION',
            'MAG_SATURATION', 'MAG_QUANTIZATION', 'MAG_NOISE',
            'GYRO_MAX_RATE', 'GYRO_SCALE_FACTOR_ERROR', 'GYRO_QUANTIZATION', 'GYRO_ANGLE_NOISE',
            'ORB_PERIAPSIS_ALT', 'ORB_APOAPSIS_ALT', 'ORB_INCLINATION', 
            'ORB_RAAN', 'ORB_ARG_PERIAPSIS', 'ORB_TRUE_ANOMALY',
            'F10_7_FLUX', 'AP_INDEX',
            'global_phase_angle', 'is_daylight'
        ]
    
    def load_rf_model(self):
        """Load the trained Random Forest model"""
        print(f"Loading RF model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            print("RF model loaded successfully")
        except Exception as e:
            print(f"Error loading RF model: {e}")
            sys.exit(1)
    
    def parse_all_parameters(self):
        """Parse all static parameters from configuration files"""
        nos3_dir = self.data_dir / "NOS3InOut"
        
        print("Parsing spacecraft parameters...")
        sc_path = nos3_dir / "SC_NOS3.txt"
        if not sc_path.exists():
            sc_path = nos3_dir / "SC_SensorFOV.txt"
        if sc_path.exists():
            sc_params = parse_sc_parameters(sc_path)
            self.static_params.update(sc_params)
        
        print("Parsing orbital parameters...")
        orb_path = nos3_dir / "Orb_LEO.txt"
        if orb_path.exists():
            orb_params = parse_orbital_parameters(orb_path)
            self.static_params.update(orb_params)
        
        print("Parsing simulation parameters...")
        inp_sim_path = nos3_dir / "Inp_Sim.txt"
        if inp_sim_path.exists():
            sim_params = parse_sim_parameters(inp_sim_path)
            self.static_params.update(sim_params)
        
        print(f"Loaded {len(self.static_params)} static parameters")
    
    def prepare_features(self, global_phase_angle: float, is_daylight: float) -> np.ndarray:
        """Prepare feature vector for RF model"""
        features = []
        for col in self.feature_columns:
            if col == 'global_phase_angle':
                features.append(global_phase_angle)
            elif col == 'is_daylight':
                features.append(is_daylight)
            else:
                features.append(self.static_params.get(col, 0.0))
        return np.array(features).reshape(1, -1)
    
    def process_timesteps(self):
        """Process all timesteps and compare with RF predictions"""
        nos3_dir = self.data_dir / "NOS3InOut"
        
        print("Loading simulation data...")
        
        # Load data files
        pos_n = read_numeric_cols(nos3_dir / "PosN.42", need=3, offset=0)
        svn = read_numeric_cols(nos3_dir / "svn.42", need=3, offset=0)
        svb = read_numeric_cols(nos3_dir / "svb.42", need=3, offset=0)
        qbn = read_numeric_cols(nos3_dir / "qbn.42", need=4, offset=0)
        albedo = read_numeric_cols(nos3_dir / "Albedo.42", need=0, offset=0)
        illum = read_numeric_cols(nos3_dir / "Illum.42", need=0, offset=0)
        
        # Find minimum common length
        min_len = min(len(pos_n), len(svn), len(svb), len(qbn), len(albedo), len(illum))
        print(f"Processing {min_len} timesteps...")
        
        # Statistics tracking
        non_zero_error_count = 0
        comparison_true_count = 0
        valid_comparisons = 0
        
        # Daylight-specific statistics
        daylight_count = 0
        daylight_non_zero_error = 0
        daylight_comparison_true = 0
        daylight_valid_comparisons = 0
        
        # Progress tracking
        progress_interval = max(1, min_len // 20)  # Show progress 20 times
        
        for t in range(min_len):
            # Show progress
            if t % progress_interval == 0 or t == min_len - 1:
                progress_pct = (t + 1) / min_len * 100
                print(f"  Progress: {t+1}/{min_len} ({progress_pct:.1f}%)", end='\r')
            
            # Calculate dynamic features
            pos = pos_n[t]
            sv_n = svn[t]
            sv_b_truth = unit(svb[t])
            
            # Global phase angle
            global_phase = angle_deg(sv_n, pos)
            
            # Is daylight
            is_day = 1.0 if dot(unit(sv_n), unit(pos)) > 0 else 0.0
            
            # Track daylight timesteps
            if is_day > 0.5:
                daylight_count += 1
            
            # Prepare features for RF model
            features = self.prepare_features(global_phase, is_day)
            
            # Get RF prediction for error_reduction
            rf_predicted_reduction = self.rf_model.predict(features)[0]
            
            # Get CSS measurements
            illum_row = illum[t][:6] if len(illum[t]) >= 6 else illum[t] + [0.0]*(6-len(illum[t]))
            albedo_row = albedo[t][:6] if len(albedo[t]) >= 6 else albedo[t] + [0.0]*(6-len(albedo[t]))
            
            # Reconstruct sun vector with albedo
            sv_b_measured = css_reconstruct_sun_direction(illum_row, self.css_axes)
            
            if sv_b_measured is not None:
                # Calculate actual sun vector error
                actual_error = angle_deg(sv_b_measured, sv_b_truth)
                
                # Track non-zero errors
                if actual_error > 0.001:  # Consider errors > 0.001 degrees as non-zero
                    non_zero_error_count += 1
                    if is_day > 0.5:
                        daylight_non_zero_error += 1
                
                # Reconstruct sun vector without albedo (ideal case)
                no_albedo_illum = [max(0, illum_row[i] - albedo_row[i]) for i in range(6)]
                sv_b_no_albedo = css_reconstruct_sun_direction(no_albedo_illum, self.css_axes)
                
                if sv_b_no_albedo is not None:
                    error_no_albedo = angle_deg(sv_b_no_albedo, sv_b_truth)
                    actual_reduction = actual_error - error_no_albedo
                else:
                    error_no_albedo = np.nan
                    actual_reduction = np.nan
                
                # Check if actual error is less than predicted
                error_less_than_predicted = actual_error < abs(rf_predicted_reduction)
                
                # Track comparison statistics
                if not np.isnan(actual_error):
                    valid_comparisons += 1
                    if error_less_than_predicted:
                        comparison_true_count += 1
                    
                    # Track daylight-specific comparisons
                    if is_day > 0.5:
                        daylight_valid_comparisons += 1
                        if error_less_than_predicted:
                            daylight_comparison_true += 1
                
                # Store results
                self.results.append({
                    'timestep': t,
                    'global_phase_angle': global_phase,
                    'is_daylight': is_day,
                    'actual_error': actual_error,
                    'error_no_albedo': error_no_albedo,
                    'actual_reduction': actual_reduction,
                    'rf_predicted_reduction': rf_predicted_reduction,
                    'prediction_error': rf_predicted_reduction - actual_reduction if not np.isnan(actual_reduction) else np.nan,
                    'error_less_than_predicted': error_less_than_predicted
                })
        
        # Clear the progress line
        print(" " * 50, end='\r')
        
        # Print summary statistics
        print(f"Processing complete!")
        print(f"\nQuick Statistics:")
        print(f"  Total timesteps: {min_len}")
        print(f"  Daylight timesteps: {daylight_count}/{min_len} ({daylight_count/min_len*100:.1f}%)")
        print(f"\n  ALL CONDITIONS:")
        print(f"    Non-zero actual errors: {non_zero_error_count}/{min_len} ({non_zero_error_count/min_len*100:.1f}%)")
        if valid_comparisons > 0:
            print(f"    |Actual error| < |RF predicted|: {comparison_true_count}/{valid_comparisons} ({comparison_true_count/valid_comparisons*100:.1f}%)")
        
        print(f"\n  DAYLIGHT ONLY:")
        if daylight_count > 0:
            print(f"    Non-zero actual errors: {daylight_non_zero_error}/{daylight_count} ({daylight_non_zero_error/daylight_count*100:.1f}%)")
            if daylight_valid_comparisons > 0:
                print(f"    |Actual error| < |RF predicted|: {daylight_comparison_true}/{daylight_valid_comparisons} ({daylight_comparison_true/daylight_valid_comparisons*100:.1f}%)")
    
    def analyze_results(self):
        """Analyze the comparison results"""
        df = pd.DataFrame(self.results)
        
        # Remove NaN values
        df_clean = df.dropna(subset=['actual_reduction', 'prediction_error'])
        
        # Separate daylight data
        df_daylight = df[df['is_daylight'] > 0.5]
        df_daylight_clean = df_daylight.dropna(subset=['actual_reduction', 'prediction_error'])
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        # Model type header
        if self.model_type == 'quantile':
            print("Model Type: Quantile RF (95th percentile)")
            print("Interpretation: Predictions represent conservative upper bounds")
        else:
            print("Model Type: Mean RF")
            print("Interpretation: Predictions represent expected average values")
        
        # Basic statistics
        print(f"\nTotal timesteps: {len(df)}")
        print(f"Daylight timesteps: {len(df_daylight)} ({len(df_daylight)/len(df)*100:.1f}%)")
        print(f"Valid comparisons: {len(df_clean)}")
        
        # Count non-zero errors
        non_zero_errors = df[df['actual_error'] > 0.001]
        non_zero_errors_daylight = df_daylight[df_daylight['actual_error'] > 0.001]
        
        print(f"\n** KEY STATISTICS - ALL CONDITIONS **")
        print(f"Non-zero actual errors: {len(non_zero_errors)}/{len(df)} ({len(non_zero_errors)/len(df)*100:.1f}%)")
        
        # Comparison: actual error vs predicted reduction
        comparison_valid = df['error_less_than_predicted'].dropna()
        true_count = comparison_valid.sum()
        false_count = len(comparison_valid) - true_count
        
        if self.model_type == 'quantile':
            print(f"|Actual error| < 95th percentile bound: {true_count}/{len(comparison_valid)} ({true_count/len(comparison_valid)*100:.1f}%)")
            print(f"  Expected: ~95% (this is a conservative safety bound)")
        else:
            print(f"|Actual error| < |RF predicted mean|: {true_count}/{len(comparison_valid)} ({true_count/len(comparison_valid)*100:.1f}%)")
        
        print(f"  - True: {true_count} cases")
        print(f"  - False: {false_count} cases")
        
        print(f"\n** KEY STATISTICS - DAYLIGHT ONLY **")
        if len(df_daylight) > 0:
            print(f"Non-zero actual errors: {len(non_zero_errors_daylight)}/{len(df_daylight)} ({len(non_zero_errors_daylight)/len(df_daylight)*100:.1f}%)")
            
            # Daylight comparison
            comparison_valid_daylight = df_daylight['error_less_than_predicted'].dropna()
            if len(comparison_valid_daylight) > 0:
                true_count_daylight = comparison_valid_daylight.sum()
                false_count_daylight = len(comparison_valid_daylight) - true_count_daylight
                
                if self.model_type == 'quantile':
                    print(f"|Actual error| < 95th percentile bound: {true_count_daylight}/{len(comparison_valid_daylight)} ({true_count_daylight/len(comparison_valid_daylight)*100:.1f}%)")
                else:
                    print(f"|Actual error| < |RF predicted mean|: {true_count_daylight}/{len(comparison_valid_daylight)} ({true_count_daylight/len(comparison_valid_daylight)*100:.1f}%)")
                print(f"  - True: {true_count_daylight} cases")
                print(f"  - False: {false_count_daylight} cases")
        
        # Prediction accuracy
        mae = np.mean(np.abs(df_clean['prediction_error']))
        rmse = np.sqrt(np.mean(df_clean['prediction_error']**2))
        
        print(f"\nPrediction Accuracy (All Conditions):")
        if self.model_type == 'quantile':
            print(f"  Note: MAE/RMSE less meaningful for quantile predictions")
        print(f"  MAE: {mae:.4f}°")
        print(f"  RMSE: {rmse:.4f}°")
        
        if len(df_daylight_clean) > 0:
            mae_daylight = np.mean(np.abs(df_daylight_clean['prediction_error']))
            rmse_daylight = np.sqrt(np.mean(df_daylight_clean['prediction_error']**2))
            
            print(f"\nPrediction Accuracy (Daylight Only):")
            print(f"  MAE: {mae_daylight:.4f}°")
            print(f"  RMSE: {rmse_daylight:.4f}°")
        
        # Error statistics
        print(f"\nActual Sun Vector Error (All Conditions):")
        print(f"  Mean: {df['actual_error'].mean():.3f}°")
        print(f"  Std: {df['actual_error'].std():.3f}°")
        print(f"  Min: {df['actual_error'].min():.3f}°")
        print(f"  Max: {df['actual_error'].max():.3f}°")
        print(f"  Median: {df['actual_error'].median():.3f}°")
        
        if len(df_daylight) > 0:
            print(f"\nActual Sun Vector Error (Daylight Only):")
            print(f"  Mean: {df_daylight['actual_error'].mean():.3f}°")
            print(f"  Std: {df_daylight['actual_error'].std():.3f}°")
            print(f"  Min: {df_daylight['actual_error'].min():.3f}°")
            print(f"  Max: {df_daylight['actual_error'].max():.3f}°")
            print(f"  Median: {df_daylight['actual_error'].median():.3f}°")
        
        if self.model_type == 'quantile':
            print(f"\nRF Predicted 95th Percentile (All Conditions):")
        else:
            print(f"\nRF Predicted Reduction (All Conditions):")
        print(f"  Mean: {df['rf_predicted_reduction'].mean():.3f}°")
        print(f"  Std: {df['rf_predicted_reduction'].std():.3f}°")
        print(f"  Min: {df['rf_predicted_reduction'].min():.3f}°")
        print(f"  Max: {df['rf_predicted_reduction'].max():.3f}°")
        
        # Save results
        df.to_csv(self.output_dir / "evaluation_results.csv", index=False)
        
        return df
    
    def generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots"""
        print("\nGenerating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3)
        
        # Separate daylight data
        df_daylight = df[df['is_daylight'] > 0.5]
        
        # Plot 1: Actual vs Predicted Reduction
        ax1 = fig.add_subplot(gs[0, 0])
        df_clean = df.dropna(subset=['actual_reduction'])
        ax1.scatter(df_clean['actual_reduction'], df_clean['rf_predicted_reduction'], 
                   alpha=0.5, s=10, label='All')
        
        # Add diagonal line
        min_val = min(df_clean['actual_reduction'].min(), df_clean['rf_predicted_reduction'].min())
        max_val = max(df_clean['actual_reduction'].max(), df_clean['rf_predicted_reduction'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax1.set_xlabel('Actual Error Reduction (°)')
        ax1.set_ylabel('RF Predicted Reduction (°)')
        ax1.set_title('Actual vs Predicted Error Reduction (All)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Prediction Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        df_clean = df.dropna(subset=['prediction_error'])
        ax2.hist(df_clean['prediction_error'], bins=50, edgecolor='black', alpha=0.7, label='All')
        if len(df_daylight) > 0:
            df_daylight_clean = df_daylight.dropna(subset=['prediction_error'])
            ax2.hist(df_daylight_clean['prediction_error'], bins=50, alpha=0.5, label='Daylight', color='orange')
        ax2.set_xlabel('Prediction Error (°)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Error Distribution')
        ax2.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison Results (All vs Daylight)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Prepare data for grouped bar chart
        categories = ['All Conditions', 'Daylight Only']
        true_counts = []
        false_counts = []
        
        # All conditions
        comparison_all = df['error_less_than_predicted'].dropna()
        true_all = comparison_all.sum()
        false_all = len(comparison_all) - true_all
        true_counts.append(true_all)
        false_counts.append(false_all)
        
        # Daylight only
        if len(df_daylight) > 0:
            comparison_day = df_daylight['error_less_than_predicted'].dropna()
            true_day = comparison_day.sum()
            false_day = len(comparison_day) - true_day
            true_counts.append(true_day)
            false_counts.append(false_day)
        else:
            true_counts.append(0)
            false_counts.append(0)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, true_counts, width, label='True', color='green', alpha=0.7)
        bars2 = ax3.bar(x + width/2, false_counts, width, label='False', color='red', alpha=0.7)
        
        ax3.set_xlabel('Condition')
        ax3.set_ylabel('Count')
        ax3.set_title('Comparison: Actual Error < |RF Predicted|')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            total = true_counts[i] + false_counts[i]
            if total > 0:
                ax3.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                        f'{true_counts[i]}\n({true_counts[i]/total*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=8)
                ax3.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                        f'{false_counts[i]}\n({false_counts[i]/total*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Time Series of Errors
        ax4 = fig.add_subplot(gs[1, 0])
        sample_size = min(500, len(df))
        if sample_size > 0:
            indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
            ax4.plot(indices, df.iloc[indices]['actual_error'], 
                    alpha=0.7, label='Actual Error', linewidth=0.8)
            ax4.plot(indices, np.abs(df.iloc[indices]['rf_predicted_reduction']), 
                    alpha=0.7, label='|RF Predicted|', linewidth=0.8)
            
            # Mark daylight periods
            daylight_mask = df.iloc[indices]['is_daylight'] > 0.5
            ax4.fill_between(indices, 0, ax4.get_ylim()[1], where=daylight_mask, 
                            alpha=0.1, color='yellow', label='Daylight')
            
            ax4.set_xlabel('Timestep (sampled)')
            ax4.set_ylabel('Error (°)')
            ax4.set_title('Error Time Series')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Error vs Phase Angle (with daylight highlighting)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Eclipse data
        df_eclipse = df[df['is_daylight'] <= 0.5]
        if len(df_eclipse) > 0:
            ax5.scatter(df_eclipse['global_phase_angle'], df_eclipse['actual_error'], 
                       alpha=0.3, s=5, label='Actual (Eclipse)', color='blue')
        
        # Daylight data
        if len(df_daylight) > 0:
            ax5.scatter(df_daylight['global_phase_angle'], df_daylight['actual_error'], 
                       alpha=0.3, s=5, label='Actual (Daylight)', color='orange')
        
        ax5.set_xlabel('Global Phase Angle (°)')
        ax5.set_ylabel('Error (°)')
        ax5.set_title('Error vs Phase Angle by Lighting Condition')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics Table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Calculate statistics
        mae = np.nanmean(np.abs(df['prediction_error']))
        rmse = np.sqrt(np.nanmean(df['prediction_error']**2))
        comparison_valid = df['error_less_than_predicted'].dropna()
        accuracy = comparison_valid.sum() / len(comparison_valid) * 100 if len(comparison_valid) > 0 else 0
        
        # Daylight statistics
        if len(df_daylight) > 0:
            df_daylight_clean = df_daylight.dropna(subset=['prediction_error'])
            mae_day = np.nanmean(np.abs(df_daylight_clean['prediction_error'])) if len(df_daylight_clean) > 0 else np.nan
            comparison_day = df_daylight['error_less_than_predicted'].dropna()
            accuracy_day = comparison_day.sum() / len(comparison_day) * 100 if len(comparison_day) > 0 else 0
        else:
            mae_day = np.nan
            accuracy_day = 0
        
        table_data = [
            ['Metric', 'All', 'Daylight'],
            ['Total Timesteps', f'{len(df)}', f'{len(df_daylight)}'],
            ['Non-zero Errors', f'{len(df[df["actual_error"] > 0.001])}', f'{len(df_daylight[df_daylight["actual_error"] > 0.001])}' if len(df_daylight) > 0 else '0'],
            ['MAE', f'{mae:.4f}°', f'{mae_day:.4f}°' if not np.isnan(mae_day) else 'N/A'],
            ['RMSE', f'{rmse:.4f}°', 'N/A'],
            ['Comparison Accuracy', f'{accuracy:.1f}%', f'{accuracy_day:.1f}%'],
            ['Mean Actual Error', f'{df["actual_error"].mean():.3f}°', f'{df_daylight["actual_error"].mean():.3f}°' if len(df_daylight) > 0 else 'N/A']
        ]
        
        table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Format header
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Plot 7: Daylight vs Eclipse Error Comparison
        ax7 = fig.add_subplot(gs[2, 0])
        
        data_to_plot = []
        labels = []
        
        # All conditions
        if len(df) > 0:
            data_to_plot.append(df['actual_error'].dropna())
            labels.append('All')
        
        # Daylight
        if len(df_daylight) > 0:
            data_to_plot.append(df_daylight['actual_error'].dropna())
            labels.append('Daylight')
        
        # Eclipse
        df_eclipse = df[df['is_daylight'] <= 0.5]
        if len(df_eclipse) > 0:
            data_to_plot.append(df_eclipse['actual_error'].dropna())
            labels.append('Eclipse')
        
        if data_to_plot:
            ax7.boxplot(data_to_plot, labels=labels)
            ax7.set_ylabel('Actual Error (°)')
            ax7.set_title('Error Distribution by Lighting Condition')
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Prediction Accuracy Scatter (Daylight highlighted)
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Eclipse points
        df_eclipse_clean = df_eclipse.dropna(subset=['actual_reduction']) if len(df_eclipse) > 0 else pd.DataFrame()
        if len(df_eclipse_clean) > 0:
            ax8.scatter(df_eclipse_clean['actual_reduction'], df_eclipse_clean['rf_predicted_reduction'], 
                       alpha=0.5, s=10, label='Eclipse', color='blue')
        
        # Daylight points
        df_daylight_clean = df_daylight.dropna(subset=['actual_reduction']) if len(df_daylight) > 0 else pd.DataFrame()
        if len(df_daylight_clean) > 0:
            ax8.scatter(df_daylight_clean['actual_reduction'], df_daylight_clean['rf_predicted_reduction'], 
                       alpha=0.5, s=10, label='Daylight', color='orange')
        
        # Add diagonal line
        if len(df_eclipse_clean) > 0 or len(df_daylight_clean) > 0:
            all_data = pd.concat([df_eclipse_clean, df_daylight_clean])
            min_val = min(all_data['actual_reduction'].min(), all_data['rf_predicted_reduction'].min())
            max_val = max(all_data['actual_reduction'].max(), all_data['rf_predicted_reduction'].max())
            ax8.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax8.set_xlabel('Actual Error Reduction (°)')
        ax8.set_ylabel('RF Predicted Reduction (°)')
        ax8.set_title('Prediction Accuracy by Lighting')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Performance Metrics Bar Chart
        ax9 = fig.add_subplot(gs[2, 2])
        
        metrics = ['Non-zero\nErrors (%)', 'Comparison\nAccuracy (%)']
        all_values = []
        daylight_values = []
        
        # Non-zero error percentage
        non_zero_pct = len(df[df['actual_error'] > 0.001]) / len(df) * 100 if len(df) > 0 else 0
        all_values.append(non_zero_pct)
        
        if len(df_daylight) > 0:
            non_zero_pct_day = len(df_daylight[df_daylight['actual_error'] > 0.001]) / len(df_daylight) * 100
            daylight_values.append(non_zero_pct_day)
        else:
            daylight_values.append(0)
        
        # Comparison accuracy
        all_values.append(accuracy)
        daylight_values.append(accuracy_day)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax9.bar(x - width/2, all_values, width, label='All', color='blue', alpha=0.7)
        bars2 = ax9.bar(x + width/2, daylight_values, width, label='Daylight', color='orange', alpha=0.7)
        
        ax9.set_ylabel('Percentage (%)')
        ax9.set_title('Performance Metrics Comparison')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Add value labels
        for bar1, bar2, val1, val2 in zip(bars1, bars2, all_values, daylight_values):
            ax9.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                    f'{val1:.1f}', ha='center', va='bottom', fontsize=8)
            ax9.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                    f'{val2:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('RF Model Evaluation: Sun Vector Error Prediction (All vs Daylight)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "rf_evaluation_plots.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def generate_report(self, df: pd.DataFrame):
        """Generate comprehensive text report"""
        report_path = self.output_dir / "rf_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RF MODEL EVALUATION REPORT\n")
            if self.model_type == 'quantile':
                f.write("Model Type: Quantile Random Forest (95th Percentile)\n")
                f.write("Sun Vector Error vs Conservative Upper Bound Comparison\n")
            else:
                f.write("Model Type: Mean Random Forest\n")
                f.write("Sun Vector Error vs Mean Prediction Comparison\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("STATIC PARAMETERS\n")
            f.write("-"*60 + "\n")
            for key, value in self.static_params.items():
                f.write(f"{key:<30} {value:.6f}\n")
            
            f.write("\n")
            f.write("EVALUATION RESULTS\n")
            f.write("-"*60 + "\n")
            f.write(f"Total timesteps analyzed: {len(df)}\n")
            f.write(f"Valid comparisons: {len(df.dropna(subset=['actual_reduction']))}\n\n")
            
            # Prediction accuracy
            df_clean = df.dropna(subset=['prediction_error'])
            if len(df_clean) > 0:
                mae = np.mean(np.abs(df_clean['prediction_error']))
                rmse = np.sqrt(np.mean(df_clean['prediction_error']**2))
                
                f.write("PREDICTION ACCURACY\n")
                if self.model_type == 'quantile':
                    f.write("(Note: For quantile models, these metrics are less meaningful)\n")
                f.write(f"Mean Absolute Error: {mae:.4f}°\n")
                f.write(f"Root Mean Square Error: {rmse:.4f}°\n\n")
            
            # Comparison results
            comparison_valid = df['error_less_than_predicted'].dropna()
            if len(comparison_valid) > 0:
                true_count = comparison_valid.sum()
                false_count = len(comparison_valid) - true_count
                
                if self.model_type == 'quantile':
                    f.write("COMPARISON: |Actual Error| < 95th Percentile Bound\n")
                    f.write("Expected: ~95% should be True (conservative safety bound)\n")
                else:
                    f.write("COMPARISON: |Actual Error| < |RF Predicted Mean|\n")
                
                f.write(f"True:  {true_count:5d} ({true_count/len(comparison_valid)*100:6.2f}%)\n")
                f.write(f"False: {false_count:5d} ({false_count/len(comparison_valid)*100:6.2f}%)\n\n")
                
                if self.model_type == 'quantile':
                    coverage = true_count/len(comparison_valid)*100
                    if coverage >= 93 and coverage <= 97:
                        f.write(" Model is well-calibrated (coverage close to 95%)\n\n")
                    elif coverage < 93:
                        f.write(" Model may be overconfident (coverage < 93%)\n\n")
                    else:
                        f.write(" Model may be too conservative (coverage > 97%)\n\n")
            
            # Error statistics
            f.write("ERROR STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write("Actual Sun Vector Error:\n")
            f.write(f"  Mean:   {df['actual_error'].mean():8.4f}°\n")
            f.write(f"  Std:    {df['actual_error'].std():8.4f}°\n")
            f.write(f"  Min:    {df['actual_error'].min():8.4f}°\n")
            f.write(f"  Max:    {df['actual_error'].max():8.4f}°\n")
            f.write(f"  Median: {df['actual_error'].median():8.4f}°\n\n")
            
            if self.model_type == 'quantile':
                f.write("RF Predicted 95th Percentile:\n")
            else:
                f.write("RF Predicted Error Reduction:\n")
            f.write(f"  Mean:   {df['rf_predicted_reduction'].mean():8.4f}°\n")
            f.write(f"  Std:    {df['rf_predicted_reduction'].std():8.4f}°\n")
            f.write(f"  Min:    {df['rf_predicted_reduction'].min():8.4f}°\n")
            f.write(f"  Max:    {df['rf_predicted_reduction'].max():8.4f}°\n\n")
            
            df_clean = df.dropna(subset=['actual_reduction'])
            if len(df_clean) > 0:
                f.write("Actual Error Reduction:\n")
                f.write(f"  Mean:   {df_clean['actual_reduction'].mean():8.4f}°\n")
                f.write(f"  Std:    {df_clean['actual_reduction'].std():8.4f}°\n")
                f.write(f"  Min:    {df_clean['actual_reduction'].min():8.4f}°\n")
                f.write(f"  Max:    {df_clean['actual_reduction'].max():8.4f}°\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
        
        print(f"Report saved to {report_path}")
    
    def run(self):
        """Run complete evaluation pipeline"""
        print("="*80)
        print("RF MODEL EVALUATION PIPELINE")
        print("="*80)
        
        # Load RF model
        self.load_rf_model()
        
        # Parse static parameters
        self.parse_all_parameters()
        
        # Process timesteps
        self.process_timesteps()
        
        # Analyze results
        df = self.analyze_results()
        
        # Generate visualizations
        self.generate_visualizations(df)
        
        # Generate report
        self.generate_report(df)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RF model predictions against actual CSS measurements')
    parser.add_argument('--data-dir',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/logs/7',
                       help='Directory containing NOS3InOut folder')
    parser.add_argument('--model-path',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/false_positives/models/rf_model.pkl',
                       help='Path to trained RF model (ignored if model-type is quantile)')
    parser.add_argument('--output-dir',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/false_positives/evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model-type', choices=['mean', 'quantile'], default='mean',
                       help='Type of model to evaluate')
    
    args = parser.parse_args()
    
    # Override model path and output dir based on model type
    if args.model_type == 'quantile':
        model_path = '<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/false_positives/models_quantile/quantile_rf_model_95.pkl'
        output_dir = args.output_dir + '_quantile'
    else:
        model_path = args.model_path
        output_dir = args.output_dir
    
    print(f"Evaluating {args.model_type} model")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    evaluator = RFModelEvaluator(
        data_dir=args.data_dir,
        model_path=model_path,
        output_dir=output_dir,
        model_type=args.model_type
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()