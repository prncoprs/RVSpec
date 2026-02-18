#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CSS Albedo Comparison - Time Series of Sun Vector Estimation Errors
Compares three methods: with_albedo, no_albedo, and compensated_albedo
Enhanced with violin plots and confidence bands
"""

import os
import sys
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

# ---------- Configuration ----------
DATA_DIR = Path("<RVSPEC_ROOT>/CPG-cFS/albedo/logs/7/NOS3InOut")
OUTPUT_DIR = Path("./output")

# ---------- Robust numeric readers (from original code) ----------
_FLOAT_RE = re.compile(r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|NAN)')

def read_numeric_cols(path, need=0, offset=0):
    """Read numeric columns from file"""
    rows = []
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
    return rows

def read_vector3_first3(path):
    """Read 3D vectors from file"""
    return read_numeric_cols(path, need=3, offset=0)

# ---------- Math helpers (from original code) ----------
def unit(v):
    """Normalize vector to unit length"""
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else [0.0,0.0,0.0]

def dot(a,b):
    """Dot product of two 3D vectors"""
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def norm(v):
    """Norm of vector"""
    return math.sqrt(sum(x*x for x in v))

def quat_to_cbn(q):
    """Convert quaternion to rotation matrix CBN (Body to Inertial)"""
    q1,q2,q3,q4 = q
    n = math.sqrt(q1*q1+q2*q2+q3*q3+q4*q4)
    if n == 0: return [[1,0,0],[0,1,0],[0,0,1]]
    q1,q2,q3,q4 = q1/n, q2/n, q3/n, q4/n
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

def angle_deg(a,b):
    """Angle between two vectors in degrees"""
    aa, bb = unit(a), unit(b)
    c = max(-1.0, min(1.0, dot(aa,bb)))
    return math.degrees(math.acos(c))

# ---------- Phase angle models ----------
def cosine_model(phase_angle, A, B, C):
    """Cosine model for albedo vs phase angle"""
    phase_rad = np.radians(phase_angle)
    return A * np.cos(phase_rad/2)**2 + B * np.cos(phase_rad/2) + C

def linear_model(phase_angle, a, b):
    """Linear model for albedo vs phase angle"""
    return a * phase_angle + b

def quadratic_model(phase_angle, a, b, c):
    """Quadratic model for albedo vs phase angle"""
    return a * phase_angle**2 + b * phase_angle + c

# ---------- CSS configuration parsing ----------
def detect_sc_file_from_inp_sim(inp_sim_path):
    """Detect spacecraft file from Inp_Sim.txt"""
    sc_file = None
    with open(inp_sim_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("<"):
                continue
            toks = ln.split()
            if len(toks) >= 3 and toks[0] in ("TRUE","FALSE") and toks[2].endswith(".txt"):
                sc_file = toks[2]
                break
    return sc_file

def parse_css_from_sc_file(sc_path):
    """Parse CSS configuration from spacecraft file"""
    axes, fov_deg = [], []
    lines = Path(sc_path).read_text().splitlines()
    i = 0
    Ncss_declared = None
    
    for j,ln in enumerate(lines):
        if "Coarse Sun Sensor" in ln:
            for k in range(j+1, min(j+6, len(lines))):
                m = re.search(r'^\s*(\d+)\s*!?\s*Number of Coarse Sun Sensors', lines[k])
                if m:
                    Ncss_declared = int(m.group(1))
                    break
            break

    while i < len(lines):
        if lines[i].strip().startswith("============================== CSS"):
            if i+3 < len(lines):
                try:
                    ax = [float(x) for x in re.split(r"\s+", lines[i+2].strip())[:3]]
                    half = float(re.split(r"\s+", lines[i+3].strip())[0])
                    axes.append(unit(ax))
                    fov_deg.append(half)
                except Exception:
                    pass
            i += 4
        else:
            i += 1

    if not axes and Ncss_declared:
        base = None
        for j,ln in enumerate(lines):
            if re.search(r'Number of Coarse Sun Sensors', ln):
                base = j + 1
                break
        if base is not None:
            i = base
            for _ in range(Ncss_declared):
                if i+2 < len(lines):
                    try:
                        ax = [float(x) for x in re.split(r"\s+", lines[i+1].strip())[:3]]
                        half = float(re.split(r"\s+", lines[i+2].strip())[0])
                        axes.append(unit(ax))
                        fov_deg.append(half)
                    except Exception:
                        pass
                i += 7
    return axes, fov_deg

def count_cols_first_nonempty(path):
    """Count columns in first non-empty line"""
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = _FLOAT_RE.findall(ln)
            if toks:
                return len(toks)
    return 0

# ---------- CSS reconstruction ----------
def css_reconstruct_sun_direction(intensities, axesB):
    """Reconstruct sun direction from CSS measurements"""
    wsum = [0.0, 0.0, 0.0]
    total = 0.0
    for I, axB in zip(intensities, axesB):
        if I > 0.0:
            wsum[0] += I * axB[0]
            wsum[1] += I * axB[1]
            wsum[2] += I * axB[2]
            total += I
    if total <= 0.0:
        return None
    return unit(wsum)

# ---------- Main processing ----------
def load_data():
    """Load all necessary data files"""
    print("Loading data files...")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    # Define file paths
    pos_path = DATA_DIR / "PosN.42"
    qbn_path = DATA_DIR / "qbn.42"
    svn_path = DATA_DIR / "svn.42"
    svb_path = DATA_DIR / "svb.42"
    alb_path = DATA_DIR / "Albedo.42"
    ill_path = DATA_DIR / "Illum.42"
    inp_sim = DATA_DIR / "Inp_Sim.txt"
    
    # Check all files exist
    for p in [pos_path, qbn_path, svn_path, svb_path, alb_path, ill_path, inp_sim]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    
    # Load data
    posN = read_vector3_first3(pos_path)
    qbn = read_numeric_cols(qbn_path, need=4, offset=0)
    svn = read_numeric_cols(svn_path, need=3, offset=0)
    svb = read_numeric_cols(svb_path, need=3, offset=0)
    alb = read_numeric_cols(alb_path, need=0)
    ill = read_numeric_cols(ill_path, need=0)
    
    # Detect CSS configuration
    sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
    sc_path = (DATA_DIR/sc_file_name) if sc_file_name else (DATA_DIR/"SC_NOS3.txt")
    if not sc_path.exists():
        sc_path = DATA_DIR/"SC_SensorFOV.txt"
    if not sc_path.exists():
        sc_path = None
    
    # Detect number of CSS
    ncss = count_cols_first_nonempty(alb_path)
    if ncss <= 0:
        ncss = count_cols_first_nonempty(ill_path)
    if ncss <= 0:
        raise ValueError("Cannot detect number of CSS")
    
    print(f"  Number of CSS: {ncss}")
    
    # Parse CSS configuration
    axesB, fov_deg = [], []
    if sc_path:
        axesB, fov_deg = parse_css_from_sc_file(sc_path)
    if not axesB or len(axesB) != ncss:
        default_axes = [
            [ 1, 0, 0], [-1, 0, 0],
            [ 0, 1, 0], [ 0,-1, 0],
            [ 0, 0, 1], [ 0, 0,-1],
        ]
        axesB = ([unit(a) for a in default_axes] * ((ncss+5)//6))[:ncss]
        fov_deg = [90.0]*ncss
    
    nsteps = min(len(posN), len(qbn), len(svn), len(svb), len(alb), len(ill))
    print(f"  Total steps: {nsteps}")
    
    return {
        'posN': posN,
        'qbn': qbn,
        'svn': svn,
        'svb': svb,
        'alb': alb,
        'ill': ill,
        'axesB': axesB,
        'ncss': ncss,
        'nsteps': nsteps
    }

def fit_phase_model(data):
    """Fit phase angle to albedo model"""
    print("Fitting phase-albedo model...")
    
    # Collect phase angles and total albedos
    phase_angles = []
    total_albedos = []
    
    for k in range(data['nsteps']):
        # Calculate global phase angle
        rN = data['posN'][k]
        rhatN = unit(rN)
        sN = unit(data['svn'][k])
        
        global_phase = angle_deg(sN, rhatN)
        phase_angles.append(global_phase)
        
        # Calculate total albedo
        alb_row = data['alb'][k][:data['ncss']] if len(data['alb'][k]) >= data['ncss'] else \
                  data['alb'][k] + [0.0]*(data['ncss']-len(data['alb'][k]))
        total_albedo = sum(alb_row)
        total_albedos.append(total_albedo)
    
    # Prepare data for fitting
    phase_angles = np.array(phase_angles)
    total_albedos = np.array(total_albedos)
    valid_mask = ~(np.isnan(phase_angles) | np.isnan(total_albedos))
    phase_valid = phase_angles[valid_mask]
    albedo_valid = total_albedos[valid_mask]
    
    # Try different models and select best
    models = {}
    
    # Cosine model
    try:
        popt_cos, _ = curve_fit(cosine_model, phase_valid, albedo_valid, 
                               p0=[0.1, 0.01, 0.001], maxfev=5000)
        pred_cos = cosine_model(phase_valid, *popt_cos)
        r2_cos = 1 - np.sum((albedo_valid - pred_cos)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
        models['cosine'] = {'params': popt_cos, 'r_squared': r2_cos}
    except:
        models['cosine'] = {'params': None, 'r_squared': 0}
    
    # Linear model
    try:
        popt_lin, _ = curve_fit(linear_model, phase_valid, albedo_valid)
        pred_lin = linear_model(phase_valid, *popt_lin)
        r2_lin = 1 - np.sum((albedo_valid - pred_lin)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
        models['linear'] = {'params': popt_lin, 'r_squared': r2_lin}
    except:
        models['linear'] = {'params': None, 'r_squared': 0}
    
    # Quadratic model
    try:
        popt_quad, _ = curve_fit(quadratic_model, phase_valid, albedo_valid)
        pred_quad = quadratic_model(phase_valid, *popt_quad)
        r2_quad = 1 - np.sum((albedo_valid - pred_quad)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
        models['quadratic'] = {'params': popt_quad, 'r_squared': r2_quad}
    except:
        models['quadratic'] = {'params': None, 'r_squared': 0}
    
    # Select best model
    best_model = max(models.keys(), key=lambda k: models[k]['r_squared'])
    best_params = models[best_model]['params']
    best_r2 = models[best_model]['r_squared']
    
    print(f"  Best model: {best_model} (R²={best_r2:.4f})")
    
    # Define prediction function
    if best_model == 'cosine':
        predict_albedo = lambda phase: cosine_model(phase, *best_params)
    elif best_model == 'quadratic':
        predict_albedo = lambda phase: quadratic_model(phase, *best_params)
    else:
        predict_albedo = lambda phase: linear_model(phase, *best_params)
    
    return predict_albedo, phase_angles

def calculate_errors(data, predict_albedo, phase_angles):
    """Calculate errors for all three methods"""
    print("Calculating errors for all methods...")
    
    errors_with_alb = []
    errors_no_alb = []
    errors_compensated = []
    
    for k in range(data['nsteps']):
        # Get current state
        sB_truth = unit(data['svb'][k])
        
        # Get measurements
        ill_row = data['ill'][k][:data['ncss']] if len(data['ill'][k]) >= data['ncss'] else \
                  data['ill'][k] + [0.0]*(data['ncss']-len(data['ill'][k]))
        alb_row = data['alb'][k][:data['ncss']] if len(data['alb'][k]) >= data['ncss'] else \
                  data['alb'][k] + [0.0]*(data['ncss']-len(data['alb'][k]))
        
        # Method 1: With albedo (baseline)
        sB_with = css_reconstruct_sun_direction(ill_row, data['axesB'])
        if sB_with is not None:
            error = angle_deg(sB_with, sB_truth)
            errors_with_alb.append(error)
        else:
            errors_with_alb.append(np.nan)
        
        # Method 2: No albedo (ideal)
        no_alb_vec = [max(0.0, ill_row[i] - alb_row[i]) for i in range(data['ncss'])]
        sB_no = css_reconstruct_sun_direction(no_alb_vec, data['axesB'])
        if sB_no is not None:
            error = angle_deg(sB_no, sB_truth)
            errors_no_alb.append(error)
        else:
            errors_no_alb.append(np.nan)
        
        # Method 3: Compensated albedo (predicted with equal distribution)
        global_phase = phase_angles[k]
        predicted_total = predict_albedo(global_phase)
        predicted_total = max(0, predicted_total)
        
        num_active = sum(1 for i in ill_row if i > 0)
        if num_active > 0:
            predicted_per_css = predicted_total / num_active
            corrected_equal = [max(0, ill_row[i] - predicted_per_css) if ill_row[i] > 0 else 0 
                              for i in range(data['ncss'])]
            sB_comp = css_reconstruct_sun_direction(corrected_equal, data['axesB'])
            if sB_comp is not None:
                error = angle_deg(sB_comp, sB_truth)
                errors_compensated.append(error)
            else:
                errors_compensated.append(np.nan)
        else:
            errors_compensated.append(np.nan)
    
    return {
        'with_albedo': np.array(errors_with_alb),
        'no_albedo': np.array(errors_no_alb),
        'compensated_albedo': np.array(errors_compensated)
    }

def plot_comparison_with_violin(errors):
    """Create enhanced comparison plot with violin plots and confidence bands"""
    print("Creating enhanced comparison plot...")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib parameters for academic style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 1
    
    import matplotlib
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,   #  TrueType (Type 42) Type 3
        "ps.fonttype": 42,
    })
    
    # Create figure with gridspec for main plot and violin plot
    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.02, figure=fig)
    
    ax_main = fig.add_subplot(gs[0])
    ax_violin = fig.add_subplot(gs[1])
    
    # Limit to first 1800 time steps
    max_steps = 1800
    time_steps = np.arange(min(len(errors['with_albedo']), max_steps))
    
    # Slice errors to first 1800 steps
    errors_with_sliced = errors['with_albedo'][:max_steps]
    errors_no_sliced = errors['no_albedo'][:max_steps]
    errors_comp_sliced = errors['compensated_albedo'][:max_steps]
    
    # Apply smoothing with Savitzky-Golay filter
    window_length = 25
    if window_length > len(time_steps):
        window_length = len(time_steps) if len(time_steps) % 2 == 1 else len(time_steps) - 1
    if window_length < 3:
        window_length = 3
    
    polyorder = min(3, window_length - 1)
    
    # Smooth each dataset and calculate rolling statistics
    def smooth_and_stats(data):
        # Handle NaN values
        mask = ~np.isnan(data)
        if np.sum(mask) < window_length:
            return data, np.zeros_like(data)
        
        # Simple forward fill for NaN values
        data_filled = data.copy()
        for i in range(1, len(data_filled)):
            if np.isnan(data_filled[i]):
                data_filled[i] = data_filled[i-1]
        
        # Apply smoothing
        try:
            smoothed = savgol_filter(data_filled, window_length, polyorder)
        except:
            smoothed = data_filled
        
        # Calculate rolling standard deviation for confidence bands
        df = pd.Series(data_filled)
        rolling_std = df.rolling(window=50, center=True).std().fillna(0).values
        
        return smoothed, rolling_std
    
    errors_with_smooth, std_with = smooth_and_stats(errors_with_sliced)
    errors_no_smooth, std_no = smooth_and_stats(errors_no_sliced)
    errors_comp_smooth, std_comp = smooth_and_stats(errors_comp_sliced)
    
    # Calculate means for the displayed portion
    mean_with = np.nanmean(errors_with_sliced)
    mean_no = np.nanmean(errors_no_sliced)
    mean_comp = np.nanmean(errors_comp_sliced)
    
    # Define colors
    color_with = '#FFB000'  # Orange
    color_no = '#A63D40'    # Dark red
    color_comp = '#648FFF'  # Blue
    
    # Plot main lines with confidence bands
    # With Albedo
    ax_main.plot(time_steps, errors_with_smooth, 
                color=color_with, linewidth=2.0, alpha=0.9,
                label=f'w/ Albedo', zorder=3)
    
    # Compensated
    ax_main.plot(time_steps, errors_comp_smooth, 
                color=color_comp, linewidth=2.0, alpha=0.9,
                label=f'Compensated', zorder=3)
    
    # No Albedo (Ideal)
    ax_main.plot(time_steps, errors_no_smooth, 
                color=color_no, linewidth=6.9, alpha=0.9, linestyle='--',
                label=f'w/o Albedo', zorder=3)

    
    # Add subtle grid
    # ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Set labels and limits for main plot
    # ax_main.set_xlabel('Time Step', fontsize=28, fontweight='bold')
    # ax_main.set_ylabel('Sun Vector\nEstimation\nError (°)', fontsize=30, fontweight='bold')
    ax_main.set_xlabel('Time Step', fontsize=28)
    ax_main.set_ylabel('Sun Vector\nEstimation\nError(°)', fontsize=30)
    ax_main.set_xlim(0, max_steps)
    ax_main.set_ylim(0, 15)
    
    # Customize tick labels
    ax_main.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=6)
    ax_main.tick_params(axis='both', which='minor', width=1, length=4)
    
    # Add minor ticks
    ax_main.xaxis.set_minor_locator(MultipleLocator(100))
    ax_main.yaxis.set_minor_locator(MultipleLocator(1))
    ax_main.xaxis.set_major_locator(MultipleLocator(500))
    ax_main.yaxis.set_major_locator(MultipleLocator(5))
    
    # Add legend
    legend = ax_main.legend(loc='upper right',                # 
                            bbox_to_anchor=(1.02, 1.05), 
                            fontsize=22, frameon=True, 
                            fancybox=False, shadow=False, 
                            framealpha=0.95, edgecolor='black', 
                            borderpad=0.3, columnspacing=0, handlelength=1)
    legend.get_frame().set_linewidth(1)
    
    # Remove top and right spines
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # --- Violin Plot ---
    # Prepare data for violin plot (remove NaN values)
    violin_data = [
        errors_with_sliced[~np.isnan(errors_with_sliced)],
        errors_comp_sliced[~np.isnan(errors_comp_sliced)],
        errors_no_sliced[~np.isnan(errors_no_sliced)]
    ]
    
    # Create violin plot
    parts = ax_violin.violinplot(
        violin_data,
        positions=[0, 1, 2],
        vert=True,
        widths=0.6,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )
    
    # Customize violin colors
    colors_violin = [color_with, color_comp, color_no]
    for pc, color in zip(parts['bodies'], colors_violin):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0)
    
    # Customize median, mean, and extrema lines
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)
    # parts['cmedians'].set_color('black')
    # parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.1)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.1)
    
    # Set violin plot properties
    ax_violin.set_ylim(0, 15)
    ax_violin.set_xticks([0, 1, 2])
    ax_violin.set_xticklabels(['w/', 'C.', 'w/o'], fontsize=22, rotation=0)
    ax_violin.set_xlabel('Dist.', fontsize=28)
    
    # Hide y-axis for violin plot (shared with main)
    ax_violin.yaxis.set_visible(False)
    
    # Add grid to violin plot
    ax_violin.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Remove top and right spines for violin plot
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    ax_violin.spines['left'].set_visible(False)
    
    # # Add mean values as text annotations on violin plot
    # for i, (data, label, mean_val) in enumerate(zip(violin_data, 
    #                                                 ['w/', 'Comp', 'w/o'],
    #                                                 [7.15, 4.87, 1.09])):
    #     ax_violin.text(i, 16, f'{mean_val:.2f}°', 
    #                 ha='center', va='top', fontsize=26,
    #                 bbox=dict(boxstyle='round,pad=0.3', 
    #                         facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add title to violin plot
    # ax_violin.set_title('Distribution', fontsize=14, fontweight='bold', pad=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = OUTPUT_DIR / "albedo_comparison.pdf"
    plt.savefig(output_file, dpi=600, format='pdf', bbox_inches='tight')
    print(f"  Saved to: {output_file}")
    
    # Display statistics
    print(f"\nError Statistics (Time steps 0-{min(len(errors['with_albedo']), max_steps)}):")
    print(f"  With Albedo:        Mean = {mean_with:.2f}°, Std = {np.nanstd(errors_with_sliced):.2f}°")
    print(f"  No Albedo:          Mean = {mean_no:.2f}°, Std = {np.nanstd(errors_no_sliced):.2f}°")
    print(f"  Compensated Albedo: Mean = {mean_comp:.2f}°, Std = {np.nanstd(errors_comp_sliced):.2f}°")
    
    improvement = ((mean_with - mean_comp) / mean_with) * 100
    ideal_improvement = ((mean_with - mean_no) / mean_with) * 100
    efficiency = (improvement / ideal_improvement) * 100 if ideal_improvement > 0 else 0
    
    print(f"\nImprovement Analysis:")
    print(f"  Compensated vs Baseline: {improvement:.1f}% improvement")
    print(f"  Achieves {efficiency:.1f}% of ideal correction")
    
    plt.close()

def main():
    """Main function"""
    try:
        print("="*60)
        print("CSS Albedo Comparison - Enhanced Time Series Plot")
        print("="*60)
        
        # Load data
        data = load_data()
        
        # Fit phase-albedo model
        predict_albedo, phase_angles = fit_phase_model(data)
        
        # Calculate errors for all methods
        errors = calculate_errors(data, predict_albedo, phase_angles)
        
        # Create enhanced comparison plot with violin plots
        plot_comparison_with_violin(errors)
        
        print("\nDone!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()