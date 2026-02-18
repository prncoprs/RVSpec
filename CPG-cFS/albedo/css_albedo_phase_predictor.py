#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS Albedo Predictor using Global Phase Angle
Predicts total albedo from global phase angle and corrects CSS measurements
"""

import argparse, os, sys, math, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
import traceback

# ---------- robust numeric readers ----------
_FLOAT_RE = re.compile(r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|NAN)')

def read_numeric_cols(path, need=0, offset=0):
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
    return read_numeric_cols(path, need=3, offset=0)

# ---------- math helpers ----------
def unit(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else [0.0,0.0,0.0]

def dot(a,b): 
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def vec_scale(v, s):
    return [v[0]*s, v[1]*s, v[2]*s]

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

def matvec(C, v):
    """Matrix-vector multiplication"""
    return [C[0][0]*v[0]+C[0][1]*v[1]+C[0][2]*v[2],
            C[1][0]*v[0]+C[1][1]*v[1]+C[1][2]*v[2],
            C[2][0]*v[0]+C[2][1]*v[1]+C[2][2]*v[2]]

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

# ---------- parse helpers ----------
def detect_sc_file_from_inp_sim(inp_sim_path):
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
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = _FLOAT_RE.findall(ln)
            if toks:
                return len(toks)
    return 0

# ---------- CSS-based reconstruction ----------
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

def angle_stats(arr):
    """Compute angle statistics"""
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, 0)
    return (float(a.mean()), float(np.median(a)), float(np.percentile(a,95)), float(a.max()), int(a.size))

# ---------- Visualization functions ----------
def create_analysis_plots(results, dest_dir):
    """Create comprehensive visualization of prediction results"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Plot 1: Global Phase vs Total Albedo with fitted model
    ax = fig.add_subplot(gs[0, 0])
    
    phase_angles = results['phase_angles']
    total_albedos = results['total_albedos']
    
    # Remove NaN values
    valid_mask = ~(np.isnan(phase_angles) | np.isnan(total_albedos))
    phase_valid = np.array(phase_angles)[valid_mask]
    albedo_valid = np.array(total_albedos)[valid_mask]
    
    if len(phase_valid) > 0:
        # Scatter plot of actual data
        scatter = ax.scatter(phase_valid, albedo_valid, alpha=0.3, s=1, c='blue', label='Actual')
        
        # Plot fitted model
        if 'model_params' in results:
            x_model = np.linspace(np.min(phase_valid), np.max(phase_valid), 200)
            if results['model_type'] == 'cosine':
                y_model = cosine_model(x_model, *results['model_params'])
            elif results['model_type'] == 'quadratic':
                y_model = quadratic_model(x_model, *results['model_params'])
            else:  # linear
                y_model = linear_model(x_model, *results['model_params'])
            
            ax.plot(x_model, y_model, 'r-', linewidth=2, 
                   label=f'{results["model_type"]} (R²={results["r_squared"]:.3f})')
        
        ax.set_xlabel('Global Phase Angle (degrees)')
        ax.set_ylabel('Total Albedo')
        ax.set_title('Phase-Albedo Relationship')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Error Comparison
    ax = fig.add_subplot(gs[0, 1])
    
    methods = ['with_alb', 'no_alb', 'predicted_equal', 'predicted_weighted']
    labels = ['With Albedo', 'No Albedo (Ideal)', 'Predicted (Equal)', 'Predicted (Weighted)']
    colors = ['red', 'green', 'blue', 'orange']
    
    mean_errors = []
    for method in methods:
        if method in results['stats']:
            mean_errors.append(results['stats'][method]['mean'])
        else:
            mean_errors.append(0)
    
    bars = ax.bar(range(len(labels)), mean_errors, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Pointing Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, mean_errors):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}°', ha='center', va='bottom')
    
    # Plot 3: Prediction Accuracy
    ax = fig.add_subplot(gs[0, 2])
    
    if 'predicted_albedos' in results and 'total_albedos' in results:
        actual = np.array(results['total_albedos'])
        predicted = np.array(results['predicted_albedos'])
        
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_valid = actual[valid_mask]
        predicted_valid = predicted[valid_mask]
        
        if len(actual_valid) > 0:
            ax.scatter(actual_valid, predicted_valid, alpha=0.5, s=1)
            
            # Add perfect prediction line
            min_val = min(np.min(actual_valid), np.min(predicted_valid))
            max_val = max(np.max(actual_valid), np.max(predicted_valid))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Calculate and display R²
            correlation = np.corrcoef(actual_valid, predicted_valid)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_xlabel('Actual Total Albedo')
            ax.set_ylabel('Predicted Total Albedo')
            ax.set_title('Albedo Prediction Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution
    ax = fig.add_subplot(gs[1, 0])
    
    for method, label, color in [('with_alb', 'With Albedo', 'red'),
                                  ('predicted_equal', 'Predicted', 'blue')]:
        if method in results and 'errors' in results[method]:
            errors = results[method]['errors']
            if len(errors) > 0:
                ax.hist(errors, bins=50, alpha=0.5, label=label, color=color, density=True)
    
    ax.set_xlabel('Pointing Error (degrees)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Improvement over baseline
    ax = fig.add_subplot(gs[1, 1])
    
    if 'with_alb' in results['stats']:
        baseline = results['stats']['with_alb']['mean']
        ideal = results['stats']['no_alb']['mean'] if 'no_alb' in results['stats'] else 0
        
        improvements = []
        improvement_labels = []
        improvement_colors = []
        
        # Ideal improvement
        if ideal > 0:
            ideal_improvement = ((baseline - ideal) / baseline) * 100
            improvements.append(ideal_improvement)
            improvement_labels.append('Ideal (No Albedo)')
            improvement_colors.append('green')
        
        # Predicted improvements
        for method, label, color in [('predicted_equal', 'Predicted (Equal)', 'blue'),
                                     ('predicted_weighted', 'Predicted (Weighted)', 'orange')]:
            if method in results['stats']:
                improvement = ((baseline - results['stats'][method]['mean']) / baseline) * 100
                improvements.append(improvement)
                improvement_labels.append(label)
                improvement_colors.append(color)
        
        bars = ax.bar(range(len(improvements)), improvements, color=improvement_colors, alpha=0.7)
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels(improvement_labels, rotation=45, ha='right')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Improvement over Baseline')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Add percentage labels
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # Plot 6: Time series of errors
    ax = fig.add_subplot(gs[1, 2])
    
    # Check if we have errors to plot
    if 'with_alb' in results and 'errors' in results['with_alb'] and len(results['with_alb']['errors']) > 0:
        num_errors = len(results['with_alb']['errors'])
        sample_size = min(500, num_errors)
        
        if sample_size > 0:
            indices = np.linspace(0, num_errors-1, sample_size, dtype=int)
            
            for method, label, color in [('with_alb', 'With Albedo', 'red'),
                                         ('predicted_equal', 'Predicted', 'blue')]:
                if method in results and 'errors' in results[method]:
                    method_errors = results[method]['errors']
                    if len(method_errors) > 0:
                        # Make sure we don't exceed the actual error list length
                        valid_indices = [i for i in indices if i < len(method_errors)]
                        sampled_errors = [method_errors[i] for i in valid_indices]
                        ax.plot(valid_indices, sampled_errors, alpha=0.5, label=label, 
                               color=color, linewidth=0.5)
            
            ax.set_xlabel('Time Step (sampled)')
            ax.set_ylabel('Pointing Error (degrees)')
            ax.set_title('Error Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No error data available', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Error Time Series')
    
    # Plot 7: P95 Comparison
    ax = fig.add_subplot(gs[2, 0])
    
    p95_values = []
    p95_labels = []
    p95_colors = []
    
    for method, label, color in [('with_alb', 'With Albedo', 'red'),
                                 ('no_alb', 'No Albedo', 'green'),
                                 ('predicted_equal', 'Predicted (Equal)', 'blue'),
                                 ('predicted_weighted', 'Predicted (Weighted)', 'orange')]:
        if method in results['stats']:
            p95_values.append(results['stats'][method]['p95'])
            p95_labels.append(label)
            p95_colors.append(color)
    
    bars = ax.bar(range(len(p95_values)), p95_values, color=p95_colors, alpha=0.7)
    ax.set_xticks(range(len(p95_values)))
    ax.set_xticklabels(p95_labels, rotation=45, ha='right')
    ax.set_ylabel('95th Percentile Error (degrees)')
    ax.set_title('P95 Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Model comparison
    ax = fig.add_subplot(gs[2, 1])
    
    if 'model_comparison' in results:
        models = list(results['model_comparison'].keys())
        r2_values = [results['model_comparison'][m]['r_squared'] for m in models]
        
        bars = ax.bar(range(len(models)), r2_values, color='purple', alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 9: Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Method', 'Mean', 'Median', 'P95', 'Max'])
    
    for method, label in [('with_alb', 'With Albedo'),
                          ('no_alb', 'No Albedo'),
                          ('predicted_equal', 'Predicted (Equal)'),
                          ('predicted_weighted', 'Predicted (Weighted)')]:
        if method in results['stats']:
            s = results['stats'][method]
            table_data.append([label,
                             f"{s['mean']:.2f}°",
                             f"{s['median']:.2f}°",
                             f"{s['p95']:.2f}°",
                             f"{s['max']:.2f}°"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Global Phase Angle Albedo Prediction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "phase_prediction_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="CSS Albedo Prediction using Global Phase Angle")
        ap.add_argument("--local-dir", required=True, 
                       help="Path to directory containing NOS3InOut folder")
        ap.add_argument("--dest", required=True,
                       help="Destination directory for analysis results")
        args = ap.parse_args()

        # Setup paths
        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        
        print(f"CSS Albedo Prediction using Global Phase Angle")
        print(f"="*60)
        print(f"Input: {local_dir}")
        print(f"Output: {dest}")
        
        dest.mkdir(parents=True, exist_ok=True)
        
        # Find NOS3InOut folder
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut folder not found in: {local_dir}")
        
        print(f"\n[1/8] Loading data files...")
        
        # Load all required files
        pos_path = local_nos3/"PosN.42"
        qbn_path = local_nos3/"qbn.42"
        svn_path = local_nos3/"svn.42"
        svb_path = local_nos3/"svb.42"
        alb_path = local_nos3/"Albedo.42"
        ill_path = local_nos3/"Illum.42"
        inp_sim = local_nos3/"Inp_Sim.txt"
        
        for p in [pos_path,qbn_path,svn_path,svb_path,alb_path,ill_path,inp_sim]:
            if not p.exists():
                sys.exit(f"Missing file: {p}")
        
        # Detect CSS configuration
        sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
        sc_path = (local_nos3/sc_file_name) if sc_file_name else (local_nos3/"SC_NOS3.txt")
        if not sc_path.exists():
            sc_path = local_nos3/"SC_SensorFOV.txt"
        if not sc_path.exists():
            sc_path = None
        
        # Detect number of CSS
        ncss = count_cols_first_nonempty(alb_path)
        if ncss <= 0:
            ncss = count_cols_first_nonempty(ill_path)
        if ncss <= 0:
            sys.exit("Cannot detect number of CSS")
        
        print(f"    Number of CSS: {ncss}")
        
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
        
        # Load data
        posN = read_vector3_first3(pos_path)
        qbn = read_numeric_cols(qbn_path, need=4, offset=0)
        svn = read_numeric_cols(svn_path, need=3, offset=0)
        svb = read_numeric_cols(svb_path, need=3, offset=0)
        alb = read_numeric_cols(alb_path, need=0)
        ill = read_numeric_cols(ill_path, need=0)
        
        nsteps = min(len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill))
        print(f"    Total steps: {nsteps}")
        
        print(f"\n[2/8] Collecting phase angle and albedo data...")
        
        # Collect global phase angles and total albedos
        phase_angles = []
        total_albedos = []
        
        for k in range(nsteps):
            # Calculate global phase angle
            rN = posN[k]
            rhatN = unit(rN)
            sN = unit(svn[k])
            
            global_phase = angle_deg(sN, rhatN)
            phase_angles.append(global_phase)
            
            # Calculate total albedo
            alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))
            total_albedo = sum(alb_row)
            total_albedos.append(total_albedo)
        
        print(f"\n[3/8] Fitting phase-albedo models...")
        
        # Prepare data for fitting
        valid_mask = ~(np.isnan(phase_angles) | np.isnan(total_albedos))
        phase_valid = np.array(phase_angles)[valid_mask]
        albedo_valid = np.array(total_albedos)[valid_mask]
        
        # Try different models
        model_comparison = {}
        
        # 1. Cosine model
        try:
            popt_cos, _ = curve_fit(cosine_model, phase_valid, albedo_valid, 
                                   p0=[0.1, 0.01, 0.001], maxfev=5000)
            pred_cos = cosine_model(phase_valid, *popt_cos)
            r2_cos = 1 - np.sum((albedo_valid - pred_cos)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
            model_comparison['cosine'] = {'params': popt_cos, 'r_squared': r2_cos}
        except:
            model_comparison['cosine'] = {'params': None, 'r_squared': 0}
        
        # 2. Linear model
        try:
            popt_lin, _ = curve_fit(linear_model, phase_valid, albedo_valid)
            pred_lin = linear_model(phase_valid, *popt_lin)
            r2_lin = 1 - np.sum((albedo_valid - pred_lin)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
            model_comparison['linear'] = {'params': popt_lin, 'r_squared': r2_lin}
        except:
            model_comparison['linear'] = {'params': None, 'r_squared': 0}
        
        # 3. Quadratic model
        try:
            popt_quad, _ = curve_fit(quadratic_model, phase_valid, albedo_valid)
            pred_quad = quadratic_model(phase_valid, *popt_quad)
            r2_quad = 1 - np.sum((albedo_valid - pred_quad)**2) / np.sum((albedo_valid - np.mean(albedo_valid))**2)
            model_comparison['quadratic'] = {'params': popt_quad, 'r_squared': r2_quad}
        except:
            model_comparison['quadratic'] = {'params': None, 'r_squared': 0}
        
        # Select best model
        best_model = max(model_comparison.keys(), key=lambda k: model_comparison[k]['r_squared'])
        best_params = model_comparison[best_model]['params']
        best_r2 = model_comparison[best_model]['r_squared']
        
        print(f"    Best model: {best_model} (R²={best_r2:.4f})")
        
        # Define prediction function based on best model
        if best_model == 'cosine':
            predict_albedo = lambda phase: cosine_model(phase, *best_params)
        elif best_model == 'quadratic':
            predict_albedo = lambda phase: quadratic_model(phase, *best_params)
        else:
            predict_albedo = lambda phase: linear_model(phase, *best_params)
        
        print(f"\n[4/8] Processing timesteps with albedo prediction...")
        
        # Initialize tracking
        methods_data = {
            'with_alb': {'errors': [], 'valid': 0},
            'no_alb': {'errors': [], 'valid': 0},
            'predicted_equal': {'errors': [], 'valid': 0},
            'predicted_weighted': {'errors': [], 'valid': 0}
        }
        
        predicted_albedos = []
        daylight_count = 0
        
        # Main processing loop
        for k in range(nsteps):
            # Get current state
            rN = posN[k]
            rhatN = unit(rN)
            CBN = quat_to_cbn(qbn[k])
            sN = unit(svn[k])
            sB_truth = unit(svb[k])
            
            is_day = dot(sN, rhatN) > 0
            if is_day:
                daylight_count += 1
            
            # Get measurements
            ill_row = ill[k][:ncss] if len(ill[k]) >= ncss else ill[k] + [0.0]*(ncss-len(ill[k]))
            alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))
            
            # Calculate global phase angle
            global_phase = angle_deg(sN, rhatN)
            
            # Predict total albedo
            predicted_total = predict_albedo(global_phase)
            predicted_total = max(0, predicted_total)  # Ensure non-negative
            predicted_albedos.append(predicted_total)
            
            # Method 1: With albedo (baseline)
            sB_with = css_reconstruct_sun_direction(ill_row, axesB)
            if sB_with is not None:
                error = angle_deg(sB_with, sB_truth)
                methods_data['with_alb']['errors'].append(error)
                methods_data['with_alb']['valid'] += 1
            
            # Method 2: No albedo (ideal)
            no_alb_vec = [max(0.0, ill_row[i] - alb_row[i]) for i in range(ncss)]
            sB_no = css_reconstruct_sun_direction(no_alb_vec, axesB)
            if sB_no is not None:
                error = angle_deg(sB_no, sB_truth)
                methods_data['no_alb']['errors'].append(error)
                methods_data['no_alb']['valid'] += 1
            
            # Method 3: Predicted with equal distribution
            num_active = sum(1 for i in ill_row if i > 0)
            if num_active > 0:
                predicted_per_css = predicted_total / num_active
                corrected_equal = [max(0, ill_row[i] - predicted_per_css) if ill_row[i] > 0 else 0 
                                  for i in range(ncss)]
                sB_pred_eq = css_reconstruct_sun_direction(corrected_equal, axesB)
                if sB_pred_eq is not None:
                    error = angle_deg(sB_pred_eq, sB_truth)
                    methods_data['predicted_equal']['errors'].append(error)
                    methods_data['predicted_equal']['valid'] += 1
            
            # Method 4: Predicted with weighted distribution
            # Weight by CSS orientation toward Earth (nadir)
            nadirB = matvec(CBN, [-rhatN[0], -rhatN[1], -rhatN[2]])
            
            weights = []
            for i in range(ncss):
                # CSS facing Earth gets more albedo
                cos_angle = dot(axesB[i], nadirB)
                weight = max(0, cos_angle)  # Only positive contributions
                weights.append(weight)
            
            total_weight = sum(weights)
            if total_weight > 0 and num_active > 0:
                # Distribute predicted albedo based on weights
                corrected_weighted = []
                for i in range(ncss):
                    if ill_row[i] > 0:
                        predicted_for_css = predicted_total * (weights[i] / total_weight)
                        corrected_weighted.append(max(0, ill_row[i] - predicted_for_css))
                    else:
                        corrected_weighted.append(0)
                
                sB_pred_wt = css_reconstruct_sun_direction(corrected_weighted, axesB)
                if sB_pred_wt is not None:
                    error = angle_deg(sB_pred_wt, sB_truth)
                    methods_data['predicted_weighted']['errors'].append(error)
                    methods_data['predicted_weighted']['valid'] += 1
        
        print(f"    Daylight steps: {daylight_count}/{nsteps}")
        
        print(f"\n[5/8] Computing statistics...")
        
        # Compute statistics for all methods
        stats = {}
        for method, data in methods_data.items():
            if data['errors']:
                mean, median, p95, max_err, count = angle_stats(data['errors'])
                stats[method] = {
                    'mean': mean,
                    'median': median,
                    'p95': p95,
                    'max': max_err,
                    'count': count,
                    'valid': data['valid'],
                    'valid_pct': (data['valid'] / nsteps) * 100
                }
        
        print(f"\n[6/8] Generating report...")
        
        # Generate report
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("CSS ALBEDO PREDICTION USING GLOBAL PHASE ANGLE")
        report_lines.append("="*70)
        report_lines.append("")
        
        report_lines.append("MODEL FITTING RESULTS")
        report_lines.append("-"*40)
        for model_name, model_data in model_comparison.items():
            r2 = model_data['r_squared']
            report_lines.append(f"{model_name:12s}: R² = {r2:.4f}")
        report_lines.append(f"\nBest model: {best_model} (R² = {best_r2:.4f})")
        
        report_lines.append("")
        report_lines.append("POINTING ERROR STATISTICS")
        report_lines.append("-"*40)
        report_lines.append("Method               | Mean    | Median  | P95     | Max     | Valid%")
        report_lines.append("---------------------|---------|---------|---------|---------|-------")
        
        for method in ['with_alb', 'no_alb', 'predicted_equal', 'predicted_weighted']:
            if method in stats:
                s = stats[method]
                report_lines.append(f"{method:20s} | {s['mean']:7.3f} | {s['median']:7.3f} | "
                                  f"{s['p95']:7.3f} | {s['max']:7.3f} | {s['valid_pct']:6.1f}")
        
        report_lines.append("")
        report_lines.append("IMPROVEMENT ANALYSIS")
        report_lines.append("-"*40)
        
        if 'with_alb' in stats:
            baseline = stats['with_alb']['mean']
            
            for method in ['no_alb', 'predicted_equal', 'predicted_weighted']:
                if method in stats:
                    improvement = ((baseline - stats[method]['mean']) / baseline) * 100
                    efficiency = (stats[method]['mean'] - stats['no_alb']['mean']) / (baseline - stats['no_alb']['mean']) * 100 if 'no_alb' in stats else 0
                    
                    method_label = {
                        'no_alb': 'No Albedo (Ideal)',
                        'predicted_equal': 'Predicted (Equal Dist)',
                        'predicted_weighted': 'Predicted (Weighted)'
                    }[method]
                    
                    report_lines.append(f"{method_label:25s}: {improvement:6.1f}% improvement")
                    if method != 'no_alb' and efficiency > 0:
                        report_lines.append(f"{'':25s}  ({100-efficiency:5.1f}% of ideal correction)")
        
        report_lines.append("")
        report_lines.append("CONCLUSION")
        report_lines.append("-"*40)
        
        if 'predicted_equal' in stats and 'no_alb' in stats and 'with_alb' in stats:
            pred_mean = stats['predicted_equal']['mean']
            ideal_mean = stats['no_alb']['mean']
            baseline_mean = stats['with_alb']['mean']
            
            correction_quality = (baseline_mean - pred_mean) / (baseline_mean - ideal_mean) * 100
            
            if correction_quality > 70:
                report_lines.append(" Excellent albedo correction using global phase angle")
            elif correction_quality > 50:
                report_lines.append(" Good albedo correction using global phase angle")
            elif correction_quality > 30:
                report_lines.append(" Moderate albedo correction using global phase angle")
            else:
                report_lines.append(" Limited albedo correction using global phase angle")
            
            report_lines.append(f"  Achieves {correction_quality:.1f}% of ideal correction")
        
        # Save report
        report_file = dest / "phase_prediction_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Prepare results for visualization
        results = {
            'phase_angles': phase_angles,
            'total_albedos': total_albedos,
            'predicted_albedos': predicted_albedos,
            'model_type': best_model,
            'model_params': best_params,
            'r_squared': best_r2,
            'model_comparison': model_comparison,
            'stats': stats,
            'with_alb': methods_data['with_alb'],
            'predicted_equal': methods_data['predicted_equal']
        }
        
        print(f"\n[7/8] Creating visualizations...")
        viz_path = create_analysis_plots(results, dest)
        
        print(f"\n[8/8] Analysis complete!")
        
        # Print summary
        print(f"\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Phase-Albedo Model: {best_model} (R²={best_r2:.4f})")
        print("")
        print("Pointing Errors:")
        print(f"  Baseline (with albedo):    {stats['with_alb']['mean']:.2f}°")
        print(f"  Ideal (no albedo):         {stats['no_alb']['mean']:.2f}°")
        if 'predicted_equal' in stats:
            print(f"  Predicted (equal dist):    {stats['predicted_equal']['mean']:.2f}°")
        if 'predicted_weighted' in stats:
            print(f"  Predicted (weighted):      {stats['predicted_weighted']['mean']:.2f}°")
        
        print(f"\nOutput files:")
        print(f"  • {report_file.name}")
        print(f"  • {viz_path.name}")
        print(f"\nAll files saved to: {dest}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()