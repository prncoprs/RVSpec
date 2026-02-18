#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS Albedo Analyzer - Final Corrected Version
Implements proper dynamic removal strategies:
1. Ground Truth: Removes CSS with highest actual albedo at each timestep
2. Predicted: Removes CSS with smallest phase angle (highest predicted albedo) at each timestep
"""

import argparse, os, sys, math, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
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

def mattranspose(C):
    """Matrix transpose"""
    return [[C[0][0], C[1][0], C[2][0]],
            [C[0][1], C[1][1], C[2][1]],
            [C[0][2], C[1][2], C[2][2]]]

def angle_deg(a,b):
    """Angle between two vectors in degrees"""
    aa, bb = unit(a), unit(b)
    c = max(-1.0, min(1.0, dot(aa,bb)))
    return math.degrees(math.acos(c))

# ---------- CSS-specific phase angle calculation ----------
def calculate_css_specific_phase_angle_simple(css_axis_N, sun_N, sat_pos_N):
    """
    Calculate CSS-specific phase angle
    Smaller phase angle = CSS sees bright side of Earth = more albedo effect
    """
    rhat_N = unit(sat_pos_N)
    global_phase = angle_deg(sun_N, rhat_N)
    
    nadir_N = vec_scale(rhat_N, -1)
    css_earth_alignment = dot(css_axis_N, nadir_N)
    
    if css_earth_alignment <= 0:
        # CSS not facing Earth
        return 180.0
    
    css_sun_alignment = dot(css_axis_N, sun_N)
    modulation = 1.0 - css_sun_alignment * 0.3
    css_phase = global_phase * modulation
    
    return max(0, min(180, css_phase))

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
def css_reconstruct_sun_direction(intensities, axesB, exclude_indices=None):
    """
    Reconstruct sun direction from CSS measurements
    Fixed version: No nadir angle filtering
    """
    wsum = [0.0, 0.0, 0.0]
    total = 0.0
    for i, (I, axB) in enumerate(zip(intensities, axesB)):
        if exclude_indices and i in exclude_indices:
            continue  # Skip excluded CSS
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
def create_comparison_plots(stats, removal_tracking, dest_dir, ncss):
    """Create comprehensive visualization of removal strategies"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Plot 1: Mean Error Comparison
    ax = fig.add_subplot(gs[0, 0])
    methods = ['with_alb', 'no_alb', 'remove_gt_n1', 'remove_gt_n2', 'remove_gt_n3',
               'remove_pred_n1', 'remove_pred_n2', 'remove_pred_n3']
    labels = ['With Alb', 'No Alb', 'GT n=1', 'GT n=2', 'GT n=3', 
              'Pred n=1', 'Pred n=2', 'Pred n=3']
    colors = ['red', 'green', 'blue', 'blue', 'blue', 'orange', 'orange', 'orange']
    
    mean_errors = []
    valid_labels = []
    valid_colors = []
    
    for m, l, c in zip(methods, labels, colors):
        if m in stats['global']:
            mean_errors.append(stats['global'][m]['mean'])
            valid_labels.append(l)
            valid_colors.append(c)
    
    bars = ax.bar(range(len(mean_errors)), mean_errors, color=valid_colors, alpha=0.7)
    ax.set_xticks(range(len(mean_errors)))
    ax.set_xticklabels(valid_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Mean Pointing Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, mean_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}°', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Error vs N removed
    ax = fig.add_subplot(gs[0, 1])
    n_values = [1, 2, 3]
    
    gt_means = [stats['global'][f'remove_gt_n{n}']['mean'] for n in n_values 
                if f'remove_gt_n{n}' in stats['global']]
    pred_means = [stats['global'][f'remove_pred_n{n}']['mean'] for n in n_values 
                  if f'remove_pred_n{n}' in stats['global']]
    
    if gt_means:
        ax.plot(n_values[:len(gt_means)], gt_means, 'b-o', label='Ground Truth', linewidth=2)
    if pred_means:
        ax.plot(n_values[:len(pred_means)], pred_means, 'r-s', label='Predicted', linewidth=2)
    
    ax.axhline(y=stats['global']['with_alb']['mean'], color='red', 
              linestyle='--', alpha=0.5, label='With Albedo')
    ax.axhline(y=stats['global']['no_alb']['mean'], color='green', 
              linestyle='--', alpha=0.5, label='No Albedo')
    
    ax.set_xlabel('Number of CSS Removed')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Error vs Number of CSS Removed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    
    # Plot 3: Improvement over baseline
    ax = fig.add_subplot(gs[0, 2])
    baseline = stats['global']['with_alb']['mean']
    
    improvements = []
    improvement_labels = []
    improvement_colors = []
    
    for n in n_values:
        if f'remove_gt_n{n}' in stats['global']:
            imp = ((baseline - stats['global'][f'remove_gt_n{n}']['mean']) / baseline) * 100
            improvements.append(imp)
            improvement_labels.append(f'GT n={n}')
            improvement_colors.append('blue')
        
        if f'remove_pred_n{n}' in stats['global']:
            imp = ((baseline - stats['global'][f'remove_pred_n{n}']['mean']) / baseline) * 100
            improvements.append(imp)
            improvement_labels.append(f'Pred n={n}')
            improvement_colors.append('orange')
    
    if 'no_alb' in stats['global']:
        imp = ((baseline - stats['global']['no_alb']['mean']) / baseline) * 100
        improvements.append(imp)
        improvement_labels.append('No Albedo')
        improvement_colors.append('green')
    
    bars = ax.bar(range(len(improvements)), improvements, color=improvement_colors, alpha=0.7)
    ax.set_xticks(range(len(improvements)))
    ax.set_xticklabels(improvement_labels, rotation=45, ha='right')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Improvement over Baseline (With Albedo)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Plot 4: CSS Removal Frequency (Ground Truth)
    ax = fig.add_subplot(gs[1, 0])
    gt_counts = removal_tracking['gt_removal_counts']
    css_indices = list(range(ncss))
    
    bars = ax.bar(css_indices, gt_counts, color='blue', alpha=0.7)
    ax.set_xlabel('CSS Index')
    ax.set_ylabel('Times Selected for Removal')
    ax.set_title('Ground Truth: CSS Removal Frequency')
    ax.set_xticks(css_indices)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: CSS Removal Frequency (Predicted)
    ax = fig.add_subplot(gs[1, 1])
    pred_counts = removal_tracking['pred_removal_counts']
    
    bars = ax.bar(css_indices, pred_counts, color='orange', alpha=0.7)
    ax.set_xlabel('CSS Index')
    ax.set_ylabel('Times Selected for Removal')
    ax.set_title('Predicted: CSS Removal Frequency')
    ax.set_xticks(css_indices)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Overlap Statistics
    ax = fig.add_subplot(gs[1, 2])
    overlap_data = []
    overlap_labels = []
    
    for n in [1, 2, 3]:
        if f'overlap_n{n}' in removal_tracking:
            overlap_pct = removal_tracking[f'overlap_n{n}']['percent']
            overlap_data.append(overlap_pct)
            overlap_labels.append(f'n={n}')
    
    if overlap_data:
        bars = ax.bar(range(len(overlap_data)), overlap_data, color='purple', alpha=0.7)
        ax.set_xticks(range(len(overlap_data)))
        ax.set_xticklabels(overlap_labels)
        ax.set_ylabel('Overlap Percentage (%)')
        ax.set_title('GT vs Predicted Selection Overlap')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, overlap_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom')
    
    # Plot 7: P95 Error Comparison
    ax = fig.add_subplot(gs[2, 0])
    p95_errors = []
    p95_labels = []
    p95_colors = []
    
    for m, l, c in zip(['with_alb', 'no_alb', 'remove_gt_n2', 'remove_pred_n2'],
                       ['With Alb', 'No Alb', 'GT n=2', 'Pred n=2'],
                       ['red', 'green', 'blue', 'orange']):
        if m in stats['global']:
            p95_errors.append(stats['global'][m]['p95'])
            p95_labels.append(l)
            p95_colors.append(c)
    
    bars = ax.bar(range(len(p95_errors)), p95_errors, color=p95_colors, alpha=0.7)
    ax.set_xticks(range(len(p95_errors)))
    ax.set_xticklabels(p95_labels)
    ax.set_ylabel('95th Percentile Error (degrees)')
    ax.set_title('P95 Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Valid Reconstruction Rate
    ax = fig.add_subplot(gs[2, 1])
    valid_pcts = []
    valid_labels = []
    valid_colors = []
    
    for m, l, c in zip(['with_alb', 'no_alb', 'remove_gt_n2', 'remove_pred_n2'],
                       ['With Alb', 'No Alb', 'GT n=2', 'Pred n=2'],
                       ['red', 'green', 'blue', 'orange']):
        if m in stats['global']:
            valid_pcts.append(stats['global'][m]['valid_pct'])
            valid_labels.append(l)
            valid_colors.append(c)
    
    bars = ax.bar(range(len(valid_pcts)), valid_pcts, color=valid_colors, alpha=0.7)
    ax.set_xticks(range(len(valid_pcts)))
    ax.set_xticklabels(valid_labels)
    ax.set_ylabel('Valid Reconstructions (%)')
    ax.set_title('Reconstruction Success Rate')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Summary Table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Method', 'Mean', 'P95', 'Valid%'])
    
    for method, label in [('with_alb', 'With Albedo'),
                          ('no_alb', 'No Albedo'),
                          ('remove_gt_n2', 'GT Remove 2'),
                          ('remove_pred_n2', 'Pred Remove 2')]:
        if method in stats['global']:
            s = stats['global'][method]
            table_data.append([label, 
                             f"{s['mean']:.1f}°",
                             f"{s['p95']:.1f}°",
                             f"{s['valid_pct']:.1f}%"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('CSS Albedo Removal Strategy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "removal_analysis_final.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="CSS Albedo Analysis with Dynamic Removal Strategies")
        ap.add_argument("--local-dir", required=True, 
                       help="Path to directory containing NOS3InOut folder")
        ap.add_argument("--dest", required=True,
                       help="Destination directory for analysis results")
        args = ap.parse_args()

        # Setup paths
        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        
        print(f"CSS Albedo Removal Analysis - Final Version")
        print(f"="*60)
        print(f"Input: {local_dir}")
        print(f"Output: {dest}")
        
        dest.mkdir(parents=True, exist_ok=True)
        
        # Find NOS3InOut folder
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut folder not found in: {local_dir}")
        
        print(f"\n[1/7] Loading data files...")
        
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
        
        print(f"\n[2/7] CSS Configuration:")
        axis_labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        for i in range(min(6, ncss)):
            label = axis_labels[i] if i < 6 else f'Custom{i}'
            print(f"    CSS{i}: {label} [{axesB[i][0]:.2f}, {axesB[i][1]:.2f}, {axesB[i][2]:.2f}]")
        
        print(f"\n[3/7] Processing timesteps...")
        
        # Initialize tracking
        removal_methods = {
            'with_alb': {'errors': [], 'errors_day': [], 'valid': 0, 'valid_day': 0},
            'no_alb': {'errors': [], 'errors_day': [], 'valid': 0, 'valid_day': 0},
        }
        
        for n in [1, 2, 3]:
            removal_methods[f'remove_gt_n{n}'] = {'errors': [], 'errors_day': [], 'valid': 0, 'valid_day': 0}
            removal_methods[f'remove_pred_n{n}'] = {'errors': [], 'errors_day': [], 'valid': 0, 'valid_day': 0}
        
        # Track removal patterns
        gt_removal_counts = [0] * ncss  # How often each CSS is removed by GT
        pred_removal_counts = [0] * ncss  # How often each CSS is removed by prediction
        overlap_counts = {1: 0, 2: 0, 3: 0}  # How often GT and predicted match
        total_counts = {1: 0, 2: 0, 3: 0}
        
        daylight_count = 0
        
        # Main processing loop
        for k in range(nsteps):
            # Get current state
            rN = posN[k]
            rhatN = unit(rN)
            CBN = quat_to_cbn(qbn[k])
            CNB = mattranspose(CBN)
            sN = unit(svn[k])
            sB_truth = unit(svb[k])
            
            is_day = dot(sN, rhatN) > 0
            if is_day:
                daylight_count += 1
            
            # Get measurements
            ill_row = ill[k][:ncss] if len(ill[k]) >= ncss else ill[k] + [0.0]*(ncss-len(ill[k]))
            alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))
            
            # === DYNAMIC GROUND TRUTH REMOVAL ===
            # Sort by actual albedo values at THIS timestep
            current_gt_ranking = sorted(range(ncss), key=lambda i: alb_row[i], reverse=True)
            
            # === DYNAMIC PREDICTED REMOVAL ===
            # Calculate CSS-specific phase angles at THIS timestep
            css_phase_angles = []
            for i in range(ncss):
                css_axis_B = axesB[i]
                css_axis_N = matvec(CNB, css_axis_B)
                css_phase = calculate_css_specific_phase_angle_simple(css_axis_N, sN, rN)
                css_phase_angles.append(css_phase)
            
            # Sort by phase angle - SMALLEST phase angles have highest albedo effect
            current_pred_ranking = sorted(range(ncss), key=lambda i: css_phase_angles[i])
            
            # Method 1: With albedo (baseline)
            sB_with = css_reconstruct_sun_direction(ill_row, axesB)
            if sB_with is not None:
                error = angle_deg(sB_with, sB_truth)
                removal_methods['with_alb']['errors'].append(error)
                removal_methods['with_alb']['valid'] += 1
                if is_day:
                    removal_methods['with_alb']['errors_day'].append(error)
                    removal_methods['with_alb']['valid_day'] += 1
            
            # Method 2: No albedo (ideal)
            no_alb_vec = [max(0.0, ill_row[i] - alb_row[i]) for i in range(ncss)]
            sB_no = css_reconstruct_sun_direction(no_alb_vec, axesB)
            if sB_no is not None:
                error = angle_deg(sB_no, sB_truth)
                removal_methods['no_alb']['errors'].append(error)
                removal_methods['no_alb']['valid'] += 1
                if is_day:
                    removal_methods['no_alb']['errors_day'].append(error)
                    removal_methods['no_alb']['valid_day'] += 1
            
            # Methods 3-5: Ground truth removal (n=1,2,3)
            for n in [1, 2, 3]:
                # Remove n CSS with highest actual albedo
                exclude_gt = set(current_gt_ranking[:n])
                sB_gt = css_reconstruct_sun_direction(ill_row, axesB, exclude_gt)
                
                if sB_gt is not None:
                    error = angle_deg(sB_gt, sB_truth)
                    removal_methods[f'remove_gt_n{n}']['errors'].append(error)
                    removal_methods[f'remove_gt_n{n}']['valid'] += 1
                    if is_day:
                        removal_methods[f'remove_gt_n{n}']['errors_day'].append(error)
                        removal_methods[f'remove_gt_n{n}']['valid_day'] += 1
                
                # Track which CSS were removed
                if n == 1:
                    for css_idx in exclude_gt:
                        gt_removal_counts[css_idx] += 1
            
            # Methods 6-8: Predicted removal (n=1,2,3)
            for n in [1, 2, 3]:
                # Remove n CSS with smallest phase angle (highest predicted albedo)
                exclude_pred = set(current_pred_ranking[:n])
                sB_pred = css_reconstruct_sun_direction(ill_row, axesB, exclude_pred)
                
                if sB_pred is not None:
                    error = angle_deg(sB_pred, sB_truth)
                    removal_methods[f'remove_pred_n{n}']['errors'].append(error)
                    removal_methods[f'remove_pred_n{n}']['valid'] += 1
                    if is_day:
                        removal_methods[f'remove_pred_n{n}']['errors_day'].append(error)
                        removal_methods[f'remove_pred_n{n}']['valid_day'] += 1
                
                # Track which CSS were removed
                if n == 1:
                    for css_idx in exclude_pred:
                        pred_removal_counts[css_idx] += 1
                
                # Track overlap between GT and predicted
                gt_set = set(current_gt_ranking[:n])
                pred_set = set(current_pred_ranking[:n])
                overlap = len(gt_set & pred_set)
                overlap_counts[n] += overlap
                total_counts[n] += n
        
        print(f"    Daylight steps: {daylight_count}/{nsteps}")
        
        print(f"\n[4/7] Computing statistics...")
        
        # Compute statistics
        stats = {'global': {}, 'daylight': {}}
        
        for method_name, method_data in removal_methods.items():
            # Global statistics
            if method_data['errors']:
                mean, median, p95, max_err, count = angle_stats(method_data['errors'])
                stats['global'][method_name] = {
                    'mean': mean,
                    'median': median,
                    'p95': p95,
                    'max': max_err,
                    'valid': method_data['valid'],
                    'valid_pct': (method_data['valid'] / nsteps) * 100
                }
            
            # Daylight statistics
            if method_data['errors_day']:
                mean, median, p95, max_err, count = angle_stats(method_data['errors_day'])
                stats['daylight'][method_name] = {
                    'mean': mean,
                    'median': median,
                    'p95': p95,
                    'max': max_err,
                    'valid': method_data['valid_day'],
                    'valid_pct': (method_data['valid_day'] / daylight_count) * 100 if daylight_count > 0 else 0
                }
        
        # Prepare removal tracking summary
        removal_tracking = {
            'gt_removal_counts': gt_removal_counts,
            'pred_removal_counts': pred_removal_counts,
        }
        
        for n in [1, 2, 3]:
            removal_tracking[f'overlap_n{n}'] = {
                'count': overlap_counts[n],
                'total': total_counts[n],
                'percent': (overlap_counts[n] / total_counts[n]) * 100 if total_counts[n] > 0 else 0
            }
        
        print(f"\n[5/7] Generating report...")
        
        # Generate detailed report
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("CSS ALBEDO REMOVAL ANALYSIS - FINAL VERSION")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("STRATEGY DESCRIPTION:")
        report_lines.append("-"*40)
        report_lines.append("1. Ground Truth: Removes CSS with highest ACTUAL albedo at each timestep")
        report_lines.append("2. Predicted: Removes CSS with SMALLEST phase angle at each timestep")
        report_lines.append("   (Small phase angle = CSS sees bright Earth = high albedo effect)")
        report_lines.append("")
        
        report_lines.append("GLOBAL STATISTICS")
        report_lines.append("-"*40)
        report_lines.append("Method               | Mean    | Median  | P95     | Valid%")
        report_lines.append("---------------------|---------|---------|---------|-------")
        
        for method in ['with_alb', 'no_alb', 'remove_gt_n1', 'remove_gt_n2', 'remove_gt_n3',
                      'remove_pred_n1', 'remove_pred_n2', 'remove_pred_n3']:
            if method in stats['global']:
                s = stats['global'][method]
                report_lines.append(f"{method:20s} | {s['mean']:7.3f} | {s['median']:7.3f} | "
                                  f"{s['p95']:7.3f} | {s['valid_pct']:6.1f}")
        
        report_lines.append("")
        report_lines.append("IMPROVEMENT OVER BASELINE (with_alb)")
        report_lines.append("-"*40)
        
        baseline = stats['global']['with_alb']['mean']
        report_lines.append("N | Ground Truth  | Predicted     | Difference")
        report_lines.append("--|---------------|---------------|------------")
        
        for n in [1, 2, 3]:
            if f'remove_gt_n{n}' in stats['global'] and f'remove_pred_n{n}' in stats['global']:
                gt_improve = ((baseline - stats['global'][f'remove_gt_n{n}']['mean']) / baseline) * 100
                pred_improve = ((baseline - stats['global'][f'remove_pred_n{n}']['mean']) / baseline) * 100
                diff = abs(gt_improve - pred_improve)
                report_lines.append(f"{n} | {gt_improve:13.1f}% | {pred_improve:13.1f}% | {diff:10.1f}%")
        
        report_lines.append("")
        report_lines.append("REMOVAL PATTERN ANALYSIS")
        report_lines.append("-"*40)
        
        # Most frequently removed CSS
        gt_most_removed = sorted(range(ncss), key=lambda i: gt_removal_counts[i], reverse=True)[:3]
        pred_most_removed = sorted(range(ncss), key=lambda i: pred_removal_counts[i], reverse=True)[:3]
        
        report_lines.append("Most frequently removed CSS (Ground Truth):")
        for i in gt_most_removed:
            pct = (gt_removal_counts[i] / nsteps) * 100
            report_lines.append(f"  CSS{i}: {gt_removal_counts[i]} times ({pct:.1f}%)")
        
        report_lines.append("")
        report_lines.append("Most frequently removed CSS (Predicted):")
        for i in pred_most_removed:
            pct = (pred_removal_counts[i] / nsteps) * 100
            report_lines.append(f"  CSS{i}: {pred_removal_counts[i]} times ({pct:.1f}%)")
        
        report_lines.append("")
        report_lines.append("PREDICTION ACCURACY")
        report_lines.append("-"*40)
        report_lines.append("Overlap between GT and Predicted selections:")
        for n in [1, 2, 3]:
            overlap_pct = removal_tracking[f'overlap_n{n}']['percent']
            report_lines.append(f"  n={n}: {overlap_pct:.1f}% overlap")
        
        avg_overlap = np.mean([removal_tracking[f'overlap_n{n}']['percent'] for n in [1,2,3]])
        report_lines.append(f"\nAverage overlap: {avg_overlap:.1f}%")
        
        if avg_overlap > 70:
            report_lines.append(" Excellent prediction accuracy using phase angles")
        elif avg_overlap > 50:
            report_lines.append(" Good prediction accuracy using phase angles")
        else:
            report_lines.append(" Limited prediction accuracy using phase angles")
        
        # Save report
        report_file = dest / "removal_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save CSV with detailed statistics
        csv_file = dest / "removal_statistics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'scope', 'mean_error', 'median_error', 'p95_error', 
                           'max_error', 'valid_count', 'valid_pct'])
            
            for scope in ['global', 'daylight']:
                for method in stats.get(scope, {}):
                    s = stats[scope][method]
                    writer.writerow([method, scope, s['mean'], s['median'], 
                                   s['p95'], s['max'], s['valid'], s['valid_pct']])
        
        print(f"\n[6/7] Creating visualizations...")
        viz_path = create_comparison_plots(stats, removal_tracking, dest, ncss)
        
        print(f"\n[7/7] Analysis complete!")
        
        # Print summary
        print(f"\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Baseline (with_alb):     {stats['global']['with_alb']['mean']:.1f}°")
        print(f"Ideal (no_alb):          {stats['global']['no_alb']['mean']:.1f}°")
        
        for n in [1, 2, 3]:
            if f'remove_gt_n{n}' in stats['global']:
                print(f"Ground Truth (n={n}):      {stats['global'][f'remove_gt_n{n}']['mean']:.1f}°")
            if f'remove_pred_n{n}' in stats['global']:
                print(f"Predicted (n={n}):        {stats['global'][f'remove_pred_n{n}']['mean']:.1f}°")
        
        print(f"\nPrediction Accuracy: {avg_overlap:.1f}% average overlap")
        
        print(f"\nOutput files:")
        print(f"  • {report_file.name}")
        print(f"  • {csv_file.name}")
        print(f"  • {viz_path.name}")
        print(f"\nAll files saved to: {dest}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()