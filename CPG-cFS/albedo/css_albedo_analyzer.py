#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS Albedo Effect Analysis Script (Fixed Version)
Analyzes the relationship between CSS position and albedo effect magnitude
"""

import argparse, os, sys, math, re, csv, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

def quat_to_cbn(q):
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
    return [C[0][0]*v[0]+C[0][1]*v[1]+C[0][2]*v[2],
            C[1][0]*v[0]+C[1][1]*v[1]+C[1][2]*v[2],
            C[2][0]*v[0]+C[2][1]*v[1]+C[2][2]*v[2]]

def angle_deg(a,b):
    aa, bb = unit(a), unit(b)
    c = max(-1.0, min(1.0, dot(aa,bb)))
    return math.degrees(math.acos(c))

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
def css_reconstruct_direction(intensities, axesB, fov_deg, nadirB, exclude_indices=None):
    wsum = [0.0, 0.0, 0.0]
    total = 0.0
    for i, (I, axB, hf) in enumerate(zip(intensities, axesB, fov_deg)):
        if exclude_indices and i in exclude_indices:
            continue
        th = angle_deg(axB, nadirB)
        if th <= hf + 1e-9 and I > 0.0:
            wsum[0] += I * axB[0]
            wsum[1] += I * axB[1]
            wsum[2] += I * axB[2]
            total += I
    if total <= 0.0:
        return None
    return unit(wsum)

def angle_stats(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return (math.nan, math.nan, math.nan, math.nan, 0)
    return (float(a.mean()), float(np.median(a)), float(np.percentile(a,95)), float(a.max()), int(a.size))

# ---------- Visualization functions ----------
def create_albedo_visualizations(alb_data, css_stats_list, dest_dir, ncss, nsteps, daylight_mask):
    """albedo"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    #  - 
    alb_list_normalized = []
    for row in alb_data:
        if len(row) < ncss:
            normalized_row = list(row) + [0.0] * (ncss - len(row))
        else:
            normalized_row = row[:ncss]
        alb_list_normalized.append(normalized_row)
    
    alb_array = np.array(alb_list_normalized)
    
    # 1: Albedo
    ax1 = fig.add_subplot(gs[0, :])
    sample_interval = max(1, nsteps // 500)
    sample_indices = range(0, min(nsteps, len(alb_array)), sample_interval)
    for i in range(min(ncss, 10)):
        sampled_data = alb_array[::sample_interval, i][:len(sample_indices)]
        ax1.plot(sample_indices, sampled_data, 
                label=f'CSS{i}', alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Albedo Intensity')
    ax1.set_title('Albedo Intensity Time Series (Sampled)')
    ax1.grid(True, alpha=0.3)
    if ncss <= 10:
        ax1.legend(loc='upper right', ncol=2, fontsize=8)
    
    # 2: Albedo
    ax2 = fig.add_subplot(gs[1, 0])
    all_albedo = alb_array.flatten()
    all_albedo = all_albedo[~np.isnan(all_albedo)]
    if len(all_albedo) > 0:
        ax2.hist(all_albedo, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Albedo Intensity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of All Albedo Values')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(all_albedo), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_albedo):.3f}')
        ax2.axvline(np.median(all_albedo), color='green', linestyle='--', 
                    label=f'Median: {np.median(all_albedo):.3f}')
        ax2.legend()
    
    # 3: CSSvsAlbedo
    ax3 = fig.add_subplot(gs[1, 1])
    mean_angles = [css_stat['mean_nadir_angle'] for css_stat in css_stats_list]
    mean_albedos = [css_stat['mean_albedo'] for css_stat in css_stats_list]
    
    scatter = ax3.scatter(mean_angles, mean_albedos, s=100, alpha=0.6, c=range(ncss), cmap='viridis')
    ax3.set_xlabel('Mean Angle to Nadir (deg)')
    ax3.set_ylabel('Mean Albedo Intensity')
    ax3.set_title('CSS Position vs Albedo Effect')
    ax3.grid(True, alpha=0.3)
    
    # 
    valid_mask = ~(np.isnan(mean_angles) | np.isnan(mean_albedos))
    valid_mask = np.array(valid_mask)
    if np.sum(valid_mask) > 2:
        angles_valid = np.array(mean_angles)[valid_mask]
        albedos_valid = np.array(mean_albedos)[valid_mask]
        z = np.polyfit(angles_valid, albedos_valid, 1)
        p = np.poly1d(z)
        ax3.plot(sorted(angles_valid), p(sorted(angles_valid)), 
                "r--", alpha=0.5, label=f'Trend: y={z[0]:.3e}x+{z[1]:.3f}')
        
        corr, p_value = scipy_stats.pearsonr(angles_valid, albedos_valid)
        ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3e}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.legend()
    if ncss <= 20:
        plt.colorbar(scatter, ax=ax3, label='CSS Index')
    
    # 4: Albedo
    ax4 = fig.add_subplot(gs[2, :])
    heatmap_sample = max(1, len(alb_array) // 200)
    heatmap_indices = range(0, len(alb_array), heatmap_sample)
    heatmap_data = alb_array[::heatmap_sample, :].T
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
    ax4.set_xlabel('Time Step (Sampled)')
    ax4.set_ylabel('CSS Index')
    ax4.set_title('Albedo Contribution Heatmap')
    
    if ncss <= 20:
        ax4.set_yticks(range(ncss))
        ax4.set_yticklabels([f'CSS{i}' for i in range(ncss)])
    
    plt.colorbar(im, ax=ax4, label='Albedo Intensity')
    
    # /
    if len(daylight_mask) > 0:
        daylight_sampled = daylight_mask[::heatmap_sample]
        for i, is_day in enumerate(daylight_sampled[:heatmap_data.shape[1]]):
            if not is_day:
                ax4.axvline(x=i, color='blue', alpha=0.1, linewidth=0.5)
    
    plt.suptitle('CSS Albedo Effect Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "albedo_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="Analyze CSS albedo effects from local NOS3InOut data.")
        ap.add_argument("--local-dir", required=True, 
                       help="Path to directory containing NOS3InOut folder")
        ap.add_argument("--dest", required=True,
                       help="Destination directory for analysis results")
        args = ap.parse_args()

        # 
        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        
        print(f"Starting CSS Albedo Analysis")
        print(f"Local directory: {local_dir}")
        print(f"Destination: {dest}")
        
        if not local_dir.exists():
            sys.exit(f"Error: Local directory does not exist: {local_dir}")
        
        dest.mkdir(parents=True, exist_ok=True)
        
        # NOS3InOut
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut folder not found in: {local_dir}")
        
        print(f"[1/7] Using local NOS3InOut: {local_nos3}")

        # 2) 
        time_path = local_nos3/"time.42"
        pos_path  = local_nos3/"PosN.42"
        qbn_path  = local_nos3/"qbn.42"
        svn_path  = local_nos3/"svn.42"
        svb_path  = local_nos3/"svb.42"
        alb_path  = local_nos3/"Albedo.42"
        ill_path  = local_nos3/"Illum.42"
        inp_sim   = local_nos3/"Inp_Sim.txt"

        for p in [pos_path,qbn_path,svn_path,svb_path,alb_path,ill_path,inp_sim]:
            if not p.exists():
                sys.exit(f"Missing file: {p}")

        #  SC 
        sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
        sc_path = (local_nos3/sc_file_name) if sc_file_name else (local_nos3/"SC_NOS3.txt")
        if not sc_path.exists():
            sc_path = local_nos3/"SC_SensorFOV.txt"
        if not sc_path.exists():
            print(f"[WARN] Cannot find SC file. Using default ±X/±Y/±Z axes.")
            sc_path = None

        # CSS
        ncss_from_alb = count_cols_first_nonempty(alb_path)
        ncss_from_ill = count_cols_first_nonempty(ill_path)
        ncss = ncss_from_alb if ncss_from_alb>0 else ncss_from_ill
        if ncss <= 0:
            sys.exit("Cannot detect Ncss from Albedo.42 or Illum.42")

        # CSS
        axesB, fov_deg = [], []
        if sc_path:
            axesB, fov_deg = parse_css_from_sc_file(sc_path)
        if not axesB or len(axesB) != ncss:
            print(f"[WARN] CSS count mismatch. Using default axes.")
            default_axes = [
                [ 1, 0, 0], [-1, 0, 0],
                [ 0, 1, 0], [ 0,-1, 0],
                [ 0, 0, 1], [ 0, 0,-1],
            ]
            axesB = ([unit(a) for a in default_axes] * ((ncss+5)//6))[:ncss]
            fov_deg = [90.0]*ncss

        # 
        print("[2/7] Loading data files...")
        times = [ln.strip() for ln in open(time_path)] if time_path.exists() else []
        posN  = read_vector3_first3(pos_path)
        qbn   = read_numeric_cols(qbn_path, need=4, offset=0)
        svn   = read_numeric_cols(svn_path, need=3, offset=0)
        svb   = read_numeric_cols(svb_path, need=3, offset=0)
        alb   = read_numeric_cols(alb_path, need=0)
        ill   = read_numeric_cols(ill_path, need=0)

        # 
        candidates = [len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill)]
        if times:
            candidates.append(len(times))
        nsteps = min([c for c in candidates if c>0]) if candidates else 0
        if nsteps == 0:
            sys.exit("No data rows found (nsteps=0).")

        # /
        daylight_mask = []
        for k in range(nsteps):
            rhatN = unit(posN[k])
            sN    = unit(svn[k])
            daylight_mask.append(1 if dot(sN, rhatN) > 0 else 0)

        print(f"    Steps: {nsteps}")
        print(f"    Number of CSS: {ncss}")
        print(f"    Daylight frames: {sum(daylight_mask)}/{nsteps}")

        # 3) CSS
        css_stats = []
        for i in range(ncss):
            css_stats.append({
                'index': i,
                'axis': axesB[i],
                'fov': fov_deg[i],
                'albedo_values': [],
                'nadir_angles': [],
                'illum_values': [],
            })

        # 4) 
        print("[3/7] Processing data and calculating errors...")
        err_with, err_no, err_syn, err_remove2 = [], [], [], []
        err_with_day, err_no_day, err_syn_day, err_remove2_day = [], [], [], []
        
        selfcheck_angles = []
        selfcheck_angles_day = []
        
        valid_with = valid_no = valid_syn = valid_remove2 = 0
        valid_with_day = valid_no_day = valid_syn_day = valid_remove2_day = 0

        out_csv = dest/"albedo_geometry_summary.csv"
        with open(out_csv,"w",newline="") as f:
            w=csv.writer(f)
            header=["step_idx","daylight","with_alb_sum","no_alb_sum","remove2_indices"] + \
                    [f"ill_css{i}" for i in range(ncss)] + \
                    [f"alb_css{i}" for i in range(ncss)]
            w.writerow(header)

            for k in range(nsteps):
                rhatN = unit(posN[k])
                CBN   = quat_to_cbn(qbn[k])
                nadirB= matvec(CBN,[-rhatN[0],-rhatN[1],-rhatN[2]])
                sN    = unit(svn[k])
                sB_truth = unit(svb[k])
                is_day = daylight_mask[k]==1

                # 
                calc_sB = unit(matvec(CBN, sN))
                ang_chk = angle_deg(sB_truth, calc_sB)
                selfcheck_angles.append(ang_chk)
                if is_day: 
                    selfcheck_angles_day.append(ang_chk)

                # 
                ill_row = ill[k][:ncss] if len(ill[k]) >= ncss else ill[k] + [0.0]*(ncss-len(ill[k]))
                alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))

                # CSS
                for i in range(ncss):
                    css_stats[i]['albedo_values'].append(alb_row[i])
                    css_stats[i]['illum_values'].append(ill_row[i])
                    angle_to_nadir = angle_deg(axesB[i], nadirB)
                    css_stats[i]['nadir_angles'].append(angle_to_nadir)

                # 
                with_vec = ill_row
                no_vec   = [max(0.0, ill_row[i]-alb_row[i]) for i in range(ncss)]
                syn_vec  = no_vec[:]
                
                # albedo2CSS
                alb_indices_sorted = sorted(range(ncss), key=lambda i: alb_row[i], reverse=True)
                top2_indices = alb_indices_sorted[:2] if ncss >= 2 else []
                
                # remove2
                remove2_vec = with_vec[:]

                # 
                sB_with = css_reconstruct_direction(with_vec, axesB, fov_deg, nadirB)
                sB_no   = css_reconstruct_direction(no_vec,   axesB, fov_deg, nadirB)
                sB_syn  = css_reconstruct_direction(syn_vec,  axesB, fov_deg, nadirB)
                sB_remove2 = css_reconstruct_direction(remove2_vec, axesB, fov_deg, nadirB, 
                                                      exclude_indices=top2_indices)

                # 
                if sB_with is not None:
                    valid_with += 1
                    err_with.append(angle_deg(sB_with, sB_truth))
                    if is_day:
                        valid_with_day += 1
                        err_with_day.append(angle_deg(sB_with, sB_truth))

                if sB_no is not None:
                    valid_no += 1
                    err_no.append(angle_deg(sB_no, sB_truth))
                    if is_day:
                        valid_no_day += 1
                        err_no_day.append(angle_deg(sB_no, sB_truth))

                if sB_syn is not None:
                    valid_syn += 1
                    err_syn.append(angle_deg(sB_syn, sB_truth))
                    if is_day:
                        valid_syn_day += 1
                        err_syn_day.append(angle_deg(sB_syn, sB_truth))
                
                if sB_remove2 is not None:
                    valid_remove2 += 1
                    err_remove2.append(angle_deg(sB_remove2, sB_truth))
                    if is_day:
                        valid_remove2_day += 1
                        err_remove2_day.append(angle_deg(sB_remove2, sB_truth))

                # CSV
                remove2_str = f"{top2_indices[0]},{top2_indices[1]}" if len(top2_indices)==2 else ""
                w.writerow([k+1, int(is_day), sum(with_vec), sum(no_vec), remove2_str] +
                           [f"{v:.6g}" for v in ill_row] + [f"{v:.6g}" for v in alb_row])

        # 5) CSS
        print("[4/7] Computing CSS albedo statistics...")
        css_position_data = []
        
        # CSS
        for i, css_stat in enumerate(css_stats):
            alb_vals = np.array(css_stat['albedo_values'])
            nadir_angles = np.array(css_stat['nadir_angles'])
            
            valid_mask = ~np.isnan(alb_vals)
            alb_valid = alb_vals[valid_mask]
            
            mean_alb = np.mean(alb_valid) if len(alb_valid) > 0 else 0
            std_alb = np.std(alb_valid) if len(alb_valid) > 0 else 0
            max_alb = np.max(alb_valid) if len(alb_valid) > 0 else 0
            mean_angle = np.mean(nadir_angles[valid_mask]) if np.sum(valid_mask) > 0 else 0
            
            css_stat['mean_albedo'] = mean_alb
            css_stat['std_albedo'] = std_alb
            css_stat['max_albedo'] = max_alb
            css_stat['mean_nadir_angle'] = mean_angle
        
        # 
        total_mean_albedo = sum(s['mean_albedo'] for s in css_stats)
        
        # 
        for i, css_stat in enumerate(css_stats):
            css_position_data.append({
                'css_index': i,
                'axis_x': css_stat['axis'][0],
                'axis_y': css_stat['axis'][1],
                'axis_z': css_stat['axis'][2],
                'fov_deg': css_stat['fov'],
                'mean_albedo': css_stat['mean_albedo'],
                'std_albedo': css_stat['std_albedo'],
                'max_albedo': css_stat['max_albedo'],
                'mean_nadir_angle': css_stat['mean_nadir_angle'],
                'albedo_contribution_percent': (css_stat['mean_albedo'] / (total_mean_albedo + 1e-10)) * 100
            })

        # CSS
        css_stats_csv = dest/"css_albedo_statistics.csv"
        with open(css_stats_csv, 'w', newline='') as f:
            fieldnames = list(css_position_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(css_position_data)

        # 6) 
        print("[5/7] Analyzing position-albedo correlation...")
        mean_angles = [s['mean_nadir_angle'] for s in css_stats]
        mean_albedos = [s['mean_albedo'] for s in css_stats]
        
        valid_pairs = [(a, b) for a, b in zip(mean_angles, mean_albedos) 
                       if not (math.isnan(a) or math.isnan(b))]
        
        corr_file = dest/"css_position_correlation.txt"
        with open(corr_file, 'w') as f:
            f.write("CSS Position vs Albedo Effect Correlation Analysis\n")
            f.write("="*50 + "\n\n")
            f.write(f"Number of CSS sensors: {ncss}\n")
            f.write(f"Valid data points: {len(valid_pairs)}\n\n")
            
            if len(valid_pairs) > 2:
                angles_arr = np.array([p[0] for p in valid_pairs])
                albedos_arr = np.array([p[1] for p in valid_pairs])
                correlation, p_value = scipy_stats.pearsonr(angles_arr, albedos_arr)
                
                f.write(f"Pearson correlation coefficient: {correlation:.4f}\n")
                f.write(f"P-value: {p_value:.4e}\n\n")
                
                if abs(correlation) > 0.7:
                    f.write("Strong correlation detected!\n")
                elif abs(correlation) > 0.4:
                    f.write("Moderate correlation detected.\n")
                else:
                    f.write("Weak or no correlation detected.\n")
                
                if correlation < 0:
                    f.write("\nCSS facing toward Earth (smaller nadir angles) receive MORE albedo.\n")
                else:
                    f.write("\nCSS facing away from Earth (larger nadir angles) receive MORE albedo.\n")
            else:
                f.write("Insufficient valid data for correlation analysis.\n")

        # 7) 
        print("[6/7] Generating visualizations...")
        viz_path = create_albedo_visualizations(alb, css_stats, dest, ncss, nsteps, daylight_mask)

        # 8) 
        print("\n[7/7] RESULTS SUMMARY")
        print("="*60)
        
        print("\nReconstruction Availability:")
        print(f"  with_alb:   {valid_with}/{nsteps} ({100*valid_with/nsteps:.1f}%)")
        print(f"  no_alb:     {valid_no}/{nsteps} ({100*valid_no/nsteps:.1f}%)")
        print(f"  synth_dir:  {valid_syn}/{nsteps} ({100*valid_syn/nsteps:.1f}%)")
        print(f"  remove2alb: {valid_remove2}/{nsteps} ({100*valid_remove2/nsteps:.1f}%)")

        def print_error_stats(tag, e_with, e_no, e_syn, e_remove2):
            print(f"\n{tag}:")
            m1, md1, p951, mx1, _ = angle_stats(e_with)
            m2, md2, p952, mx2, _ = angle_stats(e_no)
            m3, md3, p953, mx3, _ = angle_stats(e_syn)
            m4, md4, p954, mx4, _ = angle_stats(e_remove2)
            
            print(f"  with_alb:   mean={m1:.3f}°, median={md1:.3f}°, p95={p951:.3f}°, max={mx1:.3f}°")
            print(f"  no_alb:     mean={m2:.3f}°, median={md2:.3f}°, p95={p952:.3f}°, max={mx2:.3f}°")
            print(f"  synth_dir:  mean={m3:.3f}°, median={md3:.3f}°, p95={p953:.3f}°, max={mx3:.3f}°")
            print(f"  remove2alb: mean={m4:.3f}°, median={md4:.3f}°, p95={p954:.3f}°, max={mx4:.3f}°")
            
            if not math.isnan(m1) and not math.isnan(m2):
                improve_no = ((m1 - m2) / m1) * 100 if m1 != 0 else 0
                print(f"\n  Improvement (no_alb vs with_alb): {improve_no:.1f}%")
            if not math.isnan(m1) and not math.isnan(m4):
                improve_r2 = ((m1 - m4) / m1) * 100 if m1 != 0 else 0
                print(f"  Improvement (remove2 vs with_alb): {improve_r2:.1f}%")
            
            return {'with': m1, 'no': m2, 'syn': m3, 'remove2': m4}

        global_stats = print_error_stats("Pointing Error - GLOBAL", 
                                        err_with, err_no, err_syn, err_remove2)
        daylight_stats = print_error_stats("Pointing Error - DAYLIGHT ONLY", 
                                          err_with_day, err_no_day, err_syn_day, err_remove2_day)

        # 
        comparison_file = dest/"pointing_error_comparison.txt"
        with open(comparison_file, 'w') as f:
            f.write("Pointing Error Comparison Report\n")
            f.write("="*60 + "\n\n")
            f.write("GLOBAL: with_alb={:.3f}°, no_alb={:.3f}°, remove2={:.3f}°\n".format(
                    global_stats['with'], global_stats['no'], global_stats['remove2']))
            f.write("DAYLIGHT: with_alb={:.3f}°, no_alb={:.3f}°, remove2={:.3f}°\n".format(
                    daylight_stats['with'], daylight_stats['no'], daylight_stats['remove2']))

        print(f"\n All results saved to: {dest}")
        print(f"  - {out_csv.name}")
        print(f"  - {css_stats_csv.name}")
        print(f"  - {corr_file.name}")
        print(f"  - {comparison_file.name}")
        print(f"  - {viz_path.name}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()