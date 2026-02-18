#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS Albedo Factors Deep Analysis
Explores multiple factors affecting albedo on CSS sensors
"""

import argparse, os, sys, math, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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

# ---------- Visualization functions ----------
def create_multifactor_visualizations(analysis_data, dest_dir, ncss):
    """"""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Albedo Ratio
    ax1 = fig.add_subplot(gs[0, :])
    nsteps = len(analysis_data['time_steps'])
    sample_interval = max(1, nsteps // 500)
    sample_indices = range(0, nsteps, sample_interval)
    
    for i in range(min(ncss, 6)):  # 6CSS
        ratios = [analysis_data['time_steps'][j]['css_albedo_ratios'][i] 
                 for j in sample_indices if j < nsteps]
        ax1.plot(sample_indices[:len(ratios)], ratios, 
                label=f'CSS{i}', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Time Step (Sampled)')
    ax1.set_ylabel('Albedo Ratio (Albedo/Total Illum)')
    ax1.set_title('Albedo Contribution Ratio Over Time')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 2. Albedo Ratio
    ax2 = fig.add_subplot(gs[1, 0])
    all_ratios = []
    for step in analysis_data['time_steps']:
        all_ratios.extend(step['css_albedo_ratios'])
    all_ratios = [r for r in all_ratios if not math.isnan(r) and r > 0]
    
    if all_ratios:
        ax2.hist(all_ratios, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Albedo Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Albedo Ratios')
        ax2.axvline(np.mean(all_ratios), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_ratios):.3f}')
        ax2.axvline(np.median(all_ratios), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_ratios):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. CSS-Sun Angle vs Albedo
    ax3 = fig.add_subplot(gs[1, 1])
    sun_angles = []
    albedo_values = []
    colors = []
    
    for css_idx in range(ncss):
        css_sun = analysis_data['css_stats'][css_idx]['sun_angles']
        css_alb = analysis_data['css_stats'][css_idx]['albedo_values']
        valid_mask = ~(np.isnan(css_sun) | np.isnan(css_alb))
        sun_angles.extend(np.array(css_sun)[valid_mask].tolist())
        albedo_values.extend(np.array(css_alb)[valid_mask].tolist())
        colors.extend([css_idx] * np.sum(valid_mask))
    
    if sun_angles:
        scatter = ax3.scatter(sun_angles, albedo_values, c=colors, 
                            cmap='viridis', alpha=0.3, s=1)
        ax3.set_xlabel('CSS-Sun Angle (degrees)')
        ax3.set_ylabel('Albedo Intensity')
        ax3.set_title('Sun Angle vs Albedo Effect')
        ax3.set_xlim([0, 180])
        ax3.set_ylim([0, 0.3])
        ax3.grid(True, alpha=0.3)
        
        # 
        if len(sun_angles) > 10:
            z = np.polyfit(sun_angles, albedo_values, 2)  # 
            x_trend = np.linspace(0, 180, 100)
            y_trend = np.polyval(z, x_trend)
            ax3.plot(x_trend, y_trend, 'r--', alpha=0.5, linewidth=2)
    
    # 4. 2DNadir Angle vs Sun Angle
    ax4 = fig.add_subplot(gs[1, 2])
    
    # 2D
    nadir_angles_all = []
    sun_angles_all = []
    albedo_values_all = []
    
    for step in analysis_data['time_steps']:
        for i in range(ncss):
            if not math.isnan(step['css_nadir_angles'][i]) and \
               not math.isnan(step['css_sun_angles'][i]) and \
               step['css_albedo_values'][i] > 0:
                nadir_angles_all.append(step['css_nadir_angles'][i])
                sun_angles_all.append(step['css_sun_angles'][i])
                albedo_values_all.append(step['css_albedo_values'][i])
    
    if nadir_angles_all:
        # 2D bins
        nadir_bins = np.linspace(0, 180, 20)
        sun_bins = np.linspace(0, 180, 20)
        
        # binalbedo
        H = np.zeros((len(sun_bins)-1, len(nadir_bins)-1))
        counts = np.zeros((len(sun_bins)-1, len(nadir_bins)-1))
        
        for n_ang, s_ang, alb in zip(nadir_angles_all, sun_angles_all, albedo_values_all):
            n_idx = np.searchsorted(nadir_bins, n_ang) - 1
            s_idx = np.searchsorted(sun_bins, s_ang) - 1
            if 0 <= n_idx < len(nadir_bins)-1 and 0 <= s_idx < len(sun_bins)-1:
                H[s_idx, n_idx] += alb
                counts[s_idx, n_idx] += 1
        
        # 
        H_avg = np.divide(H, counts, where=counts>0)
        H_avg[counts==0] = np.nan
        
        im = ax4.imshow(H_avg, origin='lower', aspect='auto', 
                       extent=[0, 180, 0, 180], cmap='hot', vmin=0, vmax=0.1)
        ax4.set_xlabel('Nadir Angle (degrees)')
        ax4.set_ylabel('Sun Angle (degrees)')
        ax4.set_title('Average Albedo: Nadir vs Sun Angle')
        plt.colorbar(im, ax=ax4, label='Avg Albedo')
    
    # 5. Phase Angle
    ax5 = fig.add_subplot(gs[2, 0])
    phase_angles = []
    albedo_for_phase = []
    
    for step in analysis_data['time_steps']:
        if not math.isnan(step['phase_angle']):
            phase_angles.append(step['phase_angle'])
            albedo_for_phase.append(step['total_albedo'])
    
    if phase_angles:
        ax5.scatter(phase_angles, albedo_for_phase, alpha=0.3, s=1)
        ax5.set_xlabel('Phase Angle (degrees)')
        ax5.set_ylabel('Total Albedo')
        ax5.set_title('Phase Angle vs Total Albedo')
        ax5.grid(True, alpha=0.3)
        
        # 
        if len(phase_angles) > 20:
            sorted_indices = np.argsort(phase_angles)
            phase_sorted = np.array(phase_angles)[sorted_indices]
            albedo_sorted = np.array(albedo_for_phase)[sorted_indices]
            
            window = min(50, len(phase_sorted)//10)
            phase_smooth = np.convolve(phase_sorted, np.ones(window)/window, mode='valid')
            albedo_smooth = np.convolve(albedo_sorted, np.ones(window)/window, mode='valid')
            ax5.plot(phase_smooth, albedo_smooth, 'r-', alpha=0.7, linewidth=2)
    
    # 6. 
    ax6 = fig.add_subplot(gs[2, 1])
    altitudes = []
    albedo_for_alt = []
    
    for step in analysis_data['time_steps']:
        if not math.isnan(step['altitude']):
            altitudes.append(step['altitude'])
            albedo_for_alt.append(step['total_albedo'])
    
    if altitudes:
        ax6.scatter(altitudes, albedo_for_alt, alpha=0.3, s=1)
        ax6.set_xlabel('Altitude (km)')
        ax6.set_ylabel('Total Albedo')
        ax6.set_title('Altitude vs Total Albedo')
        ax6.grid(True, alpha=0.3)
        
        # 
        if len(altitudes) > 10:
            z = np.polyfit(altitudes, albedo_for_alt, 1)
            x_trend = np.linspace(min(altitudes), max(altitudes), 100)
            y_trend = np.polyval(z, x_trend)
            ax6.plot(x_trend, y_trend, 'r--', alpha=0.5, linewidth=2)
    
    # 7. 
    ax7 = fig.add_subplot(gs[2, 2])
    
    # 
    corr_data = {
        'Nadir Angle': [],
        'Sun Angle': [],
        'Phase Angle': [],
        'Altitude': [],
        'Albedo': []
    }
    
    for step in analysis_data['time_steps']:
        if step['total_albedo'] > 0:  # albedo
            avg_nadir = np.nanmean(step['css_nadir_angles'])
            avg_sun = np.nanmean(step['css_sun_angles'])
            if not math.isnan(avg_nadir) and not math.isnan(avg_sun):
                corr_data['Nadir Angle'].append(avg_nadir)
                corr_data['Sun Angle'].append(avg_sun)
                corr_data['Phase Angle'].append(step['phase_angle'])
                corr_data['Altitude'].append(step['altitude'])
                corr_data['Albedo'].append(step['total_albedo'])
    
    if len(corr_data['Albedo']) > 10:
        # 
        corr_matrix = np.zeros((5, 5))
        labels = list(corr_data.keys())
        
        for i, key1 in enumerate(labels):
            for j, key2 in enumerate(labels):
                valid_mask = ~(np.isnan(corr_data[key1]) | np.isnan(corr_data[key2]))
                if np.sum(valid_mask) > 2:
                    corr, _ = scipy_stats.pearsonr(
                        np.array(corr_data[key1])[valid_mask],
                        np.array(corr_data[key2])[valid_mask]
                    )
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = 0
        
        im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax7.set_xticks(range(5))
        ax7.set_yticks(range(5))
        ax7.set_xticklabels(labels, rotation=45, ha='right')
        ax7.set_yticklabels(labels)
        ax7.set_title('Correlation Matrix')
        
        # 
        for i in range(5):
            for j in range(5):
                text = ax7.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax7, label='Correlation')
    
    plt.suptitle('CSS Albedo Multi-Factor Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "albedo_factors_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="Deep analysis of CSS albedo factors")
        ap.add_argument("--local-dir", required=True, 
                       help="Path to directory containing NOS3InOut folder")
        ap.add_argument("--dest", required=True,
                       help="Destination directory for analysis results")
        args = ap.parse_args()

        # 
        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        
        print(f"CSS Albedo Factors Deep Analysis")
        print(f"="*50)
        print(f"Input: {local_dir}")
        print(f"Output: {dest}")
        
        dest.mkdir(parents=True, exist_ok=True)
        
        # NOS3InOut
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut folder not found in: {local_dir}")
        
        print(f"\n[1/6] Loading data files...")
        
        # 
        pos_path  = local_nos3/"PosN.42"
        qbn_path  = local_nos3/"qbn.42"
        svn_path  = local_nos3/"svn.42"
        svb_path  = local_nos3/"svb.42"
        alb_path  = local_nos3/"Albedo.42"
        ill_path  = local_nos3/"Illum.42"
        inp_sim   = local_nos3/"Inp_Sim.txt"
        
        # 
        for p in [pos_path,qbn_path,svn_path,svb_path,alb_path,ill_path,inp_sim]:
            if not p.exists():
                sys.exit(f"Missing file: {p}")
        
        # CSS
        sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
        sc_path = (local_nos3/sc_file_name) if sc_file_name else (local_nos3/"SC_NOS3.txt")
        if not sc_path.exists():
            sc_path = local_nos3/"SC_SensorFOV.txt"
        if not sc_path.exists():
            sc_path = None
        
        # CSS
        ncss = count_cols_first_nonempty(alb_path)
        if ncss <= 0:
            ncss = count_cols_first_nonempty(ill_path)
        if ncss <= 0:
            sys.exit("Cannot detect number of CSS")
        
        print(f"    Number of CSS: {ncss}")
        
        # CSS
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
        
        # 
        posN = read_vector3_first3(pos_path)
        qbn  = read_numeric_cols(qbn_path, need=4, offset=0)
        svn  = read_numeric_cols(svn_path, need=3, offset=0)
        svb  = read_numeric_cols(svb_path, need=3, offset=0)
        alb  = read_numeric_cols(alb_path, need=0)
        ill  = read_numeric_cols(ill_path, need=0)
        
        # 
        nsteps = min(len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill))
        print(f"    Total steps: {nsteps}")
        
        # 
        analysis_data = {
            'time_steps': [],
            'css_stats': [{'sun_angles': [], 'nadir_angles': [], 
                          'albedo_values': [], 'albedo_ratios': []} 
                         for _ in range(ncss)]
        }
        
        print(f"\n[2/6] Calculating multi-factor metrics...")
        
        # 
        for k in range(nsteps):
            # 
            rN = posN[k]
            rhatN = unit(rN)
            altitude = norm(rN) - 6371.0  # 6371km
            
            CBN = quat_to_cbn(qbn[k])
            nadirB = matvec(CBN, [-rhatN[0],-rhatN[1],-rhatN[2]])
            
            sN = unit(svn[k])
            sB = matvec(CBN, sN)
            
            # Phase angle (Sun-Earth-Satellite angle)
            phase_angle = angle_deg(sN, rhatN)
            
            # CSS
            ill_row = ill[k][:ncss] if len(ill[k]) >= ncss else ill[k] + [0.0]*(ncss-len(ill[k]))
            alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))
            
            # CSS
            css_sun_angles = []
            css_nadir_angles = []
            css_albedo_ratios = []
            
            for i in range(ncss):
                # CSS-Sun angle
                sun_angle = angle_deg(axesB[i], sB)
                css_sun_angles.append(sun_angle)
                
                # CSS-Nadir angle
                nadir_angle = angle_deg(axesB[i], nadirB)
                css_nadir_angles.append(nadir_angle)
                
                # Albedo ratio
                if ill_row[i] > 1e-6:
                    ratio = alb_row[i] / ill_row[i]
                else:
                    ratio = 0.0
                css_albedo_ratios.append(ratio)
                
                # CSS
                analysis_data['css_stats'][i]['sun_angles'].append(sun_angle)
                analysis_data['css_stats'][i]['nadir_angles'].append(nadir_angle)
                analysis_data['css_stats'][i]['albedo_values'].append(alb_row[i])
                analysis_data['css_stats'][i]['albedo_ratios'].append(ratio)
            
            # 
            step_data = {
                'step': k,
                'altitude': altitude,
                'phase_angle': phase_angle,
                'total_illum': sum(ill_row),
                'total_albedo': sum(alb_row),
                'css_sun_angles': css_sun_angles,
                'css_nadir_angles': css_nadir_angles,
                'css_albedo_values': alb_row,
                'css_albedo_ratios': css_albedo_ratios,
                'is_daylight': 1 if dot(sN, rhatN) > 0 else 0
            }
            analysis_data['time_steps'].append(step_data)
        
        print(f"\n[3/6] Computing statistics...")
        
        # 
        stats_report = []
        stats_report.append("="*60)
        stats_report.append("CSS ALBEDO FACTORS ANALYSIS REPORT")
        stats_report.append("="*60)
        
        # 1. Albedo Ratio
        all_ratios = []
        for step in analysis_data['time_steps']:
            all_ratios.extend([r for r in step['css_albedo_ratios'] if r > 0])
        
        if all_ratios:
            stats_report.append("\n1. ALBEDO RATIO STATISTICS (Albedo/Illumination)")
            stats_report.append("-"*40)
            stats_report.append(f"Mean ratio: {np.mean(all_ratios):.4f}")
            stats_report.append(f"Median ratio: {np.median(all_ratios):.4f}")
            stats_report.append(f"Max ratio: {np.max(all_ratios):.4f}")
            stats_report.append(f"95th percentile: {np.percentile(all_ratios, 95):.4f}")
            stats_report.append(f"Standard deviation: {np.std(all_ratios):.4f}")
        
        # 2. CSS
        stats_report.append("\n2. PER-CSS STATISTICS")
        stats_report.append("-"*40)
        
        css_summary = []
        for i in range(ncss):
            css_data = analysis_data['css_stats'][i]
            
            # 
            valid_albedo = [a for a in css_data['albedo_values'] if a > 0]
            valid_ratios = [r for r in css_data['albedo_ratios'] if r > 0]
            valid_sun = [s for s, a in zip(css_data['sun_angles'], css_data['albedo_values']) 
                        if a > 0 and not math.isnan(s)]
            valid_nadir = [n for n, a in zip(css_data['nadir_angles'], css_data['albedo_values']) 
                          if a > 0 and not math.isnan(n)]
            
            if valid_albedo:
                mean_albedo = np.mean(valid_albedo)
                mean_ratio = np.mean(valid_ratios) if valid_ratios else 0
                mean_sun = np.mean(valid_sun) if valid_sun else 0
                mean_nadir = np.mean(valid_nadir) if valid_nadir else 0
                
                css_summary.append({
                    'css_index': i,
                    'axis': axesB[i],
                    'mean_albedo': mean_albedo,
                    'mean_ratio': mean_ratio,
                    'mean_sun_angle': mean_sun,
                    'mean_nadir_angle': mean_nadir,
                    'max_ratio': max(valid_ratios) if valid_ratios else 0
                })
                
                stats_report.append(f"CSS{i}: mean_albedo={mean_albedo:.5f}, "
                                  f"mean_ratio={mean_ratio:.4f}, "
                                  f"avg_sun_angle={mean_sun:.1f}°, "
                                  f"avg_nadir_angle={mean_nadir:.1f}°")
        
        # 3. 
        stats_report.append("\n3. CORRELATION ANALYSIS")
        stats_report.append("-"*40)
        
        # 
        corr_sun_angles = []
        corr_nadir_angles = []
        corr_phase_angles = []
        corr_altitudes = []
        corr_albedo_values = []
        corr_albedo_ratios = []
        
        for step in analysis_data['time_steps']:
            if step['total_albedo'] > 0:
                avg_sun = np.nanmean(step['css_sun_angles'])
                avg_nadir = np.nanmean(step['css_nadir_angles'])
                avg_ratio = np.nanmean([r for r in step['css_albedo_ratios'] if r > 0])
                
                if not math.isnan(avg_sun) and not math.isnan(avg_nadir) and not math.isnan(avg_ratio):
                    corr_sun_angles.append(avg_sun)
                    corr_nadir_angles.append(avg_nadir)
                    corr_phase_angles.append(step['phase_angle'])
                    corr_altitudes.append(step['altitude'])
                    corr_albedo_values.append(step['total_albedo'])
                    corr_albedo_ratios.append(avg_ratio)
        
        if len(corr_albedo_values) > 10:
            # 
            factors = {
                'Sun Angle': corr_sun_angles,
                'Nadir Angle': corr_nadir_angles,
                'Phase Angle': corr_phase_angles,
                'Altitude': corr_altitudes
            }
            
            for name, values in factors.items():
                if len(values) > 2:
                    corr_val, p_val = scipy_stats.pearsonr(values, corr_albedo_values)
                    corr_ratio, p_ratio = scipy_stats.pearsonr(values, corr_albedo_ratios)
                    
                    stats_report.append(f"{name}:")
                    stats_report.append(f"  vs Total Albedo: r={corr_val:.4f}, p={p_val:.4e}")
                    stats_report.append(f"  vs Albedo Ratio: r={corr_ratio:.4f}, p={p_ratio:.4e}")
        
        # 4. 
        stats_report.append("\n4. KEY FINDINGS")
        stats_report.append("-"*40)
        
        # CSS
        max_ratio_css = max(css_summary, key=lambda x: x['mean_ratio'])
        min_ratio_css = min(css_summary, key=lambda x: x['mean_ratio'])
        
        stats_report.append(f"Most affected CSS: CSS{max_ratio_css['css_index']} "
                          f"(avg ratio={max_ratio_css['mean_ratio']:.4f})")
        stats_report.append(f"Least affected CSS: CSS{min_ratio_css['css_index']} "
                          f"(avg ratio={min_ratio_css['mean_ratio']:.4f})")
        
        # Phase angle
        if corr_phase_angles:
            phase_bins = np.linspace(0, 180, 7)
            phase_groups = np.digitize(corr_phase_angles, phase_bins)
            
            stats_report.append("\nAlbedo by Phase Angle ranges:")
            for i in range(1, len(phase_bins)):
                mask = phase_groups == i
                if np.sum(mask) > 0:
                    mean_alb = np.mean(np.array(corr_albedo_values)[mask])
                    stats_report.append(f"  {phase_bins[i-1]:.0f}°-{phase_bins[i]:.0f}°: "
                                      f"mean albedo={mean_alb:.5f} (n={np.sum(mask)})")
        
        # 
        report_file = dest / "albedo_factors_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(stats_report))
        
        print(f"\n[4/6] Saving detailed CSV files...")
        
        # CSSCSV
        css_stats_file = dest / "css_multifactor_statistics.csv"
        with open(css_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['css_index', 'axis_x', 'axis_y', 'axis_z', 
                           'mean_albedo', 'mean_ratio', 'max_ratio',
                           'mean_sun_angle', 'mean_nadir_angle'])
            for css in css_summary:
                writer.writerow([
                    css['css_index'],
                    css['axis'][0], css['axis'][1], css['axis'][2],
                    css['mean_albedo'], css['mean_ratio'], css['max_ratio'],
                    css['mean_sun_angle'], css['mean_nadir_angle']
                ])
        
        # CSV
        time_series_file = dest / "albedo_time_series_analysis.csv"
        sample_interval = max(1, nsteps // 1000)  # 1000
        with open(time_series_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['step', 'altitude', 'phase_angle', 'total_illum', 'total_albedo', 
                     'avg_albedo_ratio', 'is_daylight']
            writer.writerow(header)
            
            for k in range(0, nsteps, sample_interval):
                step = analysis_data['time_steps'][k]
                avg_ratio = np.nanmean([r for r in step['css_albedo_ratios'] if r > 0])
                writer.writerow([
                    step['step'], step['altitude'], step['phase_angle'],
                    step['total_illum'], step['total_albedo'],
                    avg_ratio if not math.isnan(avg_ratio) else 0,
                    step['is_daylight']
                ])
        
        print(f"\n[5/6] Creating visualizations...")
        viz_path = create_multifactor_visualizations(analysis_data, dest, ncss)
        
        print(f"\n[6/6] Analysis complete!")
        print(f"\nOutput files saved to: {dest}")
        print(f"  - {report_file.name}")
        print(f"  - {css_stats_file.name}")
        print(f"  - {time_series_file.name}")
        print(f"  - {viz_path.name}")
        
        # 
        print(f"\n" + "="*60)
        print("SUMMARY OF KEY FINDINGS")
        print("="*60)
        if all_ratios:
            print(f"Average Albedo Ratio: {np.mean(all_ratios):.4f}")
            print(f"Maximum Albedo Ratio: {np.max(all_ratios):.4f}")
        print(f"Most affected CSS: CSS{max_ratio_css['css_index']}")
        print(f"Least affected CSS: CSS{min_ratio_css['css_index']}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()