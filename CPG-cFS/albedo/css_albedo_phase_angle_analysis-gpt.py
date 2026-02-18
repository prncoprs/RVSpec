#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS-Specific Phase Angle Analysis (Geometry-correct)
- Per-CSS ray-sphere intersection to find ground hit point
- Per-CSS phase angle alpha = angle(sun_N, ground->sat direction)
- Also compute SZA (solar zenith) and VZA (view zenith)
- Diagnostics: per-CSS albedo variance, inter-column correlation,
              hit_ground/lit_ground ratios
"""

import argparse, os, sys, math, re, csv, traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

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
    """Convert quaternion to rotation matrix CBN (Body<-Inertial N)"""
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

# ---------- physically-correct CSS-specific phase ----------
def ray_sphere_intersect(pos, dir_unit, R):
    """
     R ()
    pos:  (N, km)
    dir_unit:  (N)
     (hit:bool, t_near:float) G = pos + t*dir_unit
    """
    b = 2.0 * dot(pos, dir_unit)
    c = dot(pos, pos) - R*R
    disc = b*b - 4.0*c
    if disc <= 0.0:
        return False, None
    sqrtD = math.sqrt(disc)
    t1 = (-b - sqrtD) / 2.0
    t2 = (-b + sqrtD) / 2.0
    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return False, None
    return True, t



def calculate_css_specific_phase_angle(css_axis_N, sN, rN, earth_radius=6371.0, hfov_deg=90.0):
    """
    CSS-specific phase angle
    """
    # 
    css_axis_N = unit(css_axis_N)
    sN = unit(sN)
    
    # FOV
    rhatN = unit(rN)
    ang_to_nadir = angle_deg(css_axis_N, [-rhatN[0], -rhatN[1], -rhatN[2]])
    if ang_to_nadir > hfov_deg + 1e-9:
        return (math.nan, math.nan, math.nan)
    
    # 
    hit, t = ray_sphere_intersect(rN, css_axis_N, earth_radius)
    if not hit:
        return (math.nan, math.nan, math.nan)
    
    # G
    G = [rN[0] + t*css_axis_N[0],
         rN[1] + t*css_axis_N[1],
         rN[2] + t*css_axis_N[2]]
    n_hat = unit(G)  # 
    
    # SZAVZA
    cos_sza = max(-1.0, min(1.0, dot(n_hat, sN)))
    sza = math.degrees(math.acos(cos_sza))
    
    u = [rN[0] - G[0], rN[1] - G[1], rN[2] - G[2]]
    u_hat = unit(u)
    cos_vza = max(-1.0, min(1.0, dot(n_hat, u_hat)))
    vza = math.degrees(math.acos(cos_vza))
    
    # *** phase angle ***
    # Phase angleG
    sun_dir = sN  # 
    obs_dir = u_hat  # G
    
    cos_phase = max(-1.0, min(1.0, dot(sun_dir, obs_dir)))
    phase_angle = math.degrees(math.acos(cos_phase))
    
    # phase_angleszaalpha
    return (phase_angle, sza, vza)  # phase angle



# ---------- Phase angle model (optional fit) ----------
def cosine_model(phase_angle, A, B, C):
    phase_rad = np.radians(phase_angle)
    return A * np.cos(phase_rad/2.0)**2 + B * np.cos(phase_rad/2.0) + C

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
                    # normalize axis
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

# ---------- Visualization (reuse, works with new alphas) ----------
def create_phase_comparison_visualization(analysis_data, dest_dir, ncss):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    gpa_list = [s['global_phase_angle'] for s in analysis_data['time_steps']]
    total_albedo = [s['total_albedo'] for s in analysis_data['time_steps']]
    
    ax1 = fig.add_subplot(gs[0, :])
    gpa_valid = np.array(gpa_list, float)
    albedo_valid = np.array(total_albedo, float)
    mask = ~np.isnan(gpa_valid) & ~np.isnan(albedo_valid)
    gpa_valid = gpa_valid[mask]; albedo_valid = albedo_valid[mask]
    
    if len(gpa_valid) > 0:
        sc = ax1.scatter(gpa_valid, albedo_valid, c=albedo_valid, cmap='viridis', alpha=0.3, s=1)
        ax1.set_xlabel('Global Phase Angle (deg)')
        ax1.set_ylabel('Total Albedo')
        ax1.set_title('Global Phase Angle vs Total Albedo')
        ax1.grid(True, alpha=0.3); ax1.set_xlim([0, 180])
        if len(gpa_valid) > 10:
            try:
                popt, _ = curve_fit(cosine_model, gpa_valid, albedo_valid, p0=[0.5,0.1,0.01])
                xs = np.linspace(0, 180, 200)
                ys = cosine_model(xs, *popt)
                ax1.plot(xs, ys, 'r-', lw=2, label=f'Cosine fit')
                ax1.legend()
            except: pass

    for css_idx in range(min(ncss, 6)):
        ax = fig.add_subplot(gs[1 + css_idx // 3, css_idx % 3])
        css_alb, css_gpa, css_spa = [], [], []
        for step in analysis_data['time_steps']:
            v = step['css_albedo'][css_idx]
            if v > 0 and not math.isnan(step['css_specific_phase'][css_idx]):
                css_alb.append(v)
                css_gpa.append(step['global_phase_angle'])
                css_spa.append(step['css_specific_phase'][css_idx])
        if css_alb:
            ax.scatter(css_gpa, css_alb, alpha=0.3, s=1, c='blue', label='vs Global')
            ax.scatter(css_spa, css_alb, alpha=0.3, s=1, c='red',  label='vs Specific')
            ax.set_xlabel('Phase Angle (deg)')
            ax.set_ylabel('CSS Albedo')
            ax.set_title(f'CSS{css_idx} ({analysis_data["css_config"][css_idx]["axis_str"]})')
            ax.set_xlim([0, 180]); ax.grid(True, alpha=0.3)
            if css_idx == 0: ax.legend(loc='upper left', fontsize=8)
    plt.suptitle('Global vs CSS-Specific Phase Angle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = dest_dir / "phase_angle_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    return out

def create_correlation_heatmap(analysis_data, dest_dir, ncss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    corr_global, corr_specific, labels = [], [], []
    for i in range(ncss):
        css_alb, css_gpa, css_spa = [], [], []
        for step in analysis_data['time_steps']:
            v = step['css_albedo'][i]
            a = step['css_specific_phase'][i]
            if v > 0 and not math.isnan(a):
                css_alb.append(v); css_gpa.append(step['global_phase_angle']); css_spa.append(a)
        if len(css_alb) > 10:
            corr_g, _ = scipy_stats.pearsonr(css_gpa, css_alb)
            corr_s, _ = scipy_stats.pearsonr(css_spa, css_alb)
            corr_global.append(corr_g); corr_specific.append(corr_s); labels.append(f'CSS{i}')
    if corr_global:
        b1 = ax1.bar(labels, corr_global, color='blue', alpha=0.7)
        ax1.set_ylabel('Corr with Albedo'); ax1.set_title('Global Phase Correlation')
        ax1.set_ylim([-1, 1]); ax1.grid(True, alpha=0.3)
        for bar, val in zip(b1, corr_global):
            ax1.text(bar.get_x()+bar.get_width()/2., val, f'{val:.3f}', ha='center',
                     va='bottom' if val>=0 else 'top')
        b2 = ax2.bar(labels, corr_specific, color='red', alpha=0.7)
        ax2.set_ylabel('Corr with Albedo'); ax2.set_title('CSS-Specific Phase Correlation')
        ax2.set_ylim([-1, 1]); ax2.grid(True, alpha=0.3)
        for bar, val in zip(b2, corr_specific):
            ax2.text(bar.get_x()+bar.get_width()/2., val, f'{val:.3f}', ha='center',
                     va='bottom' if val>=0 else 'top')
    plt.suptitle('Correlation: Global vs CSS-Specific', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = dest_dir / "correlation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    return out

def create_phase_difference_analysis(analysis_data, dest_dir, ncss):
    fig = plt.figure(figsize=(16, 10))
    gs = matplotlib.gridspec.GridSpec(2, 3)
    for css_idx in range(min(ncss, 6)):
        ax = fig.add_subplot(gs[css_idx // 3, css_idx % 3])
        diffs, vals = [], []
        for step in analysis_data['time_steps']:
            a = step['css_specific_phase'][css_idx]
            if step['css_albedo'][css_idx] > 0 and not math.isnan(a):
                diffs.append(a - step['global_phase_angle'])
                vals.append(step['css_albedo'][css_idx])
        if diffs:
            ax.scatter(diffs, vals, c=vals, cmap='hot', alpha=0.5, s=1)
            ax.set_xlabel('Phase Difference (Specific - Global) [deg]')
            ax.set_ylabel('CSS Albedo')
            ax.set_title(f'CSS{css_idx} ({analysis_data["css_config"][css_idx]["axis_str"]})')
            ax.grid(True, alpha=0.3); ax.axvline(0, color='k', ls='--', alpha=0.3)
            ax.text(0.05, 0.95, f'Mean={np.mean(diffs):.1f}°\nStd={np.std(diffs):.1f}°',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.suptitle('Phase Angle Difference (Specific - Global)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = dest_dir / "phase_difference_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    return out

def pad_rows_to_ncss(rows, ncss):
    """ ncss /"""
    out = []
    for r in rows:
        if r is None:
            r = []
        #  float
        rr = []
        for x in r:
            try:
                rr.append(float(x))
            except Exception:
                rr.append(0.0)
        if len(rr) < ncss:
            rr = rr + [0.0]*(ncss - len(rr))
        else:
            rr = rr[:ncss]
        out.append(rr)
    return out

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="CSS-specific phase angle analysis (geometry-correct)")
        ap.add_argument("--local-dir", required=True, help="Path to directory containing NOS3InOut")
        ap.add_argument("--dest", required=True, help="Destination directory for analysis results")
        args = ap.parse_args()

        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)

        print("CSS-Specific Phase Angle Analysis (Geometry-correct)")
        print("="*60)
        print(f"Input : {local_dir}")
        print(f"Output: {dest}\n")

        # Locate NOS3InOut
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut not found in {local_dir}")

        # Load paths
        pos_path = local_nos3/"PosN.42"
        qbn_path = local_nos3/"qbn.42"
        svn_path = local_nos3/"svn.42"
        svb_path = local_nos3/"svb.42"
        alb_path = local_nos3/"Albedo.42"
        ill_path = local_nos3/"Illum.42"
        inp_sim  = local_nos3/"Inp_Sim.txt"

        for p in [pos_path,qbn_path,svn_path,svb_path,alb_path,ill_path,inp_sim]:
            if not p.exists():
                sys.exit(f"Missing file: {p}")

        # Detect CSS count
        ncss = count_cols_first_nonempty(alb_path)
        if ncss <= 0: ncss = count_cols_first_nonempty(ill_path)
        if ncss <= 0: sys.exit("Cannot detect number of CSS")
        print(f"[1/6] Files loaded. CSS count: {ncss}")

        # Parse CSS config
        sc_file_name = detect_sc_file_from_inp_sim(inp_sim)
        sc_path = (local_nos3/sc_file_name) if sc_file_name else (local_nos3/"SC_NOS3.txt")
        if not sc_path.exists():
            sc_path = local_nos3/"SC_SensorFOV.txt"
        axesB, fov_deg = [], []
        if sc_path.exists():
            axesB, fov_deg = parse_css_from_sc_file(sc_path)
        if not axesB or len(axesB)!=ncss:
            # Fallback: ±X ±Y ±Z repeating
            defaults = [
                [ 1, 0, 0], [-1, 0, 0],
                [ 0, 1, 0], [ 0,-1, 0],
                [ 0, 0, 1], [ 0, 0,-1],
            ]
            axesB = ([unit(a) for a in defaults] * ((ncss+5)//6))[:ncss]
            if not fov_deg or len(fov_deg)!=ncss:
                fov_deg = [90.0]*ncss

        # Load data
        posN = read_vector3_first3(pos_path)
        qbn  = read_numeric_cols(qbn_path, need=4, offset=0)
        svn  = read_numeric_cols(svn_path, need=3, offset=0)
        svb  = read_numeric_cols(svb_path, need=3, offset=0)
        alb  = read_numeric_cols(alb_path, need=0)
        ill  = read_numeric_cols(ill_path, need=0)
        nsteps = min(len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill))
        print(f"[2/6] Steps: {nsteps}")

        # Report CSS config
        axis_labels = ['+X','-X','+Y','-Y','+Z','-Z']
        css_config = []
        for i in range(ncss):
            css_config.append({
                'index': i,
                'axis': axesB[i],
                'axis_str': axis_labels[i] if i < 6 else f'Custom{i}',
                'fov': fov_deg[i]
            })
        print("[3/6] CSS Configuration:")
        for cfg in css_config:
            ax = cfg['axis']
            print(f"  CSS{cfg['index']}: {cfg['axis_str']}  axis=[{ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}], FOV={cfg['fov']} deg")

        # Main loop
        analysis_data = {'time_steps': [], 'css_config': css_config}
        hit_cnt = [0]*ncss
        lit_cnt = [0]*ncss

        for k in range(nsteps):
            rN  = posN[k]
            CBN = quat_to_cbn(qbn[k])
            CNB = mattranspose(CBN)
            sN  = unit(svn[k])

            # Global phase: angle between sun_N and rhat_N (full-disk phase)
            rhatN = unit(rN)
            global_phase_angle = angle_deg(sN, rhatN)

            ill_row = (ill[k] + [0.0]*ncss)[:ncss]
            alb_row = (alb[k] + [0.0]*ncss)[:ncss]

            css_alpha = []
            css_sza   = []
            css_vza   = []
            for i in range(ncss):
                css_axis_B = axesB[i]
                css_axis_N = unit(matvec(CNB, css_axis_B))  
                alpha, sza, vza = calculate_css_specific_phase_angle(
                    css_axis_N, sN, rN, earth_radius=6371.0, hfov_deg=fov_deg[i]
                )
                css_alpha.append(alpha); css_sza.append(sza); css_vza.append(vza)
                if not math.isnan(alpha):  # 
                    hit_cnt[i] += 1
                    if sza < 90.0:         # 
                        lit_cnt[i] += 1


            analysis_data['time_steps'].append({
                'step': k,
                'global_phase_angle': global_phase_angle,
                'total_albedo': float(sum(alb_row)),
                'css_albedo': alb_row,
                'css_specific_phase': css_alpha,
                'css_sza': css_sza,
                'css_vza': css_vza
            })

        # Diagnostics: albedo variance & inter-column corr
        print("[4/6] Diagnostics")
        
        #  Albedo 
        alb_rows = pad_rows_to_ncss(alb, ncss)[:nsteps]
        A = np.array(alb_rows, dtype=float)

        # 
        with np.errstate(invalid='ignore'):
            col_std = np.nanstd(A, axis=0)
        print("  Albedo per-CSS std: ", " ".join(f"{s:.3e}" for s in col_std))

        #  NaN
        def safe_corr(x, y):
            if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
                return float('nan')
            r = np.corrcoef(x, y)[0, 1]
            return float(r)

        if ncss >= 2:
            for i in range(ncss):
                for j in range(i+1, ncss):
                    r = safe_corr(A[:, i], A[:, j])
                    print(f"  corr(Alb{i},Alb{j}) = {r:.4f}" if not np.isnan(r) else
                        f"  corr(Alb{i},Alb{j}) = NaN")

        print("  CSS hit_ground ratio:", [f"{c/nsteps:.2%}" for c in hit_cnt])
        print("  CSS  lit_ground ratio:", [f"{c/nsteps:.2%}" for c in lit_cnt])

        # Statistics report
        print("[5/6] Statistics & report")
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("CSS-SPECIFIC PHASE ANGLE ANALYSIS REPORT (Geometry-correct)")
        report_lines.append("="*60)

        gpa_values = np.array([s['global_phase_angle'] for s in analysis_data['time_steps']], float)
        total_albedo = np.array([s['total_albedo'] for s in analysis_data['time_steps']], float)
        mask = ~np.isnan(gpa_values) & ~np.isnan(total_albedo)
        if mask.sum() > 10:
            corr_global, _ = scipy_stats.pearsonr(gpa_values[mask], total_albedo[mask])
            # fit (optional)
            try:
                popt, _ = curve_fit(cosine_model, gpa_values[mask], total_albedo[mask], p0=[0.5,0.1,0.01])
                resid = total_albedo[mask] - cosine_model(gpa_values[mask], *popt)
                r2 = 1 - (resid@resid) / np.sum((total_albedo[mask]-total_albedo[mask].mean())**2)
            except:
                r2 = float('nan')
            report_lines.append("\n1. GLOBAL PHASE ANGLE")
            report_lines.append("-"*40)
            report_lines.append(f"Range: {np.nanmin(gpa_values):.1f}° - {np.nanmax(gpa_values):.1f}°")
            report_lines.append(f"Mean : {np.nanmean(gpa_values):.1f}°")
            report_lines.append(f"Correlation with total albedo: r={corr_global:.4f}")
            report_lines.append(f"Cosine-like model R²: {r2:.4f}")

        # Per-CSS correlations
        report_lines.append("\n2. CSS-SPECIFIC ANALYSIS")
        report_lines.append("-"*40)
        report_lines.append("CSS | Global r | Specific r | Better | Mean Albedo | Hit% | Lit%")
        report_lines.append("----|----------|------------|--------|-------------|------|-----")

        css_stats = []
        for i in range(ncss):
            # sample where alpha is valid (hit ground)
            css_alb, css_gpa, css_spa = [], [], []
            for step in analysis_data['time_steps']:
                a = step['css_specific_phase'][i]
                if not math.isnan(a) and step['css_albedo'][i] > 0:
                    css_alb.append(step['css_albedo'][i])
                    css_gpa.append(step['global_phase_angle'])
                    css_spa.append(a)
            if len(css_alb) > 10:
                rg, _ = scipy_stats.pearsonr(css_gpa, css_alb)
                rs, _ = scipy_stats.pearsonr(css_spa, css_alb)
                better = "Specific" if abs(rs) > abs(rg) else "Global"
                mean_alb = float(np.mean(css_alb))
            else:
                rg, rs, better, mean_alb = float('nan'), float('nan'), "N/A", float('nan')

            css_stats.append({
                'css': i,
                'axis': css_config[i]['axis_str'],
                'corr_global': rg,
                'corr_specific': rs,
                'better': better,
                'mean_albedo': mean_alb,
                'hit_ratio': hit_cnt[i]/nsteps,
                'lit_ratio': lit_cnt[i]/nsteps
            })
            report_lines.append(f"{i:3d} | {rg:8.3f} | {rs:10.3f} | {better:6s} | {mean_alb:11.5f} | {hit_cnt[i]/nsteps:5.1%} | {lit_cnt[i]/nsteps:5.1%}")

        # Summary
        report_lines.append("\n3. SUMMARY")
        report_lines.append("-"*40)
        valid_css = [c for c in css_stats if not (math.isnan(c['corr_global']) or math.isnan(c['corr_specific']))]
        if valid_css:
            n_better_specific = sum(1 for s in valid_css if s['better']=="Specific")
            n_better_global   = len(valid_css) - n_better_specific
            mean_impr = float(np.mean([abs(s['corr_specific'])-abs(s['corr_global']) for s in valid_css]))
            report_lines.append(f"CSS where specific phase is better: {n_better_specific}/{len(valid_css)}")
            report_lines.append(f"CSS where global phase is better:   {n_better_global}/{len(valid_css)}")
            report_lines.append(f"Average |r_specific|-|r_global|:   {mean_impr:.4f}")
            report_lines.append("Overall: " + ("CSS-specific better" if mean_impr>0 else "Global phase sufficient"))

        report_file = dest/"css_specific_phase_analysis_report.txt"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"  Report written: {report_file}")

        # CSV summary per CSS
        csv_file = dest/"css_phase_comparison.csv"
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["css_index","axis","corr_global","corr_specific","better","mean_albedo","hit_ratio","lit_ratio"])
            for s in css_stats:
                w.writerow([s['css'], s['axis'], s['corr_global'], s['corr_specific'],
                            s['better'], s['mean_albedo'], s['hit_ratio'], s['lit_ratio']])
        print(f"  CSV written:    {csv_file}")

        # Visualizations
        print("[6/6] Visualizations")
        v1 = create_phase_comparison_visualization(analysis_data, dest, ncss)
        v2 = create_correlation_heatmap(analysis_data, dest, ncss)
        v3 = create_phase_difference_analysis(analysis_data, dest, ncss)
        print(f"  {v1.name}")
        print(f"  {v2.name}")
        print(f"  {v3.name}")

        print("\nDone.")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
