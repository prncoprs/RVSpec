#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSS-Specific Phase Angle Analysis
Computes both global and CSS-specific phase angles considering spacecraft attitude changes
"""

import argparse, os, sys, math, re, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
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

def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def vec_add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def vec_sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

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
def calculate_css_specific_phase_angle(css_axis_N, sun_N, sat_pos_N, earth_radius=6371.0):
    """
    Calculate the phase angle specific to what a CSS can see
    
    This computes the illumination angle of the Earth surface that the CSS is looking at,
    taking into account the CSS orientation and the Sun-Earth-Satellite geometry.
    
    Args:
        css_axis_N: CSS boresight direction in inertial frame (unit vector)
        sun_N: Sun direction from Earth center (unit vector)
        sat_pos_N: Satellite position from Earth center (km)
        earth_radius: Earth radius in km
    
    Returns:
        CSS-specific phase angle in degrees
    """
    
    # Satellite position unit vector
    rhat_N = unit(sat_pos_N)
    sat_altitude = norm(sat_pos_N)
    
    # Check if CSS can see Earth
    # Angle between CSS boresight and nadir direction
    nadir_N = vec_scale(rhat_N, -1)  # Points from satellite to Earth
    css_nadir_angle = angle_deg(css_axis_N, nadir_N)
    
    # Maximum angle at which CSS can see Earth horizon
    horizon_angle = math.degrees(math.asin(earth_radius / sat_altitude))
    
    if css_nadir_angle > horizon_angle + 60:  # Add FOV consideration
        # CSS cannot see Earth
        return 180.0
    
    # Find intersection of CSS boresight with Earth surface (simplified)
    # This is a simplified model - assumes CSS looks at the point on Earth
    # in the direction of its boresight projected onto Earth surface
    
    if css_nadir_angle < horizon_angle:
        # CSS boresight intersects Earth
        # Calculate the point on Earth surface that CSS is looking at
        
        # Simplified approach: use the CSS boresight direction to determine
        # which part of Earth it sees
        
        # The effective sun angle at the point CSS is looking at
        # This is approximated by considering how the CSS boresight relates
        # to the sun-earth-satellite geometry
        
        # Project CSS boresight onto Earth surface direction
        css_earth_component = dot(css_axis_N, nadir_N)
        
        if css_earth_component > 0:
            # CSS is looking toward Earth
            
            # The phase angle the CSS sees depends on:
            # 1. The global phase angle
            # 2. The CSS orientation relative to the sun-satellite line
            
            # Global phase angle
            global_phase = angle_deg(sun_N, rhat_N)
            
            # CSS deviation from the satellite-sun line
            sat_sun_N = vec_sub(sun_N, rhat_N)
            sat_sun_N = unit(sat_sun_N)
            
            css_sun_alignment = dot(css_axis_N, sat_sun_N)
            
            # If CSS is aligned with sun direction, it sees less illuminated Earth
            # If CSS is opposite to sun direction, it sees more illuminated Earth
            
            # Effective phase angle combines global phase with CSS orientation
            # This is a simplified model that captures the main effect
            
            alignment_factor = (1 - css_sun_alignment) / 2  # Range [0, 1]
            
            # CSS-specific phase angle
            css_phase = global_phase * (0.5 + alignment_factor)
            
            # Ensure within valid range
            css_phase = max(0, min(180, css_phase))
            
        else:
            # CSS is looking away from Earth
            css_phase = 180.0
    else:
        # CSS is looking at Earth limb/horizon
        # Use global phase angle with reduced visibility
        global_phase = angle_deg(sun_N, rhat_N)
        visibility_factor = 1 - (css_nadir_angle - horizon_angle) / 60
        css_phase = global_phase + (180 - global_phase) * (1 - visibility_factor)
    
    return css_phase


def calculate_css_specific_phase_angle_simple(css_axis_N, sun_N, sat_pos_N):
    """
    CSSphase angle
    """
    
    # Global phase angle
    rhat_N = unit(sat_pos_N)
    global_phase = angle_deg(sun_N, rhat_N)
    
    # CSS
    nadir_N = vec_scale(rhat_N, -1)
    css_earth_alignment = dot(css_axis_N, nadir_N)
    
    if css_earth_alignment <= 0:
        # CSS
        return 180.0
    
    # CSS
    css_sun_alignment = dot(css_axis_N, sun_N)
    
    # 
    # CSSphase angle
    # CSSphase angle
    
    # sun alignment[-1,1][0.5,1.5]
    modulation = 1.0 - css_sun_alignment * 0.3
    
    css_phase = global_phase * modulation
    
    return max(0, min(180, css_phase))


def calculate_css_specific_phase_angle_accurate(css_axis_N, sun_N, sat_pos_N, earth_radius=6371.0):
    """
    CSS-specific phase angle
    -phase angle
    """
    
    # 1. -CSS
    # CSS
    # CSScss_axis_N
    
    # : |sat_pos_N + t*css_axis_N|^2 = earth_radius^2
    a = dot(css_axis_N, css_axis_N)  # = 1 for unit vector
    b = 2.0 * dot(sat_pos_N, css_axis_N)
    c = dot(sat_pos_N, sat_pos_N) - earth_radius * earth_radius
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        # CSS
        return 180.0  #  nan
    
    # 
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    # tCSS
    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return 180.0  # CSS
    
    # 2. GCSS
    G = [sat_pos_N[0] + t*css_axis_N[0],
         sat_pos_N[1] + t*css_axis_N[1],
         sat_pos_N[2] + t*css_axis_N[2]]
    
    # 3. 
    G_unit = unit(G)  # G
    sun_elevation = 90 - angle_deg(G_unit, sun_N)  # 
    
    if sun_elevation < 0:
        # 
        return 180.0
    
    # 4. CSS-specific phase angle
    # Phase angle = G
    
    # 
    #  -> G ->  
    
    # G
    sun_from_G = sun_N
    
    # G
    G_to_sat = [sat_pos_N[0] - G[0],
                sat_pos_N[1] - G[1],
                sat_pos_N[2] - G[2]]
    sat_from_G = unit(G_to_sat)
    
    # CSS-specific phase angle
    cos_phase = dot(sun_from_G, sat_from_G)
    cos_phase = max(-1.0, min(1.0, cos_phase))
    css_phase_angle = math.degrees(math.acos(cos_phase))
    
    # 5. BRDF
    # 
    view_zenith = angle_deg(G_unit, sat_from_G)
    
    # view_zenithphase angle
    # albedo
    
    return css_phase_angle

def calculate_css_specific_phase_angle_improved(
    css_axis_N, sun_N, sat_pos_N, earth_radius=6371.0):
    """
    CSSphase angle
    
    """
    
    # 
    sat_dist = norm(sat_pos_N)
    rhat_N = unit(sat_pos_N)
    
    # CSS
    css_dir = unit(css_axis_N)
    
    # CSS
    # -
    # Ray: P = sat_pos + t * css_dir
    # Sphere: |P| = earth_radius
    
    # 
    a = dot(css_dir, css_dir)  # = 1 for unit vector
    b = 2 * dot(sat_pos_N, css_dir)
    c = dot(sat_pos_N, sat_pos_N) - earth_radius * earth_radius
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        # 
        return 180.0
    
    # 
    t = (-b - math.sqrt(discriminant)) / (2*a)
    
    if t < 0:
        # CSS
        return 180.0
    
    # 
    intersection = vec_add(sat_pos_N, vec_scale(css_dir, t))
    intersection_unit = unit(intersection)
    
    # 
    # 
    point_sun_angle = angle_deg(sun_N, intersection_unit)
    
    # CSSphase angle
    # 
    if point_sun_angle > 90:
        # 
        effective_phase = 180.0
    else:
        # phase angle
        effective_phase = point_sun_angle
    
    return effective_phase

# ---------- Phase angle models ----------
def cosine_model(phase_angle, A, B, C):
    """Cosine model for albedo vs phase angle"""
    phase_rad = np.radians(phase_angle)
    return A * np.cos(phase_rad/2)**2 + B * np.cos(phase_rad/2) + C

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
def create_phase_comparison_visualization(analysis_data, dest_dir, ncss):
    """Create visualization comparing global vs CSS-specific phase angles"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Extract global data
    gpa_list = [s['global_phase_angle'] for s in analysis_data['time_steps']]
    total_albedo = [s['total_albedo'] for s in analysis_data['time_steps']]
    
    # 1. Global Phase Angle vs Total Albedo
    ax1 = fig.add_subplot(gs[0, :])
    
    valid_mask = ~(np.isnan(gpa_list) | np.isnan(total_albedo))
    gpa_valid = np.array(gpa_list)[valid_mask]
    albedo_valid = np.array(total_albedo)[valid_mask]
    
    if len(gpa_valid) > 0:
        scatter = ax1.scatter(gpa_valid, albedo_valid, c=albedo_valid, 
                            cmap='viridis', alpha=0.3, s=1)
        ax1.set_xlabel('Global Phase Angle (degrees)')
        ax1.set_ylabel('Total Albedo')
        ax1.set_title('Global Phase Angle vs Total Albedo')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 180])
        
        # Add fitted model
        if len(gpa_valid) > 10:
            try:
                popt, _ = curve_fit(cosine_model, gpa_valid, albedo_valid, 
                                  p0=[0.5, 0.1, 0.01])
                x_model = np.linspace(0, 180, 200)
                y_model = cosine_model(x_model, *popt)
                ax1.plot(x_model, y_model, 'r-', linewidth=2, 
                        label=f'Fit: R²={analysis_data.get("global_r2", 0):.3f}')
                ax1.legend()
            except:
                pass
    
    # 2-7. Individual CSS comparisons (Global vs Specific)
    for css_idx in range(min(ncss, 6)):
        ax = fig.add_subplot(gs[1 + css_idx // 3, css_idx % 3])
        
        # Collect data for this CSS
        css_albedo = []
        css_gpa = []
        css_spa = []  # CSS-specific phase angle
        
        for step in analysis_data['time_steps']:
            if step['css_albedo'][css_idx] > 0:
                css_albedo.append(step['css_albedo'][css_idx])
                css_gpa.append(step['global_phase_angle'])
                css_spa.append(step['css_specific_phase'][css_idx])
        
        if len(css_albedo) > 0:
            # Plot both relationships
            ax.scatter(css_gpa, css_albedo, alpha=0.3, s=1, c='blue', label='vs Global')
            ax.scatter(css_spa, css_albedo, alpha=0.3, s=1, c='red', label='vs Specific')
            
            ax.set_xlabel('Phase Angle (degrees)')
            ax.set_ylabel('CSS Albedo')
            ax.set_title(f'CSS{css_idx} ({analysis_data["css_config"][css_idx]["axis_str"]})')
            ax.set_xlim([0, 180])
            ax.set_ylim([0, max(css_albedo)*1.1])
            ax.grid(True, alpha=0.3)
            
            # Calculate and display correlations
            if len(css_gpa) > 10:
                corr_gpa, _ = scipy_stats.pearsonr(css_gpa, css_albedo)
                corr_spa, _ = scipy_stats.pearsonr(css_spa, css_albedo)
                
                text = f'r(global)={corr_gpa:.3f}\nr(specific)={corr_spa:.3f}\nn={len(css_gpa)}'
                
                # Highlight if specific is better
                if abs(corr_spa) > abs(corr_gpa):
                    text += '\n Specific better'
                    bbox_color = 'lightgreen'
                else:
                    bbox_color = 'wheat'
                
                ax.text(0.95, 0.95, text, 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7))
            
            if css_idx == 0:
                ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Global vs CSS-Specific Phase Angle Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "phase_angle_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_correlation_heatmap(analysis_data, dest_dir, ncss):
    """Create heatmap showing correlations for each CSS"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare correlation matrices
    corr_global = []
    corr_specific = []
    css_labels = []
    
    for i in range(ncss):
        css_albedo = []
        css_gpa = []
        css_spa = []
        
        for step in analysis_data['time_steps']:
            if step['css_albedo'][i] > 0:
                css_albedo.append(step['css_albedo'][i])
                css_gpa.append(step['global_phase_angle'])
                css_spa.append(step['css_specific_phase'][i])
        
        if len(css_albedo) > 10:
            corr_g, _ = scipy_stats.pearsonr(css_gpa, css_albedo)
            corr_s, _ = scipy_stats.pearsonr(css_spa, css_albedo)
            corr_global.append(corr_g)
            corr_specific.append(corr_s)
            css_labels.append(f'CSS{i}')
    
    if corr_global:
        # Plot global correlations
        bars1 = ax1.bar(css_labels, corr_global, color='blue', alpha=0.7)
        ax1.set_ylabel('Correlation with Albedo')
        ax1.set_title('Global Phase Angle Correlations')
        ax1.set_ylim([-1, 0])
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Add values on bars
        for bar, val in zip(bars1, corr_global):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='top' if val < 0 else 'bottom')
        
        # Plot specific correlations
        bars2 = ax2.bar(css_labels, corr_specific, color='red', alpha=0.7)
        ax2.set_ylabel('Correlation with Albedo')
        ax2.set_title('CSS-Specific Phase Angle Correlations')
        ax2.set_ylim([-1, 0])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Add values on bars
        for bar, val in zip(bars2, corr_specific):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='top' if val < 0 else 'bottom')
    
    plt.suptitle('Correlation Comparison: Global vs CSS-Specific Phase Angles', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "correlation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_phase_difference_analysis(analysis_data, dest_dir, ncss):
    """Analyze the difference between global and CSS-specific phase angles"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3)
    
    for css_idx in range(min(ncss, 6)):
        ax = fig.add_subplot(gs[css_idx // 3, css_idx % 3])
        
        # Calculate phase differences
        phase_diffs = []
        albedo_vals = []
        
        for step in analysis_data['time_steps']:
            if step['css_albedo'][css_idx] > 0:
                diff = step['css_specific_phase'][css_idx] - step['global_phase_angle']
                phase_diffs.append(diff)
                albedo_vals.append(step['css_albedo'][css_idx])
        
        if phase_diffs:
            # Plot phase difference vs albedo
            scatter = ax.scatter(phase_diffs, albedo_vals, 
                               c=albedo_vals, cmap='hot', alpha=0.5, s=1)
            
            ax.set_xlabel('Phase Difference (Specific - Global) [deg]')
            ax.set_ylabel('CSS Albedo')
            ax.set_title(f'CSS{css_idx} ({analysis_data["css_config"][css_idx]["axis_str"]})')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            
            # Add statistics
            mean_diff = np.mean(phase_diffs)
            std_diff = np.std(phase_diffs)
            ax.text(0.05, 0.95, 
                   f'Mean diff: {mean_diff:.1f}°\nStd: {std_diff:.1f}°',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('Phase Angle Difference Analysis (CSS-Specific minus Global)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = dest_dir / "phase_difference_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================== MAIN ==============================
def main():
    try:
        ap = argparse.ArgumentParser(description="CSS-specific phase angle analysis")
        ap.add_argument("--local-dir", required=True, 
                       help="Path to directory containing NOS3InOut folder")
        ap.add_argument("--dest", required=True,
                       help="Destination directory for analysis results")
        args = ap.parse_args()

        # Setup paths
        local_dir = Path(args.local_dir).expanduser().resolve()
        dest = Path(args.dest).expanduser().resolve()
        
        print(f"CSS-Specific Phase Angle Analysis")
        print(f"="*50)
        print(f"Input: {local_dir}")
        print(f"Output: {dest}")
        
        dest.mkdir(parents=True, exist_ok=True)
        
        # Find NOS3InOut folder
        local_nos3 = local_dir / "NOS3InOut"
        if not local_nos3.exists():
            sys.exit(f"Error: NOS3InOut folder not found in: {local_dir}")
        
        print(f"\n[1/6] Loading data files...")
        
        # Load all required files
        pos_path = local_nos3/"PosN.42"
        qbn_path = local_nos3/"qbn.42"
        svn_path = local_nos3/"svn.42"
        svb_path = local_nos3/"svb.42"
        alb_path = local_nos3/"Albedo.42"
        ill_path = local_nos3/"Illum.42"
        inp_sim = local_nos3/"Inp_Sim.txt"
        
        # Check file existence
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
            # Default configuration
            default_axes = [
                [ 1, 0, 0], [-1, 0, 0],  # ±X
                [ 0, 1, 0], [ 0,-1, 0],  # ±Y
                [ 0, 0, 1], [ 0, 0,-1],  # ±Z
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
        
        # Determine number of steps
        nsteps = min(len(posN),len(qbn),len(svn),len(svb),len(alb),len(ill))
        print(f"    Total steps: {nsteps}")
        
        # Store CSS configuration
        css_config = []
        axis_labels = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        for i in range(ncss):
            css_config.append({
                'index': i,
                'axis': axesB[i],
                'axis_str': axis_labels[i] if i < 6 else f'Custom{i}',
                'fov': fov_deg[i]
            })
        
        print(f"\n[2/6] CSS Configuration:")
        for cfg in css_config:
            print(f"    CSS{cfg['index']}: {cfg['axis_str']} "
                  f"[{cfg['axis'][0]:.2f}, {cfg['axis'][1]:.2f}, {cfg['axis'][2]:.2f}]")
        
        # Initialize analysis data
        analysis_data = {
            'time_steps': [],
            'css_config': css_config
        }
        
        print(f"\n[3/6] Computing global and CSS-specific phase angles...")
        
        # Main analysis loop
        for k in range(nsteps):
            # Basic vectors
            rN = posN[k]
            rhatN = unit(rN)
            
            # Coordinate transformations
            CBN = quat_to_cbn(qbn[k])
            CNB = mattranspose(CBN)
            
            # Sun vector
            sN = unit(svn[k])
            
            # Global Phase Angle
            global_phase_angle = angle_deg(sN, rhatN)
            
            # CSS measurements
            ill_row = ill[k][:ncss] if len(ill[k]) >= ncss else ill[k] + [0.0]*(ncss-len(ill[k]))
            alb_row = alb[k][:ncss] if len(alb[k]) >= ncss else alb[k] + [0.0]*(ncss-len(alb[k]))
            
            # Calculate CSS-specific phase angles
            css_specific_phase = []
            
            for i in range(ncss):
                # CSS axis in Body frame
                css_axis_B = axesB[i]
                
                # Transform to Inertial frame (changes with spacecraft attitude!)
                css_axis_N = matvec(CNB, css_axis_B)
                
                # Calculate CSS-specific phase angle
                # css_phase = calculate_css_specific_phase_angle(
                #     css_axis_N, sN, rN, earth_radius=6371.0
                # )
                css_phase = calculate_css_specific_phase_angle_simple(
                    css_axis_N, sN, rN
                )
                css_specific_phase.append(css_phase)
            
            # Store time step data
            step_data = {
                'step': k,
                'global_phase_angle': global_phase_angle,
                'total_albedo': sum(alb_row),
                'css_albedo': alb_row,
                'css_specific_phase': css_specific_phase
            }
            analysis_data['time_steps'].append(step_data)
        
        print(f"\n[4/6] Computing statistics...")
        
        # Statistical analysis
        stats_report = []
        stats_report.append("="*60)
        stats_report.append("CSS-SPECIFIC PHASE ANGLE ANALYSIS REPORT")
        stats_report.append("="*60)
        
        # Global phase statistics
        gpa_values = [s['global_phase_angle'] for s in analysis_data['time_steps']]
        total_albedo = [s['total_albedo'] for s in analysis_data['time_steps']]
        
        valid_mask = ~(np.isnan(gpa_values) | np.isnan(total_albedo))
        gpa_valid = np.array(gpa_values)[valid_mask]
        albedo_valid = np.array(total_albedo)[valid_mask]
        
        if len(gpa_valid) > 10:
            corr_global, p_global = scipy_stats.pearsonr(gpa_valid, albedo_valid)
            
            # Fit model
            try:
                popt, pcov = curve_fit(cosine_model, gpa_valid, albedo_valid, 
                                     p0=[0.5, 0.1, 0.01])
                residuals = albedo_valid - cosine_model(gpa_valid, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((albedo_valid - np.mean(albedo_valid))**2)
                r_squared = 1 - (ss_res / ss_tot)
                analysis_data['global_r2'] = r_squared
            except:
                r_squared = 0
            
            stats_report.append("\n1. GLOBAL PHASE ANGLE")
            stats_report.append("-"*40)
            stats_report.append(f"Range: {np.min(gpa_valid):.1f}° - {np.max(gpa_valid):.1f}°")
            stats_report.append(f"Mean: {np.mean(gpa_valid):.1f}°")
            stats_report.append(f"Correlation with total albedo: r={corr_global:.4f}")
            stats_report.append(f"Model R²: {r_squared:.4f}")
        
        # Per-CSS analysis
        stats_report.append("\n2. CSS-SPECIFIC ANALYSIS")
        stats_report.append("-"*40)
        stats_report.append("CSS | Global r | Specific r | Better | Mean Albedo")
        stats_report.append("----|----------|------------|--------|------------")
        
        css_stats = []
        for i in range(ncss):
            css_gpa = []
            css_spa = []
            css_alb = []
            
            for step in analysis_data['time_steps']:
                if step['css_albedo'][i] > 0:
                    css_gpa.append(step['global_phase_angle'])
                    css_spa.append(step['css_specific_phase'][i])
                    css_alb.append(step['css_albedo'][i])
            
            if len(css_alb) > 10:
                corr_gpa, _ = scipy_stats.pearsonr(css_gpa, css_alb)
                corr_spa, _ = scipy_stats.pearsonr(css_spa, css_alb)
                
                better = "Specific" if abs(corr_spa) > abs(corr_gpa) else "Global"
                
                css_stats.append({
                    'css': i,
                    'axis': css_config[i]['axis_str'],
                    'corr_global': corr_gpa,
                    'corr_specific': corr_spa,
                    'better': better,
                    'mean_albedo': np.mean(css_alb)
                })
                
                stats_report.append(
                    f"{i:3d} | {corr_gpa:8.3f} | {corr_spa:10.3f} | {better:6s} | {np.mean(css_alb):11.5f}"
                )
        
        # Summary
        stats_report.append("\n3. SUMMARY")
        stats_report.append("-"*40)
        
        if css_stats:
            n_better_specific = sum(1 for s in css_stats if s['better'] == "Specific")
            n_better_global = len(css_stats) - n_better_specific
            
            stats_report.append(f"CSS where specific phase is better: {n_better_specific}/{len(css_stats)}")
            stats_report.append(f"CSS where global phase is better: {n_better_global}/{len(css_stats)}")
            
            # Average improvement
            improvements = []
            for s in css_stats:
                improvement = abs(s['corr_specific']) - abs(s['corr_global'])
                improvements.append(improvement)
            
            mean_improvement = np.mean(improvements)
            stats_report.append(f"Average correlation change: {mean_improvement:.4f}")
            
            if mean_improvement > 0:
                stats_report.append("Overall: CSS-specific phase angles provide better predictions")
            else:
                stats_report.append("Overall: Global phase angle is sufficient")
        
        # Save report
        report_file = dest / "css_specific_phase_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(stats_report))
        
        # Save CSV
        print(f"\n[5/6] Saving CSV files...")
        
        csv_file = dest / "css_phase_comparison.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['css_index', 'axis', 'corr_global', 'corr_specific', 
                           'better', 'mean_albedo'])
            for s in css_stats:
                writer.writerow([s['css'], s['axis'], s['corr_global'], 
                               s['corr_specific'], s['better'], s['mean_albedo']])
        
        # Create visualizations
        print(f"\n[6/6] Creating visualizations...")
        
        viz1 = create_phase_comparison_visualization(analysis_data, dest, ncss)
        viz2 = create_correlation_heatmap(analysis_data, dest, ncss)
        viz3 = create_phase_difference_analysis(analysis_data, dest, ncss)
        
        print(f"\nAnalysis complete!")
        print(f"\nOutput files:")
        print(f"  - {report_file.name}")
        print(f"  - {csv_file.name}")
        print(f"  - {viz1.name}")
        print(f"  - {viz2.name}")
        print(f"  - {viz3.name}")
        
        # Print key findings
        print(f"\n" + "="*50)
        print("KEY FINDINGS")
        print("="*50)
        if css_stats:
            print(f"Global phase correlation: r={corr_global:.4f}")
            print(f"CSS-specific phase better for: {n_better_specific}/{len(css_stats)} CSS")
            if mean_improvement > 0.01:
                print(" CSS-specific phase angles improve predictions")
            elif mean_improvement < -0.01:
                print(" Global phase angle is sufficient")
            else:
                print(" Both methods perform similarly")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()