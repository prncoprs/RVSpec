#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Albedo Factor Analysis for cFS/NOS3 Simulations
Analyzes factors affecting CSS sun vector estimation error with albedo effects
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Set font type to avoid Type 3 fonts in PDF
matplotlib.rcParams.update({
    "pdf.fonttype": 42,   # Use TrueType (Type 42)
    "ps.fonttype": 42,
    "figure.max_open_warning": 50,
    "figure.dpi": 100,
})

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

# ============== CSS Functions ==============
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

# ============== Main Analysis Class ==============
class AlbedoFactorAnalyzer:
    def __init__(self, param_dir: str, data_dir: str, output_dir: str):
        self.param_dir = Path(param_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.css_axes = parse_css_axes()
        self.data_rows = []
        
    def find_valid_configs(self) -> List[str]:
        """Find configs that exist in both parameter and data directories"""
        param_configs = set()
        for p in self.param_dir.glob("config_*/metadata.json"):
            param_configs.add(p.parent.name)
        
        data_configs = set()
        for d in self.data_dir.glob("config_*/NOS3InOut"):
            if d.is_dir():
                data_configs.add(d.parent.name)
        
        valid = sorted(param_configs & data_configs)
        print(f"Found {len(valid)} valid configurations")
        return valid
    
    def load_config_data(self, config_id: str) -> Optional[Dict]:
        """Load all data for a configuration"""
        # Load metadata
        metadata_path = self.param_dir / config_id / "metadata.json"
        if not metadata_path.exists():
            print(f"Missing metadata for {config_id}")
            return None
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load 42 data files
        data_path = self.data_dir / config_id / "NOS3InOut"
        if not data_path.exists():
            print(f"Missing data directory for {config_id}")
            return None
        
        try:
            # Load all required files
            pos_n = read_numeric_cols(data_path / "PosN.42", need=3, offset=0)
            svn = read_numeric_cols(data_path / "svn.42", need=3, offset=0)
            svb = read_numeric_cols(data_path / "svb.42", need=3, offset=0)
            qbn = read_numeric_cols(data_path / "qbn.42", need=4, offset=0)
            albedo = read_numeric_cols(data_path / "Albedo.42", need=0, offset=0)
            illum = read_numeric_cols(data_path / "Illum.42", need=0, offset=0)
            
            # Find minimum common length
            min_len = min(len(pos_n), len(svn), len(svb), len(qbn), 
                         len(albedo), len(illum))
            
            if min_len == 0:
                print(f"No valid data for {config_id}")
                return None
            
            return {
                'metadata': metadata,
                'pos_n': pos_n[:min_len],
                'svn': svn[:min_len],
                'svb': svb[:min_len],
                'qbn': qbn[:min_len],
                'albedo': albedo[:min_len],
                'illum': illum[:min_len],
                'n_steps': min_len
            }
        
        except Exception as e:
            print(f"Error loading data for {config_id}: {e}")
            return None
    
    def calculate_dynamic_factors(self, data: Dict, timestep: int) -> Dict:
        """Calculate dynamic factors for a timestep"""
        factors = {}
        
        # Position and sun vectors
        pos_n = data['pos_n'][timestep]
        svn = data['svn'][timestep]
        svb = data['svb'][timestep]
        qbn = data['qbn'][timestep]
        
        # Global phase angle
        factors['global_phase_angle'] = angle_deg(svn, pos_n)
        
        # Altitude
        factors['altitude'] = math.sqrt(sum(x*x for x in pos_n)) - 6371.0
        
        # Daylight indicator
        factors['is_daylight'] = 1.0 if dot(unit(svn), unit(pos_n)) > 0 else 0.0
        
        # Latitude and longitude
        r = math.sqrt(sum(x*x for x in pos_n))
        if r > 0:
            factors['latitude'] = math.degrees(math.asin(pos_n[2] / r))
            factors['longitude'] = math.degrees(math.atan2(pos_n[1], pos_n[0]))
        else:
            factors['latitude'] = 0.0
            factors['longitude'] = 0.0
        
        # Beta angle (simplified)
        if timestep > 0:
            prev_pos = data['pos_n'][timestep-1]
            vel = [pos_n[i] - prev_pos[i] for i in range(3)]
            orbit_normal = [
                pos_n[1]*vel[2] - pos_n[2]*vel[1],
                pos_n[2]*vel[0] - pos_n[0]*vel[2],
                pos_n[0]*vel[1] - pos_n[1]*vel[0]
            ]
            if math.sqrt(sum(x*x for x in orbit_normal)) > 0:
                factors['beta_angle'] = 90 - angle_deg(svn, orbit_normal)
            else:
                factors['beta_angle'] = 0.0
        else:
            factors['beta_angle'] = 0.0
        
        # CSS statistics
        albedo_row = data['albedo'][timestep]
        illum_row = data['illum'][timestep]
        
        # Ensure same length
        n_css = min(len(albedo_row), len(illum_row), 6)
        albedo_row = albedo_row[:n_css]
        illum_row = illum_row[:n_css]
        
        factors['total_albedo'] = sum(albedo_row)
        factors['total_illum'] = sum(illum_row)
        factors['active_css'] = sum(1 for i in illum_row if i > 0)
        
        return factors
    
    def calculate_sun_errors(self, data: Dict, timestep: int) -> Dict:
        """Calculate sun vector estimation errors"""
        errors = {}
        
        svb_truth = unit(data['svb'][timestep])
        illum_row = data['illum'][timestep]
        albedo_row = data['albedo'][timestep]
        
        # Ensure same length
        n_css = min(len(albedo_row), len(illum_row), 6)
        albedo_row = albedo_row[:n_css]
        illum_row = illum_row[:n_css]
        
        # With albedo
        svb_with = css_reconstruct_sun_direction(illum_row, self.css_axes)
        if svb_with:
            errors['sun_error_with'] = angle_deg(svb_with, svb_truth)
        else:
            errors['sun_error_with'] = np.nan
        
        # Without albedo
        no_albedo = [max(0, illum_row[i] - albedo_row[i]) for i in range(n_css)]
        svb_no = css_reconstruct_sun_direction(no_albedo, self.css_axes)
        if svb_no:
            errors['sun_error_no'] = angle_deg(svb_no, svb_truth)
        else:
            errors['sun_error_no'] = np.nan
        
        # Improvement
        if not np.isnan(errors['sun_error_with']) and not np.isnan(errors['sun_error_no']):
            errors['error_reduction'] = errors['sun_error_with'] - errors['sun_error_no']
        else:
            errors['error_reduction'] = np.nan
        
        return errors
    
    def process_all_configs(self):
        """Process all valid configurations"""
        valid_configs = self.find_valid_configs()
        
        for i, config_id in enumerate(valid_configs):
            print(f"Processing {config_id} ({i+1}/{len(valid_configs)})...")
            
            data = self.load_config_data(config_id)
            if not data:
                continue
            
            static_factors = data['metadata']['parameters']
            
            # Process each timestep
            for t in range(data['n_steps']):
                row = {
                    'config_id': config_id,
                    'timestep': t,
                    **static_factors
                }
                
                # Add dynamic factors
                dynamic = self.calculate_dynamic_factors(data, t)
                row.update(dynamic)
                
                # Add errors
                errors = self.calculate_sun_errors(data, t)
                row.update(errors)
                
                self.data_rows.append(row)
        
        print(f"Processed {len(self.data_rows)} total timesteps")
    
    def analyze_correlations(self):
        """Analyze correlations between factors and sun error"""
        df = pd.DataFrame(self.data_rows)
        
        # Remove rows with NaN in target variable
        df = df.dropna(subset=['sun_error_with'])
        
        # Save full dataset
        df.to_csv(self.output_dir / "analysis_data.csv", index=False)
        
        # Calculate correlations
        target = 'sun_error_with'
        exclude_cols = [
            'config_id', 'timestep', 
            'sun_error_with', 'sun_error_no', 'error_reduction',
            # albedo
            'total_albedo', 'total_illum', 'active_css'
        ]
        
        correlations = {}
        for col in df.columns:
            if col not in exclude_cols:
                try:
                    pearson_r = df[col].corr(df[target])
                    spearman_r = df[col].corr(df[target], method='spearman')
                    
                    correlations[col] = {
                        'pearson': pearson_r,
                        'spearman': spearman_r,
                        'abs_mean': (abs(pearson_r) + abs(spearman_r)) / 2
                    }
                except:
                    continue
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), 
                           key=lambda x: x[1]['abs_mean'], 
                           reverse=True)
        
        # Save correlations
        corr_df = pd.DataFrame.from_dict(correlations, orient='index')
        corr_df.to_csv(self.output_dir / "factor_correlations.csv")
        
        # Generate report
        self.generate_report(sorted_corr[:10], df)
        
        # Generate visualizations
        self.generate_visualizations(sorted_corr[:10], df)
    
    def generate_report(self, top_10: List, df: pd.DataFrame):
        """Generate text report"""
        report_path = self.output_dir / "top_factors_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ALBEDO FACTOR ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total configurations analyzed: {df['config_id'].nunique()}\n")
            f.write(f"Total timesteps analyzed: {len(df)}\n")
            f.write(f"Mean sun error with albedo: {df['sun_error_with'].mean():.3f}°\n")
            f.write(f"Mean sun error without albedo: {df['sun_error_no'].mean():.3f}°\n\n")
            
            f.write("TOP 10 FACTORS AFFECTING SUN VECTOR ERROR (WITH ALBEDO)\n")
            f.write("-"*60 + "\n")
            
            for i, (factor, corr) in enumerate(top_10, 1):
                f.write(f"\n{i}. {factor}\n")
                f.write(f"   Pearson r = {corr['pearson']:.4f}\n")
                f.write(f"   Spearman ρ = {corr['spearman']:.4f}\n")
                f.write(f"   Mean |correlation| = {corr['abs_mean']:.4f}\n")
        
        print(f"Report saved to {report_path}")
    
    def generate_visualizations(self, top_10: List, df: pd.DataFrame):
        """Generate visualization plots"""
        # Correlation bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        factors = [f[0] for f in top_10]
        pearson = [f[1]['pearson'] for f in top_10]
        
        bars = ax.barh(range(len(factors)), pearson)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel('Pearson Correlation with Sun Error')
        ax.set_title('Top 10 Factors Affecting CSS Sun Vector Error')
        ax.grid(True, alpha=0.3)
        
        # Color bars by sign
        for i, bar in enumerate(bars):
            if pearson[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "top_factors_bar.pdf")
        plt.close()
        
        # Scatter plots for top 5
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (factor, _) in enumerate(top_10[:6]):
            ax = axes[i]
            
            # Sample data to avoid overplotting
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size)
            
            ax.scatter(df_sample[factor], df_sample['sun_error_with'], 
                      alpha=0.3, s=1)
            ax.set_xlabel(factor)
            ax.set_ylabel('Sun Error (deg)')
            ax.set_title(f'{factor} vs Sun Error')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Factor-Error Relationships', fontsize=14)
        plt.tight_layout()
        fig.savefig(self.output_dir / "factor_scatter_plots.pdf")
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def run(self):
        """Run complete analysis"""
        print("Starting Albedo Factor Analysis...")
        self.process_all_configs()
        
        if len(self.data_rows) == 0:
            print("No data to analyze!")
            return
        
        self.analyze_correlations()
        print("Analysis complete!")


def main():
    parser = argparse.ArgumentParser(description='Analyze factors affecting CSS sun vector error')
    parser.add_argument('--param-dir',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/parameter_generation/parameter_output/parameter_sets',
                       help='Parameter configurations directory')
    parser.add_argument('--data-dir',
                       default='<DATA_DIR>',
                       help='Simulation data directory')
    parser.add_argument('--output-dir',
                       default='./results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = AlbedoFactorAnalyzer(
        param_dir=args.param_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    analyzer.run()


if __name__ == "__main__":
    main()