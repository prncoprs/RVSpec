#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figures for Albedo Factor Analysis Paper
Creates publication-quality heatmap and distribution plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Set style for paper-quality figures
plt.style.use('classic')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 26,
    'axes.titlesize': 32,
    'axes.labelsize': 29,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

import matplotlib
matplotlib.rcParams.update({
    "pdf.fonttype": 42,   # Use TrueType (Type 42)
    "ps.fonttype": 42,
})


class AlbedoPlotter:
    def __init__(self, data_path: str, output_dir: str = "albedo_figures"):
        """Initialize plotter"""
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Remove NaN values
        self.df = self.df.dropna(subset=['sun_error_with'])
        print(f"Loaded {len(self.df)} valid data points")
        
        # Define top 5 factors with their correlation values
        self.top_factors = {
            'global_phase_angle': 0.5869,
            'is_daylight': 0.5421,
            'MTB_0_SATURATION': 0.3090,
            'ORB_ARG_PERIAPSIS': 0.3054,
            'ORB_RAAN': 0.1627
        }
        
        self.factor_labels = {
            'global_phase_angle': 'Global\nPhase\nAngle\n',
            'is_daylight': 'Eclipse\nStatus',
            'MTB_0_SATURATION': 'MTB\nSaturation',
            'ORB_ARG_PERIAPSIS': 'Periapsis',
            'ORB_RAAN': 'RAAN'
        }
    
    def plot_factor_importance_heatmap(self):
        """Create vertical heatmap for top 5 factors"""
        fig, ax = plt.subplots(figsize=(5, 7))
        fig.patch.set_facecolor('white')
        
        # Extract factors and values (already sorted by importance)
        factors = list(self.top_factors.keys())
        values = list(self.top_factors.values())
        labels = [self.factor_labels[f] for f in factors]
        
        # Create heatmap data as column vector
        heatmap_data = np.array([[v] for v in values])
        
        # Create custom colormap - gradient from white to deep red (for albedo theme)
        colors_list = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', 
                       '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d']
        # 
        colors_list = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                       '#6baed6', '#4292c6', '#2171b5', '#084594']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom_reds', colors_list, N=n_bins)
        
        # Position and width of heatmap
        left = 0.3
        right = 0.7
        bottom = len(labels) - 0.5
        top = -0.5
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=0.7,
                      interpolation='bilinear', extent=[left, right, bottom, top], 
                      origin='upper')
        
        # Set labels
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=30, ha='center')
        ax.set_xticks([])
        ax.set_xlim(-0.1, 1.0)
        
        # Add right-side label
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        ax2.set_ylabel('|Correlation with Sun\nVector Estimation Error|', fontsize=32,
                      rotation=270, labelpad=20)
        
        # Add value annotations
        for i, val in enumerate(values):
            text_color = 'white' if val > 0.35 else 'black'
            text = ax.text((left + right) / 2, i, f'{val:.3f}',
                          ha='center', va='center',
                          fontsize=24,
                          color=text_color)
            if val > 0.35:
                text.set_path_effects([patheffects.withStroke(linewidth=2, 
                                                             foreground='black', 
                                                             alpha=0.3)])
        
        # Add grid lines between cells
        for i in range(len(labels) - 1):
            ax.plot([left, right], [i + 0.5, i + 0.5],
                   color='black', linewidth=1.5, alpha=0.8)
        
        # Add frame around heatmap
        rect = Rectangle((left, -0.5), right-left, len(labels),
                        linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)
        ax2.tick_params(axis='both', which='both', length=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'albedo_factors_heatmap.pdf', 
                   format='pdf', dpi=300, bbox_inches='tight')
        print("Saved: albedo_factors_heatmap.pdf")
    
    def plot_sun_error_distribution(self):
        """Create sun vector error distribution histogram"""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Get sun error values
        error_values = self.df['sun_error_with']
        
        # Calculate statistics
        mean_error = error_values.mean()
        median_error = error_values.median()
        min_error = error_values.min()
        max_error = error_values.max()
        std_error = error_values.std()
        
        # Create histogram
        n_bins = min(40, int(len(error_values) / 50))
        counts, bins, patches = ax.hist(error_values, bins=n_bins,
                                       color='white', alpha=1.0,
                                       edgecolor='black', linewidth=1.5,
                                       density=False)
        
        # Add hatching pattern
        for patch in patches:
            patch.set_hatch('///')
        
        # Add mean line
        ax.axvline(mean_error, color='black', linestyle='-', linewidth=3)
        
        # Create legend with statistics
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=3,
                      label=f'Mean = {mean_error:.2f}°'),
            # plt.Line2D([0], [0], color='none',
            #           label=f'Std = {std_error:.2f}°'),
            plt.Line2D([0], [0], color='none',
                      label=f'Min = {min_error:.2f}°'),
            plt.Line2D([0], [0], color='none',
                      label=f'Max = {max_error:.2f}°')
        ]
        
        ax.set_xlabel('Sun Vector Estimation Error (°)', fontsize=29)
        ax.set_ylabel('Frequency', fontsize=29)
        
        # Add legend
        legend = ax.legend(handles=legend_elements, loc='upper right',
                          fontsize=28, frameon=True,
                          fancybox=False, shadow=False, framealpha=0.95,
                          edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.2)
        
        # Remove grid
        ax.grid(False)
        ax.set_axisbelow(True)
        
        # Set border style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', direction='in', width=1.2, length=6)
        ax.tick_params(axis='y', labelsize=28, left=True, right=False)
        ax.tick_params(axis='x', labelsize=28, bottom=True, top=False)
        
        # Set y-axis limit
        ax.set_ylim(0, max(counts) * 1.1)
        y_max = ax.get_ylim()[1]
        ax.set_yticks(np.arange(0, y_max + 1, 200))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sun_error_distribution.pdf',
                   format='pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("Saved: sun_error_distribution.pdf")
    
    def plot_error_comparison_distribution(self):
        """Create comparison distribution of errors with and without albedo"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        # Error with albedo
        error_with = self.df['sun_error_with']
        error_no = self.df['sun_error_no']
        
        # Plot for error WITH albedo
        n_bins = min(40, int(len(error_with) / 50))
        counts1, bins1, patches1 = ax1.hist(error_with, bins=n_bins,
                                           color='white', alpha=1.0,
                                           edgecolor='black', linewidth=1.5)
        for patch in patches1:
            patch.set_hatch('///')
        
        ax1.axvline(error_with.mean(), color='black', linestyle='-', linewidth=3)
        ax1.set_xlabel('Sun Vector Error [degrees]', fontsize=26)
        ax1.set_ylabel('Frequency', fontsize=26)
        ax1.set_title('With Albedo', fontsize=28)
        
        # Plot for error WITHOUT albedo
        counts2, bins2, patches2 = ax2.hist(error_no, bins=n_bins,
                                           color='white', alpha=1.0,
                                           edgecolor='black', linewidth=1.5)
        for patch in patches2:
            patch.set_hatch('\\\\\\')
        
        ax2.axvline(error_no.mean(), color='black', linestyle='-', linewidth=3)
        ax2.set_xlabel('Sun Vector Error [degrees]', fontsize=26)
        ax2.set_ylabel('Frequency', fontsize=26)
        ax2.set_title('Without Albedo', fontsize=28)
        
        # Style both axes
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.tick_params(axis='both', which='major', direction='in', 
                          width=1.2, length=6, labelsize=24)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_comparison_distribution.pdf',
                   format='pdf', dpi=300, bbox_inches='tight')
        print("Saved: error_comparison_distribution.pdf")
    
    def generate_summary_report(self):
        """Generate summary statistics report"""
        report_file = self.output_dir / 'albedo_analysis_summary.txt'
        
        error_with = self.df['sun_error_with']
        error_no = self.df['sun_error_no']
        
        with open(report_file, 'w') as f:
            f.write("ALBEDO FACTOR ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total data points: {len(self.df)}\n")
            f.write(f"Unique configurations: {self.df['config_id'].nunique()}\n\n")
            
            f.write("SUN VECTOR ERROR STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write("With Albedo:\n")
            f.write(f"  Mean: {error_with.mean():.3f}°\n")
            f.write(f"  Std: {error_with.std():.3f}°\n")
            f.write(f"  Min: {error_with.min():.3f}°\n")
            f.write(f"  Max: {error_with.max():.3f}°\n\n")
            
            f.write("Without Albedo:\n")
            f.write(f"  Mean: {error_no.mean():.3f}°\n")
            f.write(f"  Std: {error_no.std():.3f}°\n")
            f.write(f"  Min: {error_no.min():.3f}°\n")
            f.write(f"  Max: {error_no.max():.3f}°\n\n")
            
            f.write("TOP 5 INFLUENCING FACTORS:\n")
            f.write("-" * 30 + "\n")
            for i, (factor, corr) in enumerate(self.top_factors.items(), 1):
                label = self.factor_labels[factor].replace('\n', ' ')
                f.write(f"{i}. {label:20}: |r| = {corr:.4f}\n")
        
        print(f"Summary report saved to: {report_file}")
    
    def run_all_plots(self):
        """Generate all plots and reports"""
        print("\n" + "="*60)
        print("GENERATING ALBEDO ANALYSIS FIGURES")
        print("="*60 + "\n")
        
        print("Creating factor importance heatmap...")
        self.plot_factor_importance_heatmap()
        
        print("Creating sun error distribution...")
        self.plot_sun_error_distribution()
        
        print("Creating error comparison distribution...")
        self.plot_error_comparison_distribution()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"\nAll figures saved to: {self.output_dir}/")
        print("Generated files:")
        print("  - albedo_factors_heatmap.pdf")
        print("  - sun_error_distribution.pdf")
        print("  - error_comparison_distribution.pdf")
        print("  - albedo_analysis_summary.txt")


def main():
    # Configuration
    data_path = "<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/results/analysis_data.csv"
    output_dir = "albedo_figures"
    
    # Create plotter and generate figures
    plotter = AlbedoPlotter(data_path, output_dir)
    plotter.run_all_plots()


if __name__ == "__main__":
    main()