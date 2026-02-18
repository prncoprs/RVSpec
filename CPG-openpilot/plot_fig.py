#!/usr/bin/env python3
"""
Tau Analysis Script for Paper Figures
Analyzes the relationship between core factors and required tau values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Set style for paper-quality figures - using the reference style
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
    "pdf.fonttype": 42,   #  TrueType (Type 42) Type 3
    "ps.fonttype": 42,
})


class TauAnalyzer:
    def __init__(self, csv_path: str, output_dir: str = "tau_analysis_output"):
        """Initialize tau analyzer"""
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load and clean data
        self.df = self.load_and_clean_data()
        
        # Define factor names and labels for plotting
        self.factor_names = [
            'applied_mass', 'applied_drag_coefficient', 'applied_tire_friction', 
            'applied_brake_torque_front', 'applied_road_friction'
        ]
        
        self.factor_labels = {
            'applied_mass': 'Vehicle Mass (kg)',
            'applied_drag_coefficient': 'Drag Coefficient',
            'applied_tire_friction': 'Tire Friction Coefficient', 
            'applied_brake_torque_front': 'Brake Torque (Nm)',
            'applied_road_friction': 'Road Friction Coefficient'
        }
        
        self.factor_short_labels = {
            'applied_mass': 'Mass',
            'applied_drag_coefficient': 'Drag',
            'applied_tire_friction': 'Tire\nFriction', 
            'applied_brake_torque_front': 'Brake\nTorque',
            'applied_road_friction': 'Road\nFriction'
        }
        
    def load_and_clean_data(self):
        """Load and clean the CSV data"""
        print(f"Loading data from: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Filter successful experiments only
        df_clean = df[df['success'] == True].copy()
        
        # Remove outliers (tau > 10s likely indicates errors)
        df_clean = df_clean[df_clean['required_tau'] < 10.0]
        
        print(f"Loaded {len(df)} total experiments")
        print(f"Using {len(df_clean)} successful experiments for analysis")
        print(f"Tau range: {df_clean['required_tau'].min():.2f}s - {df_clean['required_tau'].max():.2f}s")
        
        return df_clean
    
    def calculate_correlations(self):
        """Calculate correlation coefficients between factors and tau"""
        correlations = {}
        
        for factor in self.factor_names:
            if factor in self.df.columns:
                corr_coef = self.df[factor].corr(self.df['required_tau'])
                correlations[factor] = corr_coef
        
        # Sort by absolute correlation value
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: abs(x[1]), reverse=True))
        
        print("\nFactor Importance (Correlation with Tau):")
        print("-" * 50)
        for factor, corr in sorted_correlations.items():
            print(f"{self.factor_short_labels[factor]:15}: {corr:6.3f}")
        
        return sorted_correlations
    
    def plot_factor_importance(self, correlations):
        """Create a beautiful vertical heatmap for factor importance"""
        fig, ax = plt.subplots(figsize=(5, 7))  # Increased width from 4 to 5.5
        fig.patch.set_facecolor('white')
        
        # Sort factors by absolute correlation (highest first)
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Extract factors, values and labels - NO REVERSAL, keep highest at top
        factors = [item[0] for item in sorted_items]
        values = [abs(item[1]) for item in sorted_items]  # Use absolute values
        
        # Create better labels - shorter names for compactness
        labels = []
        for f in factors:
            label = self.factor_short_labels[f]
            # Keep line breaks for two-word terms
            if ' ' in label and '\n' not in label:
                words = label.split(' ')
                if len(words) == 2:
                    labels.append(f'{words[0]}\n{words[1]}')
                else:
                    labels.append(label)
            else:
                labels.append(label)
        
        # Create heatmap data as a column vector
        # Note: imshow displays first element at top, so data order matches label order
        heatmap_data = np.array([[v] for v in values])
        
        # Create custom colormap - use gradient from white to deep blue
        colors_list = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom_blues', colors_list, N=n_bins)
        
        # Adjust the position and width of the heatmap
        # Use extent to control the position and width of the heatmap
        left = 0.2  # Start position (shift right)
        right = 0.6  # End position (make narrower)
        bottom = len(labels) - 0.5  # FLIP: bottom is at top index
        top = -0.5  # FLIP: top is at bottom index
        
        # Create the heatmap with specific extent
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=0.7, 
                      interpolation='bilinear', extent=[left, right, bottom, top], origin='upper')
        
        # Set ticks and labels - CENTER ALIGNED
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=30, ha='center')
        ax.set_xticks([])  # Remove x-axis ticks
        
        # Set x-axis limits to show full width
        ax.set_xlim(-0.1, 1.0)
        
        # Add vertical label on the right side
        ax2 = ax.twinx()  # Create a twin axis for the right side
        ax2.set_ylim(ax.get_ylim())  # Match the y-limits
        ax2.set_yticks([])  # Remove right y-ticks
        ax2.set_ylabel('|Correlation with τ|', fontsize=32,
                      rotation=270, labelpad=-25)  # Reduced padding from 35 to 18
        
        # Add value annotations with better styling - adjust x position
        for i, val in enumerate(values):
            # Determine text color based on background darkness
            text_color = 'white' if val > 0.35 else 'black'
            
            # Add the correlation value with outline for visibility
            # Position at center of narrowed heatmap
            text = ax.text((left + right) / 2, i, f'{val:.3f}', 
                          ha='center', va='center',
                          fontsize=24,
                          color=text_color)
            
            # Add subtle stroke effect for white text
            if val > 0.35:
                text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black', alpha=0.3)])
        
        # Add MORE VISIBLE grid lines between cells - only within heatmap bounds
        for i in range(len(labels) - 1):
            ax.plot([left, right], [i + 0.5, i + 0.5], 
                   color='black', linewidth=1.5, alpha=0.8)
        
        # Add frame around the heatmap - only around the actual heatmap area
        from matplotlib.patches import Rectangle
        rect = Rectangle((left, -0.5), right-left, len(labels), 
                        linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # Hide the default spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Also hide spines for ax2
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)
        ax2.tick_params(axis='both', which='both', length=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'carla_factors_heatmap.pdf', format='pdf', dpi=300,
                   bbox_inches='tight')
        # plt.show()  # Commented out as requested
    
    def plot_tau_distribution(self):
        """Create tau distribution histogram - improved style matching reference"""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        tau_values = self.df['required_tau']
        
        # Calculate statistics
        mean_tau = tau_values.mean()
        median_tau = tau_values.median()
        min_tau = tau_values.min()
        max_tau = tau_values.max()
        
        # Histogram with style matching reference
        n_bins = min(40, int(len(tau_values) / 15))
        counts, bins, patches = ax.hist(tau_values, bins=n_bins, 
                                       color='white', alpha=1.0,
                                       edgecolor='black', linewidth=1.5,
                                       density=False)
        
        # Add hatching pattern to bars for visual interest
        for patch in patches:
            patch.set_hatch('///')
        
        # Add mean line only
        ax.axvline(mean_tau, color='black', linestyle='-', linewidth=3)
        
        # Create custom legend with all statistics - INCREASED FONT SIZE
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=3, 
                      label=f'Mean = {mean_tau:.2f}s'),
            # plt.Line2D([0], [0], color='none', 
            #           label=f'Median = {median_tau:.2f}s'),
            plt.Line2D([0], [0], color='none', 
                      label=f'Min = {min_tau:.2f}s'),
            plt.Line2D([0], [0], color='none', 
                      label=f'Max = {max_tau:.2f}s')
        ]
        
        ax.set_xlabel('τ (seconds)', fontsize=29)
        ax.set_ylabel('Frequency', fontsize=29)
        
        # Legend in upper right with LARGER FONT
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          fontsize=28, frameon=True,  # Increased from 24 to 28
                          fancybox=False, shadow=False, framealpha=0.95,
                          edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(1.2)
        
        # Remove grid
        ax.grid(False)
        ax.set_axisbelow(True)
        
        # Set border style - hide top and right borders
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
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tau_distribution.pdf', format='pdf', dpi=300,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        # plt.show()  # Commented out as requested
    
    def generate_summary_report(self, correlations):
        """Generate summary report"""
        report_file = self.output_dir / 'tau_analysis_report.txt'
        
        tau_values = self.df['required_tau']
        
        with open(report_file, 'w') as f:
            f.write("TAU ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total successful experiments: {len(self.df)}\n")
            f.write(f"Tau range: {tau_values.min():.3f}s - {tau_values.max():.3f}s\n")
            f.write(f"Tau mean: {tau_values.mean():.3f}s\n")
            f.write(f"Tau median: {tau_values.median():.3f}s\n")
            f.write(f"Tau std deviation: {tau_values.std():.3f}s\n\n")
            
            f.write("FACTOR IMPORTANCE RANKING:\n")
            f.write("-" * 30 + "\n")
            for i, (factor, corr) in enumerate(correlations.items(), 1):
                label = self.factor_short_labels[factor].replace('\n', ' ')
                f.write(f"{i}. {label:15}: r = {corr:6.3f}\n")
            
            f.write(f"\nDYNAMIC TAU ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tau range: {tau_values.max() - tau_values.min():.2f}s\n")
            f.write(f"Dynamic tau provides {tau_values.max() - tau_values.min():.2f}s adaptation range\n")
        
        print(f"\nSummary report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run complete tau analysis"""
        print("Starting Tau Analysis for Paper Figures...")
        print("=" * 60)
        
        # Calculate correlations
        correlations = self.calculate_correlations()
        
        # Generate plots with new styles
        print("\nGenerating plots...")
        print("Creating gradient heatmap bar chart for factor importance...")
        self.plot_factor_importance(correlations)
        print("Creating tau distribution histogram...")
        self.plot_tau_distribution()
        
        # Generate summary report
        self.generate_summary_report(correlations)
        
        print(f"\nAnalysis complete! All figures saved to: {self.output_dir}")
        print("Generated files:")
        print("  - carla_factors_heatmap.pdf (gradient bar chart)")
        print("  - tau_distribution.pdf")
        print("  - tau_analysis_report.txt")


def main():
    # Configuration
    csv_path = "core_experiment_results/core_factors_brake_results.csv"
    output_dir = "tau_analysis_output"
    
    # Run analysis
    analyzer = TauAnalyzer(csv_path, output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()