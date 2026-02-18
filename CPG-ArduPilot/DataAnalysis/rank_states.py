"""
State Performance Ranking Analysis Script
=========================================

This script analyzes the performance of different states across all phases and models to:
1. Rank states from best to worst performing based on R² values
2. Identify consistently high-performing states
3. Identify problematic states that should be excluded
4. Generate comprehensive reports and visualizations
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def collect_all_state_performances(analysis_output_dir, exclude_keywords=None, state_column='state', 
                                 r2_column='r2_cv', outlier_threshold=-10):
    """
    Collect R² performance data for all states across all phases and models.
    
    Returns:
        DataFrame with columns: ['base_state', 'phase', 'model_type', 'r2_cv', 'simulator', 'original_state']
    """
    print(f" Collecting state performance data from {analysis_output_dir}")
    
    all_data = []
    simulator_name = os.path.basename(analysis_output_dir).replace("analysis_output_", "")
    
    # Model suffixes and their display names
    model_configs = {
        "": "Polynomial_Ridge",
        "_gpr": "Gaussian_Process", 
        "_xgb": "XGBoost",
        "_lgb": "LightGBM",
        "_nn": "Neural_Network",
        "_rf": "Random_Forest",
        "_svr": "SVR"
    }
    
    # Collect data from all phases and models
    for phase in range(1, 13):
        for suffix, model_name in model_configs.items():
            coef_file = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            
            if not os.path.exists(coef_file):
                continue
                
            try:
                df = pd.read_csv(coef_file)
                
                # Check required columns
                if state_column not in df.columns or r2_column not in df.columns:
                    continue
                
                # Filter outliers
                df_clean = df[df[r2_column] > outlier_threshold].copy()
                
                # Apply keyword filtering if specified
                if exclude_keywords:
                    keep_mask = pd.Series([True] * len(df_clean), index=df_clean.index)
                    for keyword_pattern in exclude_keywords:
                        import fnmatch
                        for idx, state in df_clean[state_column].items():
                            if fnmatch.fnmatch(state, keyword_pattern):
                                keep_mask[idx] = False
                    df_clean = df_clean[keep_mask]
                
                # Extract base state name by removing phase suffix
                df_clean['base_state'] = df_clean[state_column].str.replace(f'_Phase{phase}', '', regex=False)
                df_clean['original_state'] = df_clean[state_column]
                
                # Add metadata
                df_clean['phase'] = phase
                df_clean['model_type'] = model_name
                df_clean['simulator'] = simulator_name
                
                # Select relevant columns
                state_data = df_clean[['base_state', 'phase', 'model_type', r2_column, 'simulator', 'original_state']].copy()
                state_data.columns = ['base_state', 'phase', 'model_type', 'r2_cv', 'simulator', 'original_state']
                
                all_data.append(state_data)
                
            except Exception as e:
                print(f"Warning: Error processing {coef_file}: {e}")
                continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"   Collected {len(combined_df)} state performance records")
        print(f"   Unique base states: {combined_df['base_state'].nunique()}")
        print(f"   Phases covered: {sorted(combined_df['phase'].unique())}")
        print(f"   Models covered: {sorted(combined_df['model_type'].unique())}")
        return combined_df
    else:
        print("   No data collected")
        return pd.DataFrame()


def analyze_state_performance(state_data_df):
    """
    Analyze performance statistics for each base state across all phases and models.
    
    Returns:
        DataFrame with comprehensive state performance statistics
    """
    print("\n Analyzing state performance statistics...")
    
    if state_data_df.empty:
        return pd.DataFrame()
    
    # Group by base_state and calculate comprehensive statistics
    state_stats = []
    
    for base_state in state_data_df['base_state'].unique():
        state_records = state_data_df[state_data_df['base_state'] == base_state]['r2_cv']
        
        if len(state_records) == 0:
            continue
            
        # Calculate statistics
        stats = {
            'base_state': base_state,
            'count': len(state_records),
            'mean_r2': state_records.mean(),
            'median_r2': state_records.median(),
            'std_r2': state_records.std(),
            'min_r2': state_records.min(),
            'max_r2': state_records.max(),
            'q25_r2': state_records.quantile(0.25),
            'q75_r2': state_records.quantile(0.75),
            'iqr_r2': state_records.quantile(0.75) - state_records.quantile(0.25),
            'cv_r2': state_records.std() / state_records.mean() if state_records.mean() != 0 else np.inf,
            'count_r2_gte_0_9': (state_records >= 0.9).sum(),
            'count_r2_gte_0_8': (state_records >= 0.8).sum(),
            'count_r2_gte_0_7': (state_records >= 0.7).sum(),
            'count_r2_gte_0_5': (state_records >= 0.5).sum(),
            'count_r2_lt_0': (state_records < 0).sum(),
            'pct_r2_gte_0_9': (state_records >= 0.9).mean() * 100,
            'pct_r2_gte_0_8': (state_records >= 0.8).mean() * 100,
            'pct_r2_gte_0_7': (state_records >= 0.7).mean() * 100,
            'pct_r2_gte_0_5': (state_records >= 0.5).mean() * 100,
            'pct_r2_lt_0': (state_records < 0).mean() * 100,
        }
        
        # Count unique phases and models this base state appears in
        state_subset = state_data_df[state_data_df['base_state'] == base_state]
        stats['unique_phases'] = state_subset['phase'].nunique()
        stats['unique_models'] = state_subset['model_type'].nunique()
        stats['unique_simulators'] = state_subset['simulator'].nunique()
        
        # Calculate consistency metrics across phases and models
        phase_means = state_subset.groupby('phase')['r2_cv'].mean()
        model_means = state_subset.groupby('model_type')['r2_cv'].mean()
        
        stats['phase_consistency'] = 1 / (1 + phase_means.std()) if len(phase_means) > 1 else 1.0
        stats['model_consistency'] = 1 / (1 + model_means.std()) if len(model_means) > 1 else 1.0
        
        # List of phases where this state appears
        stats['phases_present'] = sorted(state_subset['phase'].unique())
        stats['phases_present_str'] = ','.join(map(str, stats['phases_present']))
        
        # Overall performance score (weighted combination)
        stats['performance_score'] = (
            0.4 * stats['mean_r2'] +
            0.2 * stats['median_r2'] +
            0.2 * (stats['pct_r2_gte_0_8'] / 100) +
            0.1 * stats['phase_consistency'] +
            0.1 * stats['model_consistency']
        )
        
        state_stats.append(stats)
    
    # Convert to DataFrame and sort by performance score
    df_stats = pd.DataFrame(state_stats)
    df_stats = df_stats.sort_values('performance_score', ascending=False)
    
    print(f"   Analyzed {len(df_stats)} unique base states")
    
    return df_stats


def identify_best_worst_states(state_stats_df, top_n=20, bottom_n=20):
    """
    Identify the best and worst performing base states.
    """
    print(f"\n Identifying top {top_n} best and {bottom_n} worst performing base states...")
    
    if state_stats_df.empty:
        return None, None
    
    # Best states: high performance score, good consistency, high mean R²
    best_states = state_stats_df.head(top_n).copy()
    
    # Worst states: low performance score, poor consistency, low mean R²
    worst_states = state_stats_df.tail(bottom_n).copy()
    
    print(f"   Best base states (top {len(best_states)}):")
    for idx, row in best_states.head(10).iterrows():
        print(f"     {row['base_state']:30} | Mean R²: {row['mean_r2']:.3f} | Score: {row['performance_score']:.3f} | "
              f"Count: {row['count']:3d} | Phases: {row['unique_phases']}")
    
    print(f"   Worst base states (bottom {len(worst_states)}):")
    for idx, row in worst_states.tail(10).iterrows():
        print(f"     {row['base_state']:30} | Mean R²: {row['mean_r2']:.3f} | Score: {row['performance_score']:.3f} | "
              f"Count: {row['count']:3d} | Phases: {row['unique_phases']}")
    
    return best_states, worst_states


def analyze_state_categories(state_stats_df):
    """
    Categorize states by performance and analyze patterns.
    """
    print("\n  Analyzing state categories...")
    
    if state_stats_df.empty:
        return {}
    
    # Define performance categories
    categories = {
        'Excellent': state_stats_df[state_stats_df['mean_r2'] >= 0.9],
        'Good': state_stats_df[(state_stats_df['mean_r2'] >= 0.7) & (state_stats_df['mean_r2'] < 0.9)],
        'Fair': state_stats_df[(state_stats_df['mean_r2'] >= 0.5) & (state_stats_df['mean_r2'] < 0.7)],
        'Poor': state_stats_df[(state_stats_df['mean_r2'] >= 0.0) & (state_stats_df['mean_r2'] < 0.5)],
        'Very Poor': state_stats_df[state_stats_df['mean_r2'] < 0.0]
    }
    
    category_stats = {}
    
    for cat_name, cat_df in categories.items():
        if len(cat_df) > 0:
            category_stats[cat_name] = {
                'count': len(cat_df),
                'percentage': len(cat_df) / len(state_stats_df) * 100,
                'mean_r2_range': f"{cat_df['mean_r2'].min():.3f} to {cat_df['mean_r2'].max():.3f}",
                'avg_consistency': cat_df['phase_consistency'].mean(),
                'top_states': cat_df.head(5)['base_state'].tolist()  # Fixed: use 'base_state' instead of 'state'
            }
            
            print(f"   {cat_name:12} | Count: {len(cat_df):4d} ({len(cat_df)/len(state_stats_df)*100:5.1f}%) | "
                  f"R² Range: {cat_df['mean_r2'].min():.3f}-{cat_df['mean_r2'].max():.3f}")
    
    return category_stats


def analyze_sensor_type_patterns(state_stats_df):
    """
    Analyze performance patterns by sensor/state type for base states.
    """
    print("\n Analyzing sensor type patterns for base states...")
    
    if state_stats_df.empty:
        return {}
    
    # Define sensor type patterns based on actual data structure
    sensor_patterns = {
        'GPS': ['GPS_'],
        'IMU_ACCEL': ['IMU_Acc'],
        'IMU_GYRO': ['IMU_Gyr'],
        'IMU_OTHER': ['IMU_T', 'IMU_AHz', 'IMU_GHz'],
        'ATTITUDE': ['ATT_'],
        'BAROMETER': ['BARO_'],
        'MAGNETOMETER': ['MAG_'],
        'POSITION': ['POS_'],
        'SIMULATION': ['SIM_', 'SIM2_'],
        'EXTENDED_KALMAN': ['XKF'],
        'VIBRATION': ['VIBE_'],
        'GPS_ANALYSIS': ['GPA_'],
        'BATTERY': ['Battery_'],
        'MISSION': ['Mission_'],
        'ERROR_METRICS': ['_Error', 'Error_'],
        'RPY_METRICS': ['Roll_', 'Pitch_', 'Yaw_', 'RPY_']
    }
    
    sensor_stats = {}
    
    for sensor_type, patterns in sensor_patterns.items():
        # Find base states matching this sensor type
        matching_states = state_stats_df[
            state_stats_df['base_state'].str.contains('|'.join(patterns), case=False, na=False)
        ]
        
        if len(matching_states) > 0:
            sensor_stats[sensor_type] = {
                'count': len(matching_states),
                'mean_r2': matching_states['mean_r2'].mean(),
                'median_r2': matching_states['median_r2'].median(),
                'std_r2': matching_states['mean_r2'].std(),
                'best_state': matching_states.loc[matching_states['mean_r2'].idxmax(), 'base_state'],
                'worst_state': matching_states.loc[matching_states['mean_r2'].idxmin(), 'base_state'],
                'pct_good': (matching_states['mean_r2'] >= 0.7).mean() * 100
            }
            
            print(f"   {sensor_type:15} | Count: {len(matching_states):3d} | "
                  f"Mean R²: {matching_states['mean_r2'].mean():.3f} | "
                  f"% Good (≥0.7): {(matching_states['mean_r2'] >= 0.7).mean()*100:5.1f}%")
    
    return sensor_stats


def create_performance_plots(state_stats_df, output_dir, simulator_name):
    """
    Create visualization plots for state performance analysis.
    """
    print(f"\n Creating performance visualization plots...")
    
    if state_stats_df.empty:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory for plots
    plot_dir = os.path.join(output_dir, "state_performance_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Distribution of mean R² values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(state_stats_df['mean_r2'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean R²')
    plt.ylabel('Number of States')
    plt.title(f'Distribution of State Performance\n({simulator_name.upper()})')
    plt.axvline(state_stats_df['mean_r2'].median(), color='red', linestyle='--', label=f'Median: {state_stats_df["mean_r2"].median():.3f}')
    plt.legend()
    
    # 2. Performance vs Count scatter
    plt.subplot(1, 2, 2)
    plt.scatter(state_stats_df['count'], state_stats_df['mean_r2'], alpha=0.6)
    plt.xlabel('Number of Predictions')
    plt.ylabel('Mean R²')
    plt.title('Performance vs Sample Size')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{simulator_name}_performance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top 20 best and worst states
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Best states
    best_20 = state_stats_df.head(20)
    ax1.barh(range(len(best_20)), best_20['mean_r2'], color='green', alpha=0.7)
    ax1.set_yticks(range(len(best_20)))
    ax1.set_yticklabels(best_20['base_state'], fontsize=8)  # Fixed: use 'base_state' instead of 'state'
    ax1.set_xlabel('Mean R²')
    ax1.set_title(f'Top 20 Best Performing States ({simulator_name.upper()})')
    ax1.grid(axis='x', alpha=0.3)
    
    # Worst states
    worst_20 = state_stats_df.tail(20)
    ax2.barh(range(len(worst_20)), worst_20['mean_r2'], color='red', alpha=0.7)
    ax2.set_yticks(range(len(worst_20)))
    ax2.set_yticklabels(worst_20['base_state'], fontsize=8)  # Fixed: use 'base_state' instead of 'state'
    ax2.set_xlabel('Mean R²')
    ax2.set_title(f'Top 20 Worst Performing States ({simulator_name.upper()})')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{simulator_name}_best_worst_states.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Plots saved to: {plot_dir}")


def save_detailed_reports(state_stats_df, best_states, worst_states, category_stats, 
                         sensor_stats, output_dir, simulator_name):
    """
    Save comprehensive reports to CSV files.
    """
    print(f"\n Saving detailed reports...")
    
    # 1. Complete state performance statistics
    complete_report_path = os.path.join(output_dir, f'{simulator_name}_complete_state_performance.csv')
    state_stats_df.to_csv(complete_report_path, index=False)
    print(f"   Complete state performance: {complete_report_path}")
    
    # 2. Best performing states
    if best_states is not None:
        best_report_path = os.path.join(output_dir, f'{simulator_name}_best_performing_states.csv')
        best_states.to_csv(best_report_path, index=False)
        print(f"   Best performing states: {best_report_path}")
    
    # 3. Worst performing states  
    if worst_states is not None:
        worst_report_path = os.path.join(output_dir, f'{simulator_name}_worst_performing_states.csv')
        worst_states.to_csv(worst_report_path, index=False)
        print(f"   Worst performing states: {worst_report_path}")
    
    # 4. Category summary
    if category_stats:
        category_summary = []
        for cat_name, stats in category_stats.items():
            category_summary.append({
                'Category': cat_name,
                'Count': stats['count'],
                'Percentage': stats['percentage'],
                'R2_Range': stats['mean_r2_range'],
                'Avg_Consistency': stats['avg_consistency'],
                'Top_States': '; '.join(stats['top_states'])
            })
        
        category_df = pd.DataFrame(category_summary)
        category_report_path = os.path.join(output_dir, f'{simulator_name}_performance_categories.csv')
        category_df.to_csv(category_report_path, index=False)
        print(f"   Performance categories: {category_report_path}")
    
    # 5. Sensor type analysis
    if sensor_stats:
        sensor_summary = []
        for sensor_type, stats in sensor_stats.items():
            sensor_summary.append({
                'Sensor_Type': sensor_type,
                'Count': stats['count'],
                'Mean_R2': stats['mean_r2'],
                'Median_R2': stats['median_r2'],
                'Std_R2': stats['std_r2'],
                'Best_State': stats['best_state'],
                'Worst_State': stats['worst_state'],
                'Pct_Good_Performance': stats['pct_good']
            })
        
        sensor_df = pd.DataFrame(sensor_summary)
        sensor_df = sensor_df.sort_values('Mean_R2', ascending=False)
        sensor_report_path = os.path.join(output_dir, f'{simulator_name}_sensor_type_performance.csv')
        sensor_df.to_csv(sensor_report_path, index=False)
        print(f"   Sensor type performance: {sensor_report_path}")


def generate_exclusion_recommendations(state_stats_df, worst_threshold=0.3, 
                                     consistency_threshold=0.5, min_count=10):
    """
    Generate recommendations for base states to exclude based on poor performance.
    """
    print(f"\n  Generating exclusion recommendations for base states...")
    
    if state_stats_df.empty:
        return []
    
    # Criteria for exclusion:
    # 1. Mean R² below threshold
    # 2. Low consistency across phases/models
    # 3. Sufficient sample size to be confident
    
    exclude_candidates = state_stats_df[
        (state_stats_df['mean_r2'] < worst_threshold) &
        (state_stats_df['phase_consistency'] < consistency_threshold) &
        (state_stats_df['count'] >= min_count)
    ]
    
    print(f"   Found {len(exclude_candidates)} base states recommended for exclusion:")
    print(f"   Criteria: Mean R² < {worst_threshold}, Consistency < {consistency_threshold}, Count ≥ {min_count}")
    
    exclusion_list = []
    for idx, row in exclude_candidates.iterrows():
        exclusion_info = {
            'base_state': row['base_state'],
            'mean_r2': row['mean_r2'],
            'consistency': row['phase_consistency'],
            'count': row['count'],
            'unique_phases': row['unique_phases'],
            'phases_present': row['phases_present_str'],
            'reason': f"Low R² ({row['mean_r2']:.3f}) and poor consistency ({row['phase_consistency']:.3f})"
        }
        exclusion_list.append(exclusion_info)
        print(f"     {row['base_state']:30} | R²: {row['mean_r2']:.3f} | Consistency: {row['phase_consistency']:.3f} | "
              f"Count: {row['count']} | Phases: {row['unique_phases']}")
    
    return exclusion_list


def analyze_cross_simulator_states(simulators=["sitl", "gazebo"], exclude_keywords=None):
    """
    Compare base state performance across different simulators.
    """
    print(f"\n Analyzing base state performance across simulators...")
    
    all_simulator_data = []
    
    for sim in simulators:
        analysis_dir = f"./analysis_output_{sim}"
        if not os.path.exists(analysis_dir):
            print(f"   Warning: Directory not found: {analysis_dir}")
            continue
            
        # Collect data for this simulator
        sim_data = collect_all_state_performances(analysis_dir, exclude_keywords)
        if not sim_data.empty:
            all_simulator_data.append(sim_data)
    
    if not all_simulator_data:
        print("   No simulator data found")
        return None, None
    
    # Combine all simulator data
    combined_data = pd.concat(all_simulator_data, ignore_index=True)
    
    # Analyze base states that appear in multiple simulators
    state_sim_summary = combined_data.groupby(['base_state', 'simulator'])['r2_cv'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    # Find base states that appear in all simulators
    state_counts = combined_data.groupby('base_state')['simulator'].nunique()
    common_states = state_counts[state_counts == len(simulators)].index
    
    print(f"   Base states appearing in all {len(simulators)} simulators: {len(common_states)}")
    
    # Compare performance across simulators for common base states
    cross_sim_comparison = []
    for base_state in common_states:
        state_data = combined_data[combined_data['base_state'] == base_state]
        sim_performance = state_data.groupby('simulator')['r2_cv'].mean()
        
        comparison = {
            'base_state': base_state,
            'overall_mean': state_data['r2_cv'].mean(),
            'overall_std': state_data['r2_cv'].std(),
            'sim_difference': sim_performance.max() - sim_performance.min(),
            'consistent_across_sims': sim_performance.std() < 0.1  # Low std = consistent
        }
        
        for sim in simulators:
            if sim in sim_performance.index:
                comparison[f'{sim}_mean_r2'] = sim_performance[sim]
            else:
                comparison[f'{sim}_mean_r2'] = np.nan
                
        cross_sim_comparison.append(comparison)
    
    cross_sim_df = pd.DataFrame(cross_sim_comparison)
    cross_sim_df = cross_sim_df.sort_values('overall_mean', ascending=False)
    
    return cross_sim_df, combined_data


def main():
    """
    Main function to analyze state performance across all phases and models.
    """
    print(" STATE PERFORMANCE RANKING ANALYSIS")
    print("=" * 60)
    
    # Configuration
    simulators = ["sitl", "gazebo"]
    exclude_keywords = [
        "*ATT_*", "BARO_*", "GPS_*", "*MAG_*", "*IMU_*", "*XKF*", "*VIBE_*"
    ]  # Updated to match actual data patterns, set to None to include all states
    
    # Create output directory for results
    output_base_dir = "./state_analysis_output"
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Output directory: {output_base_dir}")
    if exclude_keywords:
        print(f"Excluding states matching: {exclude_keywords}")
    else:
        print("Including all states")
    
    # Analyze each simulator individually
    for sim in simulators:
        analysis_dir = f"./analysis_output_{sim}"
        
        if not os.path.exists(analysis_dir):
            print(f"  Directory not found: {analysis_dir}")
            continue
            
        print(f"\n Analyzing {sim.upper()} simulator...")
        
        # Collect state performance data
        state_data = collect_all_state_performances(analysis_dir, exclude_keywords)
        
        if state_data.empty:
            print(f"   No data found for {sim}")
            continue
        
        # Analyze performance statistics
        state_stats = analyze_state_performance(state_data)
        
        # Identify best and worst states
        best_states, worst_states = identify_best_worst_states(state_stats)
        
        # Analyze performance categories
        category_stats = analyze_state_categories(state_stats)
        
        # Analyze sensor type patterns
        sensor_stats = analyze_sensor_type_patterns(state_stats)
        
        # Create visualization plots
        create_performance_plots(state_stats, output_base_dir, sim)
        
        # Generate exclusion recommendations
        exclusion_recommendations = generate_exclusion_recommendations(state_stats)
        
        # Save all reports
        save_detailed_reports(state_stats, best_states, worst_states, 
                            category_stats, sensor_stats, output_base_dir, sim)
        
        # Save exclusion recommendations
        if exclusion_recommendations:
            exclusion_df = pd.DataFrame(exclusion_recommendations)
            exclusion_path = os.path.join(output_base_dir, f'{sim}_exclusion_recommendations.csv')
            exclusion_df.to_csv(exclusion_path, index=False)
            print(f"   Exclusion recommendations: {exclusion_path}")
    
    # Cross-simulator analysis
    cross_sim_df, combined_data = analyze_cross_simulator_states(simulators, exclude_keywords)
    
    if cross_sim_df is not None:
        print(f"\n Cross-simulator analysis completed")
        
        # Save cross-simulator comparison
        cross_sim_path = os.path.join(output_base_dir, "cross_simulator_state_comparison.csv")
        cross_sim_df.to_csv(cross_sim_path, index=False)
        print(f"   Cross-simulator comparison: {cross_sim_path}")
        
        # Analyze overall state performance across all simulators
        overall_stats = analyze_state_performance(combined_data)
        overall_path = os.path.join(output_base_dir, "overall_state_performance_all_simulators.csv")
        overall_stats.to_csv(overall_path, index=False)
        print(f"   Overall state performance: {overall_path}")
        
        # Create overall visualization
        create_performance_plots(overall_stats, output_base_dir, "all_simulators")
    
    print(f"\n STATE PERFORMANCE ANALYSIS COMPLETE!")
    print(" Summary of generated files:")
    print("   • Individual simulator state performance rankings")
    print("   • Best and worst performing states lists")
    print("   • Performance category breakdowns")
    print("   • Sensor type performance analysis")
    print("   • Exclusion recommendations based on poor performance")
    print("   • Cross-simulator comparison for common states")
    print("   • Visualization plots for all analyses")
    
    print(f"\n INSIGHTS:")
    print("   • Review 'best_performing_states.csv' for consistently reliable states")
    print("   • Review 'exclusion_recommendations.csv' for states to filter out")
    print("   • Check sensor type analysis to understand which sensors perform best")
    print("   • Use cross-simulator comparison to identify simulator-specific issues")


if __name__ == "__main__":
    main()