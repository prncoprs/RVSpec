"""
Enhanced State Filter - Remove Unreasonable States
==================================================

This script identifies and removes states that are fundamentally problematic
based on the analysis results, including:
1. States with extreme outliers (R² < -1000)
2. States that fail catastrophically across ALL models
3. Derived/computed states that may be unstable
4. States with very poor predictability patterns
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def identify_unreasonable_states(analysis_dir, outlier_threshold=-10, extreme_threshold=-1000):
    """
    Identify states that should be removed based on various criteria.
    
    Returns:
        dict: Categories of unreasonable states with lists of state names
    """
    print(f"\n IDENTIFYING UNREASONABLE STATES IN {analysis_dir}")
    print("="*60)
    
    unreasonable_states = {
        'extreme_outliers': [],      # States with R² < -1000
        'universal_failures': [],    # States failing on ALL models
        'error_derived': [],         # Error/difference states
        'standard_deviation': [],    # Standard deviation states
        'computed_derived': [],      # Other computed/derived states
        'sensor_noise': [],          # Likely sensor noise states
        'consistently_poor': []      # States with poor performance across >90% models
    }
    
    # Load catastrophic and poor performance analyses
    catastrophic_file = os.path.join(analysis_dir, "catastrophic_states_analysis.csv")
    poor_performance_file = os.path.join(analysis_dir, "poor_performance_states_analysis.csv")
    
    if os.path.exists(catastrophic_file):
        df_catastrophic = pd.read_csv(catastrophic_file)
        print(f" Loaded {len(df_catastrophic)} catastrophic states")
        
        # Find extreme outliers
        extreme_outliers = df_catastrophic[df_catastrophic['worst_r2'] < extreme_threshold]
        unreasonable_states['extreme_outliers'] = extreme_outliers['state_name'].tolist()
        
        # Find states failing on ALL models (failure_breadth >= 6 for 7 models)
        universal_failures = df_catastrophic[df_catastrophic['failure_breadth'] >= 6]
        unreasonable_states['universal_failures'] = universal_failures['state_name'].tolist()
        
        print(f" Found {len(unreasonable_states['extreme_outliers'])} extreme outliers (R² < {extreme_threshold})")
        print(f" Found {len(unreasonable_states['universal_failures'])} universal failures (fail on ≥6 models)")
    
    if os.path.exists(poor_performance_file):
        df_poor = pd.read_csv(poor_performance_file)
        print(f" Loaded {len(df_poor)} poor performing states")
        
        # Find states with consistently poor performance (>90% of models)
        total_models = 7
        consistently_poor = df_poor[df_poor['failure_breadth'] >= (total_models * 0.9)]
        unreasonable_states['consistently_poor'] = consistently_poor['state_name'].tolist()
        
        print(f" Found {len(unreasonable_states['consistently_poor'])} consistently poor states")
    
    # Analyze all states from coefficient files to identify problematic patterns
    all_states = set()
    model_suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    
    for phase in range(1, 13):
        for suffix in model_suffixes:
            coef_file = os.path.join(analysis_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            if os.path.exists(coef_file):
                df = pd.read_csv(coef_file)
                if 'state' in df.columns:
                    all_states.update(df['state'].tolist())
    
    print(f" Analyzing {len(all_states)} unique states across all files")
    
    # Categorize states by naming patterns
    for state in all_states:
        state_lower = state.lower()
        
        # Error/difference states
        if any(keyword in state_lower for keyword in ['error', 'diff', 'delta', 'residual']):
            unreasonable_states['error_derived'].append(state)
        
        # Standard deviation states
        elif '_std_' in state_lower or state_lower.endswith('_std'):
            unreasonable_states['standard_deviation'].append(state)
        
        # Computed/derived states (often unstable)
        elif any(keyword in state_lower for keyword in ['gpa_sms', 'xkf4', 'rate', 'computed']):
            unreasonable_states['computed_derived'].append(state)
        
        # Potential sensor noise indicators
        elif any(keyword in state_lower for keyword in ['noise', 'raw', 'uncalibrated']):
            unreasonable_states['sensor_noise'].append(state)
    
    # Remove duplicates and print summary
    for category, states in unreasonable_states.items():
        unreasonable_states[category] = list(set(states))
        print(f" {category.replace('_', ' ').title()}: {len(unreasonable_states[category])} states")
    
    return unreasonable_states

def analyze_state_patterns(analysis_dir):
    """
    Analyze patterns in state names to understand what's causing issues.
    """
    print(f"\n ANALYZING STATE PATTERNS")
    print("="*40)
    
    # Collect all states and their performance
    state_performance = defaultdict(list)
    model_suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    
    for phase in range(1, 13):
        for suffix in model_suffixes:
            coef_file = os.path.join(analysis_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            if os.path.exists(coef_file):
                df = pd.read_csv(coef_file)
                if 'state' in df.columns and 'r2_cv' in df.columns:
                    for _, row in df.iterrows():
                        state_performance[row['state']].append(row['r2_cv'])
    
    # Analyze patterns
    pattern_analysis = {
        'error_states': [],
        'std_states': [],
        'mean_states': [],
        'derived_states': [],
        'sensor_states': [],
        'attitude_states': [],
        'velocity_states': [],
        'position_states': []
    }
    
    pattern_performance = {
        'error_states': [],
        'std_states': [],
        'mean_states': [],
        'derived_states': [],
        'sensor_states': [],
        'attitude_states': [],
        'velocity_states': [],
        'position_states': []
    }
    
    for state, r2_values in state_performance.items():
        state_lower = state.lower()
        avg_r2 = np.mean(r2_values)
        
        # Categorize by patterns
        if 'error' in state_lower:
            pattern_analysis['error_states'].append(state)
            pattern_performance['error_states'].extend(r2_values)
        elif '_std_' in state_lower:
            pattern_analysis['std_states'].append(state)
            pattern_performance['std_states'].extend(r2_values)
        elif '_mean_' in state_lower:
            pattern_analysis['mean_states'].append(state)
            pattern_performance['mean_states'].extend(r2_values)
        elif any(keyword in state_lower for keyword in ['gpa', 'xkf', 'computed']):
            pattern_analysis['derived_states'].append(state)
            pattern_performance['derived_states'].extend(r2_values)
        elif any(keyword in state_lower for keyword in ['att', 'roll', 'pitch', 'yaw']):
            pattern_analysis['attitude_states'].append(state)
            pattern_performance['attitude_states'].extend(r2_values)
        elif any(keyword in state_lower for keyword in ['vel', 'speed']):
            pattern_analysis['velocity_states'].append(state)
            pattern_performance['velocity_states'].extend(r2_values)
        elif any(keyword in state_lower for keyword in ['pos', 'alt', 'lat', 'lng']):
            pattern_analysis['position_states'].append(state)
            pattern_performance['position_states'].extend(r2_values)
        else:
            pattern_analysis['sensor_states'].append(state)
            pattern_performance['sensor_states'].extend(r2_values)
    
    # Print analysis
    print(f" STATE PATTERN ANALYSIS:")
    print(f"{'Category':<20} {'Count':<8} {'Mean R²':<12} {'Median R²':<12} {'% R² < -10':<12}")
    print("-" * 70)
    
    for category, states in pattern_analysis.items():
        if len(states) > 0:
            r2_values = pattern_performance[category]
            mean_r2 = np.mean(r2_values)
            median_r2 = np.median(r2_values)
            pct_catastrophic = (sum(1 for r2 in r2_values if r2 < -10) / len(r2_values)) * 100
            
            print(f"{category:<20} {len(states):<8} {mean_r2:<12.4f} {median_r2:<12.4f} {pct_catastrophic:<12.1f}%")
    
    return pattern_analysis, pattern_performance

def create_filtering_recommendations(unreasonable_states, pattern_analysis, pattern_performance):
    """
    Create specific recommendations for which states to filter out.
    """
    print(f"\n FILTERING RECOMMENDATIONS")
    print("="*50)
    
    recommendations = {
        'definitely_remove': [],
        'probably_remove': [],
        'consider_remove': [],
        'keep_but_investigate': []
    }
    
    # Definitely remove
    recommendations['definitely_remove'].extend(unreasonable_states['extreme_outliers'])
    recommendations['definitely_remove'].extend(unreasonable_states['universal_failures'])
    
    # Probably remove
    recommendations['probably_remove'].extend(unreasonable_states['error_derived'])
    recommendations['probably_remove'].extend(unreasonable_states['consistently_poor'])
    
    # Consider removing
    recommendations['consider_remove'].extend(unreasonable_states['standard_deviation'])
    recommendations['consider_remove'].extend(unreasonable_states['computed_derived'])
    
    # Keep but investigate
    recommendations['keep_but_investigate'].extend(unreasonable_states['sensor_noise'])
    
    # Remove duplicates
    for category in recommendations:
        recommendations[category] = list(set(recommendations[category]))
    
    print(f" DEFINITELY REMOVE ({len(recommendations['definitely_remove'])} states):")
    print("   - Extreme outliers (R² < -1000)")
    print("   - Universal failures (fail on all models)")
    print("   - These states are fundamentally unpredictable")
    
    print(f"\n PROBABLY REMOVE ({len(recommendations['probably_remove'])} states):")
    print("   - Error/derived states (often unstable)")
    print("   - Consistently poor performers (>90% models)")
    print("   - These states add noise without information")
    
    print(f"\n CONSIDER REMOVING ({len(recommendations['consider_remove'])} states):")
    print("   - Standard deviation states (may be noisy)")
    print("   - Computed/derived states (may be unstable)")
    print("   - Review case-by-case based on domain knowledge")
    
    print(f"\n KEEP BUT INVESTIGATE ({len(recommendations['keep_but_investigate'])} states):")
    print("   - Sensor noise indicators")
    print("   - May need preprocessing rather than removal")
    
    return recommendations

def apply_state_filtering(analysis_dir, recommendations, filter_level='conservative'):
    """
    Apply state filtering based on recommendations.
    
    Args:
        filter_level: 'conservative' (only definitely_remove), 
                     'moderate' (+ probably_remove), 
                     'aggressive' (+ consider_remove)
    """
    print(f"\n APPLYING {filter_level.upper()} FILTERING")
    print("="*50)
    
    # Determine which states to remove
    states_to_remove = set()
    
    if filter_level in ['conservative', 'moderate', 'aggressive']:
        states_to_remove.update(recommendations['definitely_remove'])
        
    if filter_level in ['moderate', 'aggressive']:
        states_to_remove.update(recommendations['probably_remove'])
        
    if filter_level == 'aggressive':
        states_to_remove.update(recommendations['consider_remove'])
    
    print(f" Removing {len(states_to_remove)} states with {filter_level} filtering")
    
    # Apply filtering to all coefficient files
    model_suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    files_processed = 0
    states_removed_count = 0
    
    for phase in range(1, 13):
        for suffix in model_suffixes:
            coef_file = os.path.join(analysis_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            
            if os.path.exists(coef_file):
                df = pd.read_csv(coef_file)
                
                if 'state' in df.columns:
                    original_count = len(df)
                    
                    # Remove problematic states
                    df_filtered = df[~df['state'].isin(states_to_remove)].copy()
                    
                    removed_count = original_count - len(df_filtered)
                    states_removed_count += removed_count
                    
                    # Save filtered version
                    filtered_file = coef_file.replace('.csv', f'_state_filtered_{filter_level}.csv')
                    df_filtered.to_csv(filtered_file, index=False)
                    
                    files_processed += 1
                    
                    if removed_count > 0:
                        print(f" {os.path.basename(coef_file)}: {original_count} → {len(df_filtered)} ({removed_count} removed)")
    
    print(f"\n Processed {files_processed} files")
    print(f"  Removed {states_removed_count} state entries total")
    
    # Save list of removed states
    removed_states_file = os.path.join(analysis_dir, f"removed_states_{filter_level}.csv")
    pd.DataFrame({'removed_states': list(states_to_remove)}).to_csv(removed_states_file, index=False)
    print(f" List of removed states saved to: {removed_states_file}")
    
    return states_to_remove

def regenerate_analysis_with_filtered_states(analysis_dir, filter_level='conservative'):
    """
    Regenerate the analysis using the state-filtered data.
    """
    print(f"\n REGENERATING ANALYSIS WITH FILTERED STATES")
    print("="*50)
    
    from outlier_filter_v1 import regenerate_summary_from_filtered_data, regenerate_model_comparison_table
    
    # Regenerate summaries using state-filtered data
    model_suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    
    for suffix in model_suffixes:
        print(f" Regenerating summary for {suffix if suffix else 'polynomial_ridge'}...")
        
        # Process state-filtered files
        all_phase_summaries = []
        
        for phase in range(1, 13):
            filtered_file = os.path.join(analysis_dir, f"multivariate_coefficients_phase{phase}{suffix}_state_filtered_{filter_level}.csv")
            
            if os.path.exists(filtered_file):
                df = pd.read_csv(filtered_file)
                
                if 'r2_cv' in df.columns and len(df) > 0:
                    # Apply R² outlier filtering as well
                    df_clean = df[df['r2_cv'] > -10].copy()
                    
                    if len(df_clean) > 0:
                        r2_values = df_clean['r2_cv'].values
                        
                        summary = {
                            "Phase": phase,
                            "Num_Models": len(r2_values),
                            "Num_R2_GTE_0.9": sum(r >= 0.9 for r in r2_values),
                            "Num_R2_GTE_0.8": sum(r >= 0.8 for r in r2_values),
                            "Num_R2_GTE_0.7": sum(r >= 0.7 for r in r2_values),
                            "Mean_R2_CV": np.mean(r2_values),
                            "Median_R2_CV": np.median(r2_values),
                            "Min_R2_CV": np.min(r2_values),
                            "Max_R2_CV": np.max(r2_values),
                        }
                        all_phase_summaries.append(summary)
        
        # Save new summary
        if all_phase_summaries:
            summary_df = pd.DataFrame(all_phase_summaries)
            new_summary_path = os.path.join(analysis_dir, f"multivariate_r2_summary{suffix}_state_filtered_{filter_level}.csv")
            summary_df.to_csv(new_summary_path, index=False)
    
    # Regenerate comparison table
    regenerate_model_comparison_table_state_filtered(analysis_dir, filter_level)

def regenerate_model_comparison_table_state_filtered(output_dir, filter_level):
    """Generate model comparison table using state-filtered data."""
    
    model_configs = [
        {"name": "Polynomial_Ridge", "suffix": "", "display_name": "Polynomial + Ridge"},
        {"name": "Gaussian_Process", "suffix": "_gpr", "display_name": "Gaussian Process"},
        {"name": "XGBoost", "suffix": "_xgb", "display_name": "XGBoost"},
        {"name": "LightGBM", "suffix": "_lgb", "display_name": "LightGBM"},
        {"name": "Neural_Network", "suffix": "_nn", "display_name": "Neural Network"},
        {"name": "Random_Forest", "suffix": "_rf", "display_name": "Random Forest"},
        {"name": "SVR", "suffix": "_svr", "display_name": "Support Vector Regression"}
    ]
    
    comparison_results = []
    
    for model_config in model_configs:
        display_name = model_config["display_name"]
        suffix = model_config["suffix"]
        
        summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}_state_filtered_{filter_level}.csv")
        
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            
            # Calculate overall statistics
            overall_stats = {
                "Model": display_name,
                "Total_Models": df_summary["Num_Models"].sum(),
                "Total_R2_GTE_0.9": df_summary["Num_R2_GTE_0.9"].sum(),
                "Total_R2_GTE_0.8": df_summary["Num_R2_GTE_0.8"].sum(),
                "Total_R2_GTE_0.7": df_summary["Num_R2_GTE_0.7"].sum(),
                "Overall_Mean_R2": df_summary["Mean_R2_CV"].mean(),
                "Overall_Median_R2": df_summary["Median_R2_CV"].median(),
                "Best_Phase_R2": df_summary["Mean_R2_CV"].max(),
                "Worst_Phase_R2": df_summary["Mean_R2_CV"].min(),
                "R2_Std_Across_Phases": df_summary["Mean_R2_CV"].std()
            }
            
            # Calculate percentages
            if overall_stats["Total_Models"] > 0:
                overall_stats["Pct_R2_GTE_0.9"] = (overall_stats["Total_R2_GTE_0.9"] / overall_stats["Total_Models"]) * 100
                overall_stats["Pct_R2_GTE_0.8"] = (overall_stats["Total_R2_GTE_0.8"] / overall_stats["Total_Models"]) * 100
                overall_stats["Pct_R2_GTE_0.7"] = (overall_stats["Total_R2_GTE_0.7"] / overall_stats["Total_Models"]) * 100
            else:
                overall_stats["Pct_R2_GTE_0.9"] = 0
                overall_stats["Pct_R2_GTE_0.8"] = 0
                overall_stats["Pct_R2_GTE_0.7"] = 0
                
            comparison_results.append(overall_stats)
    
    # Convert to DataFrame and sort by overall mean R²
    df_comparison = pd.DataFrame(comparison_results)
    if not df_comparison.empty:
        df_comparison = df_comparison.sort_values("Overall_Mean_R2", ascending=False)
        
        # Round numerical columns
        numerical_cols = df_comparison.select_dtypes(include=[np.number]).columns
        df_comparison[numerical_cols] = df_comparison[numerical_cols].round(4)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"model_performance_comparison_state_filtered_{filter_level}.csv")
        df_comparison.to_csv(output_path, index=False)
        
        print(f" State-filtered comparison table saved to: {output_path}")
        
        # Print summary
        print(f"\n STATE-FILTERED MODEL PERFORMANCE ({filter_level.upper()}):")
        print(df_comparison[['Model', 'Total_Models', 'Overall_Mean_R2', 'Pct_R2_GTE_0.7', 'Pct_R2_GTE_0.8', 'Pct_R2_GTE_0.9']].to_string(index=False))
    
    return df_comparison

def main():
    """Main function to run the enhanced state filtering analysis."""
    
    print(" ENHANCED STATE FILTERING ANALYSIS")
    print("=" * 60)
    print("This script identifies and removes fundamentally problematic states")
    print("to improve model performance and interpretability.")
    print()
    
    simulators = ["sitl", "gazebo"]
    filter_levels = ['conservative', 'moderate', 'aggressive']
    
    for sim in simulators:
        analysis_dir = f"./analysis_output_{sim}"
        
        if not os.path.exists(analysis_dir):
            print(f"  Directory not found: {analysis_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f" ANALYZING {sim.upper()} SIMULATOR")
        print(f"{'='*60}")
        
        # Step 1: Identify unreasonable states
        unreasonable_states = identify_unreasonable_states(analysis_dir)
        
        # Step 2: Analyze state patterns
        pattern_analysis, pattern_performance = analyze_state_patterns(analysis_dir)
        
        # Step 3: Create filtering recommendations
        recommendations = create_filtering_recommendations(unreasonable_states, pattern_analysis, pattern_performance)
        
        # Step 4: Apply different levels of filtering
        for filter_level in filter_levels:
            print(f"\n{'='*40}")
            print(f"APPLYING {filter_level.upper()} FILTERING")
            print(f"{'='*40}")
            
            states_removed = apply_state_filtering(analysis_dir, recommendations, filter_level)
            regenerate_analysis_with_filtered_states(analysis_dir, filter_level)
    
    print(f"\n ENHANCED STATE FILTERING COMPLETE!")
    print("\n Files created for each simulator:")
    print("   • *_state_filtered_conservative.csv - Remove extreme outliers + universal failures")
    print("   • *_state_filtered_moderate.csv - Also remove error states + consistently poor")
    print("   • *_state_filtered_aggressive.csv - Also remove std + computed states")
    print("   • removed_states_*.csv - Lists of removed states")
    print("   • model_performance_comparison_state_filtered_*.csv - Updated performance tables")
    print("\n Recommendations:")
    print("   • Start with CONSERVATIVE filtering for safety")
    print("   • Use MODERATE filtering for better performance")
    print("   • Use AGGRESSIVE filtering only if you understand the domain impact")
    print("   • Always validate results with domain expertise")

if __name__ == "__main__":
    main()