"""
Enhanced Post-Processing Script to Filter Outliers and Exclude States by Keywords
================================================================================

This script processes your existing CSV files to:
1. Remove extreme R² outliers
2. Exclude states based on keyword patterns (e.g., *ATT_*, BARO_, GPS_)
3. Regenerate clean performance summaries
"""

import pandas as pd
import numpy as np
import os
import glob
import fnmatch

def filter_states_by_keywords(df, state_column='state', exclude_keywords=None):
    """
    Filter out states that match specified keyword patterns.
    
    Args:
        df: DataFrame containing model results
        state_column: Name of the column containing state names
        exclude_keywords: List of keyword patterns to exclude (supports wildcards)
    
    Returns:
        Filtered DataFrame with excluded states removed
    """
    if exclude_keywords is None:
        exclude_keywords = []
    
    if state_column not in df.columns:
        print(f"Warning: {state_column} column not found in DataFrame")
        return df
    
    original_count = len(df)
    
    # Create a mask for states to keep
    keep_mask = pd.Series([True] * len(df), index=df.index)
    
    excluded_states = set()
    
    for keyword_pattern in exclude_keywords:
        # Use fnmatch for wildcard pattern matching
        for idx, state in df[state_column].items():
            if fnmatch.fnmatch(state, keyword_pattern):
                keep_mask[idx] = False
                excluded_states.add(state)
    
    # Apply the filter
    df_filtered = df[keep_mask].copy()
    
    filtered_count = len(df_filtered)
    states_removed = original_count - filtered_count
    
    if states_removed > 0:
        print(f"    Excluded {states_removed}/{original_count} entries ({states_removed/original_count*100:.1f}%) based on keywords")
        print(f"    Excluded state patterns: {sorted(excluded_states)}")
    
    return df_filtered


def filter_outliers_from_csv(csv_path, r2_column='r2_cv', outlier_threshold=-5, 
                            exclude_keywords=None, state_column='state'):
    """
    Filter extreme outliers and excluded states from a CSV file containing model results.
    
    Args:
        csv_path: Path to CSV file
        r2_column: Name of the R² column
        outlier_threshold: R² values below this are considered outliers
        exclude_keywords: List of keyword patterns to exclude states
        state_column: Name of the column containing state names
    
    Returns:
        Filtered DataFrame
    """
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    if r2_column not in df.columns:
        print(f"Warning: {r2_column} not found in {csv_path}")
        return df
    
    print(f"  Processing {os.path.basename(csv_path)}:")
    
    # Count original
    original_count = len(df)
    print(f"    Original entries: {original_count}")
    
    # First, filter by keywords if specified
    if exclude_keywords:
        df = filter_states_by_keywords(df, state_column, exclude_keywords)
        keyword_filtered_count = len(df)
        print(f"    After keyword filtering: {keyword_filtered_count}")
    
    # Then filter extreme outliers
    df_filtered = df[df[r2_column] > outlier_threshold].copy()
    
    # Count after filtering
    final_count = len(df_filtered)
    outliers_removed = len(df) - final_count
    total_removed = original_count - final_count
    
    if outliers_removed > 0:
        print(f"    Outliers removed (R² < {outlier_threshold}): {outliers_removed}")
    
    print(f"    Final entries: {final_count} (removed {total_removed} total, {total_removed/original_count*100:.1f}%)")
    
    return df_filtered


def regenerate_summary_from_filtered_data(analysis_output_dir, suffix="", outlier_threshold=-10, 
                                        exclude_keywords=None, state_column='state', exclude_phases=None):
    """
    Regenerate R² summary after filtering outliers, excluded states, and excluded phases.
    
    Args:
        exclude_phases: List of phase numbers to exclude (e.g., [1, 2, 12])
    """
    print(f"\nProcessing {analysis_output_dir} (suffix: '{suffix}')")
    
    if exclude_phases is None:
        exclude_phases = []
    
    if exclude_phases:
        print(f"  Excluding phases: {exclude_phases}")
    
    all_phase_summaries = []
    
    # Process each phase
    for phase in range(1, 13):
        # Skip excluded phases
        if phase in exclude_phases:
            print(f"  Skipping phase {phase} (excluded)")
            continue
            
        coef_file = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
        
        if not os.path.exists(coef_file):
            continue
            
        # Filter outliers and excluded states from this phase
        df_filtered = filter_outliers_from_csv(coef_file, 
                                             outlier_threshold=outlier_threshold,
                                             exclude_keywords=exclude_keywords,
                                             state_column=state_column)
        
        if df_filtered is None or df_filtered.empty:
            continue
            
        # Recalculate summary statistics
        r2_values = df_filtered['r2_cv'].values
        
        if len(r2_values) > 0:
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
            
            # Save filtered data back
            output_suffix = "_filtered"
            if exclude_keywords:
                output_suffix += "_excluded"
            if exclude_phases:
                output_suffix += "_phases"
            df_filtered.to_csv(coef_file.replace('.csv', f'{output_suffix}.csv'), index=False)
    
    # Save new summary
    if all_phase_summaries:
        summary_df = pd.DataFrame(all_phase_summaries)
        output_suffix = "_filtered"
        if exclude_keywords:
            output_suffix += "_excluded"
        if exclude_phases:
            output_suffix += "_phases"
        new_summary_path = os.path.join(analysis_output_dir, f"multivariate_r2_summary{suffix}{output_suffix}.csv")
        summary_df.to_csv(new_summary_path, index=False)
        print(f"   Saved filtered summary: {new_summary_path}")
        
        return summary_df
    
    return None


def regenerate_model_comparison_table(output_dir, use_filtered=True, use_excluded=False, use_phase_exclusion=False):
    """Regenerate the model performance comparison table using filtered data."""
    
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
        
        # Determine the appropriate summary file
        if use_filtered and use_excluded and use_phase_exclusion:
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}_filtered_excluded_phases.csv")
        elif use_filtered and use_excluded:
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}_filtered_excluded.csv")
        elif use_filtered and use_phase_exclusion:
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}_filtered_phases.csv")
        elif use_filtered:
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}_filtered.csv")
        else:
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}.csv")
        
        if not os.path.exists(summary_path):
            # Fallback to original if filtered version doesn't exist
            summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}.csv")
        
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found for {display_name}: {summary_path}")
            continue
            
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
            "Overall_Max_R2": df_summary["Max_R2_CV"].max(),
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
        output_suffix = ""
        if use_filtered and use_excluded and use_phase_exclusion:
            output_suffix = "_filtered_excluded_phases"
        elif use_filtered and use_excluded:
            output_suffix = "_filtered_excluded"
        elif use_filtered and use_phase_exclusion:
            output_suffix = "_filtered_phases"
        elif use_filtered:
            output_suffix = "_filtered"
        
        output_path = os.path.join(output_dir, f"model_performance_comparison{output_suffix}.csv")
        df_comparison.to_csv(output_path, index=False)
        print(f" Model performance comparison saved to: {output_path}")
        
        # Create paper-ready table
        paper_table = df_comparison[[
            "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", "Overall_Max_R2",
            "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7", 
            "Best_Phase_R2", "R2_Std_Across_Phases"
        ]].copy()
        
        # Rename columns for paper
        paper_table.columns = [
            "Model", "Total Models", "Mean R²", "Median R²", "Max R²",
            "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7", 
            "Best Phase R²", "R² Std Dev"
        ]
        
        paper_output_path = os.path.join(output_dir, f"model_performance_paper_table{output_suffix}.csv")
        paper_table.to_csv(paper_output_path, index=False)
        print(f" Paper-ready table saved to: {paper_output_path}")
        
        # Print summary to console
        simulator_name = os.path.basename(output_dir).replace("analysis_output_", "").upper()
        filter_types = []
        if use_filtered:
            filter_types.append("FILTERED")
        if use_excluded:
            filter_types.append("EXCLUDED STATES")
        if use_phase_exclusion:
            filter_types.append("EXCLUDED PHASES")
        filter_type = " + ".join(filter_types) if filter_types else "ORIGINAL"
        
        print(f"\n" + "="*80)
        print(f"{filter_type} MODEL PERFORMANCE COMPARISON - {simulator_name}")
        print("="*80)
        print(paper_table.to_string(index=False))
        print("="*80)
        
    return df_comparison


def get_phase_performance_summary(analysis_output_dir, exclude_keywords=None, state_column='state'):
    """
    Analyze performance across all phases to help decide which phases to exclude.
    """
    print(f"\n Analyzing phase performance in {analysis_output_dir}")
    
    phase_stats = []
    
    # Analyze each phase
    for phase in range(1, 13):
        phase_data = []
        
        # Collect data from all model types for this phase
        for suffix in ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]:
            coef_file = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            
            if os.path.exists(coef_file):
                df = pd.read_csv(coef_file)
                
                # Apply keyword filtering if specified
                if exclude_keywords:
                    df = filter_states_by_keywords(df, state_column, exclude_keywords)
                
                if 'r2_cv' in df.columns and len(df) > 0:
                    phase_data.extend(df['r2_cv'].tolist())
        
        if phase_data:
            # Remove extreme outliers for analysis
            phase_data_clean = [r for r in phase_data if r > -10]
            
            if phase_data_clean:
                stats = {
                    "Phase": phase,
                    "Total_Models": len(phase_data),
                    "Clean_Models": len(phase_data_clean),
                    "Outliers": len(phase_data) - len(phase_data_clean),
                    "Mean_R2": np.mean(phase_data_clean),
                    "Median_R2": np.median(phase_data_clean),
                    "Min_R2": np.min(phase_data_clean),
                    "Max_R2": np.max(phase_data_clean),
                    "Std_R2": np.std(phase_data_clean),
                    "R2_GTE_0.7": sum(r >= 0.7 for r in phase_data_clean),
                    "R2_GTE_0.8": sum(r >= 0.8 for r in phase_data_clean),
                    "R2_GTE_0.9": sum(r >= 0.9 for r in phase_data_clean)
                }
                
                # Calculate percentages
                if len(phase_data_clean) > 0:
                    stats["Pct_R2_GTE_0.7"] = (stats["R2_GTE_0.7"] / len(phase_data_clean)) * 100
                    stats["Pct_R2_GTE_0.8"] = (stats["R2_GTE_0.8"] / len(phase_data_clean)) * 100
                    stats["Pct_R2_GTE_0.9"] = (stats["R2_GTE_0.9"] / len(phase_data_clean)) * 100
                
                phase_stats.append(stats)
    
    if phase_stats:
        df_phase_stats = pd.DataFrame(phase_stats)
        
        # Sort by mean R² to identify best/worst phases
        df_phase_stats = df_phase_stats.sort_values("Mean_R2", ascending=False)
        
        print(f"\nPhase Performance Summary (sorted by Mean R²):")
        print("=" * 80)
        
        # Create a readable summary table
        summary_cols = ["Phase", "Clean_Models", "Mean_R2", "Median_R2", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.9"]
        summary_table = df_phase_stats[summary_cols].round(3)
        summary_table.columns = ["Phase", "Models", "Mean R²", "Median R²", "% ≥ 0.8", "% ≥ 0.9"]
        print(summary_table.to_string(index=False))
        
        # Identify potentially problematic phases
        mean_r2_threshold = 0.5  # You can adjust this
        low_performance_phases = df_phase_stats[df_phase_stats["Mean_R2"] < mean_r2_threshold]["Phase"].tolist()
        
        if low_performance_phases:
            print(f"\n  Phases with Mean R² < {mean_r2_threshold}: {low_performance_phases}")
            print("   Consider excluding these phases if they have consistently poor performance.")
        
        # Save detailed phase analysis
        output_path = os.path.join(analysis_output_dir, "phase_performance_analysis.csv")
        df_phase_stats.to_csv(output_path, index=False)
        print(f"\n Detailed phase analysis saved to: {output_path}")
        
        return df_phase_stats
    
    return None


def get_state_distribution(analysis_output_dir, exclude_keywords=None, state_column='state', exclude_phases=None):
    """
    Analyze the distribution of states in the dataset and show what would be excluded.
    """
    print(f"\n Analyzing state distribution in {analysis_output_dir}")
    
    if exclude_phases is None:
        exclude_phases = []
    
    all_states = []
    
    # Collect all states from all phases and models
    for phase in range(1, 13):
        # Skip excluded phases
        if phase in exclude_phases:
            continue
            
        for suffix in ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]:
            coef_file = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            
            if os.path.exists(coef_file):
                df = pd.read_csv(coef_file)
                if state_column in df.columns:
                    all_states.extend(df[state_column].tolist())
    
    if not all_states:
        print("No states found in the dataset.")
        return
    
    # Count unique states
    state_counts = pd.Series(all_states).value_counts()
    print(f"Total unique states: {len(state_counts)}")
    print(f"Total state entries: {len(all_states)}")
    
    if exclude_phases:
        print(f"(Excluding phases: {exclude_phases})")
    
    # Show top 20 most common states
    print(f"\nTop 20 most common states:")
    print(state_counts.head(20))
    
    # If exclude_keywords is provided, show what would be excluded
    if exclude_keywords:
        print(f"\nStates that would be EXCLUDED with keywords {exclude_keywords}:")
        excluded_states = []
        for state in state_counts.index:
            for keyword_pattern in exclude_keywords:
                if fnmatch.fnmatch(state, keyword_pattern):
                    excluded_states.append(state)
                    break
        
        if excluded_states:
            excluded_counts = state_counts[excluded_states]
            print(f"Number of excluded state types: {len(excluded_counts)}")
            print(f"Total excluded entries: {excluded_counts.sum()}")
            print(f"Percentage excluded: {excluded_counts.sum()/len(all_states)*100:.1f}%")
            print("\nExcluded states:")
            print(excluded_counts.head(20))
        else:
            print("No states would be excluded with the given keywords.")


def main():
    """Main function to process both simulators with keyword and phase filtering."""
    
    print(" ENHANCED OUTLIER, STATE, AND PHASE FILTERING")
    print("=" * 60)
    
    # Configuration
    outlier_threshold = -10  # Remove R² values below -10
    simulators = ["sitl", "gazebo"]
    model_suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    
    # Define keyword patterns to exclude (you can modify these)
    exclude_keywords = [
        "*ATT_*",      # Attitude-related states
        "BARO_*",      # Barometer states
        "GPS_*",       # GPS states  
        "IMU_*",       # IMU states
        "*Error*"
    ]
    
    # Define phases to exclude (you can modify these)
    # Example: exclude phases with poor performance or specific flight conditions
    exclude_phases = [
        # 1,   # Takeoff phase - uncomment to exclude
        # 12,  # Landing phase - uncomment to exclude
        # 6,   # Specific maneuver phase - uncomment to exclude
    ]
    
    # Set to None to disable filtering, or use the lists above
    # exclude_keywords = None
    # exclude_phases = None
    
    print(f"Outlier threshold: R² < {outlier_threshold}")
    if exclude_keywords:
        print(f"Excluding states matching: {exclude_keywords}")
    else:
        print("No keyword-based state exclusion")
    
    if exclude_phases:
        print(f"Excluding phases: {exclude_phases}")
    else:
        print("No phase exclusion")
    
    # Process each simulator
    for sim in simulators:
        analysis_dir = f"./analysis_output_{sim}"
        
        if not os.path.exists(analysis_dir):
            print(f"  Directory not found: {analysis_dir}")
            continue
            
        print(f"\n Processing {sim.upper()} simulator...")
        
        # Show phase performance analysis first
        get_phase_performance_summary(analysis_dir, exclude_keywords)
        
        # Show state distribution analysis
        get_state_distribution(analysis_dir, exclude_keywords, exclude_phases=exclude_phases)
        
        # Process each model type
        for suffix in model_suffixes:
            regenerate_summary_from_filtered_data(analysis_dir, suffix, outlier_threshold, 
                                                exclude_keywords, exclude_phases=exclude_phases)
        
        # Regenerate comparison table
        print(f"\n Regenerating comparison table for {sim.upper()}...")
        regenerate_model_comparison_table(analysis_dir, 
                                        use_filtered=True, 
                                        use_excluded=bool(exclude_keywords),
                                        use_phase_exclusion=bool(exclude_phases))
    
    # Cross-simulator comparison
    print(f"\n Generating cross-simulator comparison...")
    
    cross_comparison_results = []
    simulator_configs = {
        "SITL": "./analysis_output_sitl",
        "Gazebo": "./analysis_output_gazebo"
    }
    
    for sim_name, analysis_output in simulator_configs.items():
        # Look for the most filtered version first
        output_suffix = ""
        if exclude_keywords and exclude_phases:
            output_suffix = "_filtered_excluded_phases"
        elif exclude_keywords:
            output_suffix = "_filtered_excluded"
        elif exclude_phases:
            output_suffix = "_filtered_phases"
        else:
            output_suffix = "_filtered"
        
        comparison_path = os.path.join(analysis_output, f"model_performance_comparison{output_suffix}.csv")
        
        if not os.path.exists(comparison_path):
            comparison_path = os.path.join(analysis_output, "model_performance_comparison.csv")
        
        if os.path.exists(comparison_path):
            df_sim_comparison = pd.read_csv(comparison_path)
            df_sim_comparison["Simulator"] = sim_name
            cross_comparison_results.append(df_sim_comparison)
    
    if cross_comparison_results:
        df_cross_comparison = pd.concat(cross_comparison_results, ignore_index=True)
        
        # Reorder columns
        cols = df_cross_comparison.columns.tolist()
        cols = ["Simulator"] + [col for col in cols if col != "Simulator"]
        df_cross_comparison = df_cross_comparison[cols]
        
        # Save cross-simulator comparison
        cross_output_dir = "./analysis_output_combined"
        os.makedirs(cross_output_dir, exist_ok=True)
        
        output_suffix = ""
        if exclude_keywords and exclude_phases:
            output_suffix = "_filtered_excluded_phases"
        elif exclude_keywords:
            output_suffix = "_filtered_excluded"
        elif exclude_phases:
            output_suffix = "_filtered_phases"
        else:
            output_suffix = "_filtered"
        
        output_path = os.path.join(cross_output_dir, f"cross_simulator_model_comparison{output_suffix}.csv")
        df_cross_comparison.to_csv(output_path, index=False)
        print(f" Cross-simulator comparison saved to: {output_path}")
        
        # Create paper-ready table
        paper_cross_table = df_cross_comparison[[
            "Simulator", "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", "Overall_Max_R2",
            "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7"
        ]].copy()
        
        paper_cross_table.columns = [
            "Simulator", "Model", "Total Models", "Mean R²", "Median R²", "Max R²",
            "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7"
        ]
        
        paper_cross_output_path = os.path.join(cross_output_dir, f"cross_simulator_paper_table{output_suffix}.csv")
        paper_cross_table.to_csv(paper_cross_output_path, index=False)
        print(f" Cross-simulator paper table saved to: {paper_cross_output_path}")
        
        # Print summary
        filter_types = []
        if exclude_keywords:
            filter_types.append("EXCLUDED STATES")
        if exclude_phases:
            filter_types.append("EXCLUDED PHASES")
        filter_types.append("FILTERED")
        filter_type = " + ".join(filter_types)
        
        print(f"\n" + "="*100)
        print(f"{filter_type} CROSS-SIMULATOR MODEL PERFORMANCE COMPARISON")
        print("="*100)
        print(paper_cross_table.to_string(index=False))
        print("="*100)
    
    print(f"\n ENHANCED FILTERING COMPLETE!")
    print(" Summary of what was created:")
    print("   • Phase performance analysis to guide phase exclusion decisions")
    print("   • *_filtered_excluded_phases.csv files with all filters applied")
    print("   • Updated comparison tables with realistic mean R² values")
    print("   • Cross-simulator comparison with filtered data")
    print("   • State and phase distribution analysis")
    
    # Provide guidance on customization
    print(f"\n CUSTOMIZATION TIPS:")
    print("   • Edit 'exclude_phases' list to exclude specific flight phases")
    print("   • Edit 'exclude_keywords' list to exclude specific sensor types")
    print("   • Adjust 'outlier_threshold' to change R² filtering sensitivity")
    print("   • Review phase_performance_analysis.csv to identify problematic phases")


if __name__ == "__main__":
    main()