#!/usr/bin/env python3
"""
Data Analysis Tool for RVSpec cFS Experiment Data
Author: CZ
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('DataAnalyzer')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        log_dir / f'processing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for simple logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class DataAnalyzer:
    """Main class for analyzing cFS experiment data"""
    
    def __init__(self, base_dir: Path = Path('<RVSPEC_ROOT>/CPG-cFS/DataAnalysis')):
        """Initialize the analyzer"""
        self.base_dir = base_dir
        self.results_dir = base_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        self.param_summary_path = Path('<RVSPEC_ROOT>/CPG-cFS/parameter_generation/parameter_output/parameter_summary.csv')
        self.experiment_data_dir = Path('<DATA_DIR>')
        
        # Setup logging
        self.logger = setup_logging(self.results_dir)
        self.logger.info("="*80)
        self.logger.info("DataAnalyzer initialized")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Results directory: {self.results_dir}")
        
        # Data containers
        self.param_df = None
        self.matched_configs = {}
        self.statistics_data = {}
        self.correlation_results = {}
        
    def load_parameter_summary(self) -> pd.DataFrame:
        """Load parameter summary file"""
        self.logger.info(f"Loading parameter summary from: {self.param_summary_path}")
        
        try:
            self.param_df = pd.read_csv(self.param_summary_path)
            # Extract factor columns (from MASS onwards)
            mass_col_idx = self.param_df.columns.get_loc('MASS')
            self.factor_columns = list(self.param_df.columns[mass_col_idx:])
            
            self.logger.info(f"Loaded {len(self.param_df)} parameter configurations")
            self.logger.info(f"Found {len(self.factor_columns)} factor columns")
            self.logger.debug(f"Factor columns: {self.factor_columns}")
            
            return self.param_df
            
        except Exception as e:
            self.logger.error(f"Failed to load parameter summary: {e}")
            raise
            
    def scan_and_match_experiments(self) -> Dict[str, Path]:
        """Scan experiment files and match with parameter configurations"""
        self.logger.info(f"Scanning experiment data directory: {self.experiment_data_dir}")
        
        if not self.experiment_data_dir.exists():
            self.logger.error(f"Experiment data directory does not exist: {self.experiment_data_dir}")
            raise FileNotFoundError(f"Directory not found: {self.experiment_data_dir}")
            
        # Get all CSV files
        csv_files = list(self.experiment_data_dir.glob("config_*_tlm.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files in experiment directory")
        
        # Extract config IDs from parameter summary
        param_config_ids = set(self.param_df['parameter_set_id'].values)
        
        # Match files with parameter configurations
        for csv_file in csv_files:
            # Extract config ID from filename (e.g., config_0000 from config_0000_2025_09_05_06_57_18_tlm.csv)
            filename = csv_file.name
            if filename.startswith('config_'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    config_id = f"config_{parts[1]}"
                    
                    if config_id in param_config_ids:
                        self.matched_configs[config_id] = csv_file
                        self.logger.debug(f"Matched: {config_id} -> {csv_file.name}")
                        
        self.logger.info(f"Successfully matched {len(self.matched_configs)} configurations")
        
        # Log unmatched configurations
        unmatched_params = param_config_ids - set(self.matched_configs.keys())
        if unmatched_params:
            self.logger.warning(f"Parameter configurations without matching data files: {sorted(unmatched_params)}")
            
        return self.matched_configs
        
    def process_experiment_file(self, config_id: str, file_path: Path) -> Optional[Dict]:
        """Process a single experiment file and calculate statistics"""
        self.logger.debug(f"Processing {config_id}: {file_path.name}")
        
        try:
            # Read CSV file without any type inference first
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check if file is completely empty
            if df.empty:
                self.logger.warning(f"Skipping {config_id}: File is empty")
                return None
            
            self.logger.debug(f"  File shape: {df.shape}")
            self.logger.debug(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                
            # Calculate statistics for each column
            stats = {}
            valid_columns = []
            skipped_columns = []
            
            for col in df.columns:
                try:
                    # First check if the column contains any non-numeric looking values
                    # Sample first few non-null values to check type
                    sample_values = df[col].dropna().head(10)
                    
                    if len(sample_values) == 0:
                        self.logger.debug(f"  Column '{col}' is all NaN, skipping")
                        skipped_columns.append((col, "all NaN"))
                        continue
                    
                    # Check if values look like boolean or strings
                    sample_str = sample_values.astype(str)
                    if any(val.lower() in ['true', 'false'] for val in sample_str):
                        self.logger.debug(f"  Column '{col}' contains boolean values, skipping")
                        skipped_columns.append((col, "boolean"))
                        continue
                    
                    # Try to convert to numeric
                    try:
                        numeric_col = pd.to_numeric(df[col], errors='coerce')
                    except Exception as conv_error:
                        self.logger.debug(f"  Column '{col}' cannot be converted to numeric: {conv_error}")
                        skipped_columns.append((col, "conversion failed"))
                        continue
                    
                    # Check if conversion resulted in all NaN (meaning all values were non-numeric)
                    valid_count = numeric_col.notna().sum()
                    total_count = len(df[col])
                    
                    if valid_count == 0:
                        self.logger.debug(f"  Column '{col}' has no numeric values after conversion, skipping")
                        skipped_columns.append((col, "no numeric values"))
                        continue
                    
                    # Log conversion success rate
                    conversion_rate = valid_count / total_count * 100
                    if conversion_rate < 50:
                        self.logger.debug(f"  Column '{col}' has low numeric conversion rate ({conversion_rate:.1f}%), skipping")
                        skipped_columns.append((col, f"low conversion rate: {conversion_rate:.1f}%"))
                        continue
                    
                    if valid_count < 2:  # Need at least 2 values for std
                        self.logger.debug(f"  Column '{col}' has less than 2 valid numeric values, skipping")
                        skipped_columns.append((col, "insufficient values"))
                        continue
                    
                    # Calculate mean and std using only numeric values
                    mean_val = float(numeric_col.mean(skipna=True))
                    std_val = float(numeric_col.std(skipna=True))
                    
                    # Skip columns with std = 0 (constant values)
                    if not np.isnan(std_val) and std_val > 0:
                        # Use the original column name with spaces
                        stats[f"{col}_mean"] = mean_val
                        stats[f"{col}_std"] = std_val
                        valid_columns.append(col)
                        self.logger.debug(f"   Column '{col}': mean={mean_val:.6g}, std={std_val:.6g}, valid={valid_count}/{total_count}")
                    else:
                        self.logger.debug(f"  Column '{col}' has std=0 or invalid std, skipping")
                        skipped_columns.append((col, "std=0 or invalid"))
                        
                except Exception as col_error:
                    self.logger.debug(f"  Error processing column '{col}': {str(col_error)[:100]}")
                    skipped_columns.append((col, f"error: {str(col_error)[:50]}"))
                    continue
            
            # Log summary
            if valid_columns:
                self.logger.info(f"   {config_id}: {len(valid_columns)} valid columns out of {len(df.columns)}")
                if len(skipped_columns) > 0:
                    self.logger.debug(f"    Skipped {len(skipped_columns)} columns")
                    # Log reasons for skipping (summarized)
                    skip_reasons = {}
                    for col, reason in skipped_columns:
                        reason_key = reason.split(':')[0] if ':' in reason else reason
                        skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
                    self.logger.debug(f"    Skip reasons: {skip_reasons}")
            else:
                self.logger.warning(f"   {config_id}: No valid numeric columns found")
                
            if not stats:
                return None
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error reading/processing {config_id}: {str(e)[:200]}", exc_info=True)
            return None
            
    def process_all_experiments(self) -> pd.DataFrame:
        """Process all matched experiment files"""
        self.logger.info("Processing all matched experiment files...")
        self.logger.info(f"Total files to process: {len(self.matched_configs)}")
        
        success_count = 0
        failed_count = 0
        
        for config_id, file_path in self.matched_configs.items():
            stats = self.process_experiment_file(config_id, file_path)
            if stats:
                self.statistics_data[config_id] = stats
                success_count += 1
            else:
                failed_count += 1
                
        self.logger.info(f"Processing complete: {success_count} successful, {failed_count} failed")
        self.logger.info(f"Successfully processed {len(self.statistics_data)} configurations")
        
        # Create combined dataframe
        if not self.statistics_data:
            self.logger.error("No valid data to process")
            self.logger.error("All experiment files failed processing - check data format")
            raise ValueError("No valid experiment data found")
            
        # Convert to DataFrame
        stats_df = pd.DataFrame.from_dict(self.statistics_data, orient='index')
        stats_df.index.name = 'parameter_set_id'
        stats_df = stats_df.reset_index()
        
        # Merge with parameter data
        combined_df = pd.merge(
            self.param_df[['parameter_set_id'] + self.factor_columns],
            stats_df,
            on='parameter_set_id',
            how='inner'
        )
        
        self.logger.info(f"Created combined dataset with shape: {combined_df.shape}")
        self.logger.info(f"Dataset has {len(combined_df)} samples and {len(combined_df.columns)} columns")
        return combined_df
        
    def calculate_correlations(self, combined_df: pd.DataFrame, top_n: int = 5) -> Dict:
        """Calculate correlations between factors and experiment statistics"""
        self.logger.info(f"Calculating correlations (Top {top_n} factors per metric)...")
        
        # Get statistic columns (means and stds)
        stat_columns = [col for col in combined_df.columns 
                       if col not in ['parameter_set_id'] + self.factor_columns]
        
        results = {}
        
        for stat_col in stat_columns:
            # Calculate correlation with each factor
            correlations = {}
            
            for factor in self.factor_columns:
                # Check if we have enough valid data
                valid_mask = combined_df[[factor, stat_col]].notna().all(axis=1)
                if valid_mask.sum() < 3:  # Need at least 3 points for correlation
                    continue
                    
                corr = combined_df[factor].corr(combined_df[stat_col])
                if pd.notna(corr):
                    correlations[factor] = corr
                    
            # Get top N factors by absolute correlation
            if correlations:
                sorted_corrs = sorted(correlations.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:top_n]
                results[stat_col] = sorted_corrs
                
                self.logger.debug(f"  {stat_col}: Top factor is {sorted_corrs[0][0]} (r={sorted_corrs[0][1]:.3f})")
                
        self.correlation_results = results
        return results
        
    def save_top_factors(self, results: Dict) -> None:
        """Save top factors analysis to file"""
        output_file = self.results_dir / 'top_factors_analysis.csv'
        self.logger.info(f"Saving top factors analysis to: {output_file}")
        
        # Convert to structured format
        rows = []
        for stat_col, top_factors in results.items():
            for rank, (factor, corr) in enumerate(top_factors, 1):
                rows.append({
                    'statistic': stat_col,
                    'rank': rank,
                    'factor': factor,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
                
        # Save as CSV
        top_factors_df = pd.DataFrame(rows)
        top_factors_df.to_csv(output_file, index=False)
        
        # Also save as JSON for easier parsing
        json_output = self.results_dir / 'top_factors_analysis.json'
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
            
        self.logger.info(f"Saved results to {output_file} and {json_output}")
        
    def generate_correlation_heatmap(self, combined_df: pd.DataFrame, threshold: float = 0.5) -> None:
        """Generate correlation heatmap for strong correlations"""
        self.logger.info(f"Generating correlation heatmap (threshold = {threshold})...")
        
        # Get statistic columns
        stat_columns = [col for col in combined_df.columns 
                       if col not in ['parameter_set_id'] + self.factor_columns]
        
        # Calculate full correlation matrix
        corr_matrix = pd.DataFrame(index=self.factor_columns, columns=stat_columns)
        
        for factor in self.factor_columns:
            for stat_col in stat_columns:
                corr = combined_df[factor].corr(combined_df[stat_col])
                corr_matrix.loc[factor, stat_col] = corr
                
        # Convert to numeric
        corr_matrix = corr_matrix.astype(float)
        
        # Filter for strong correlations
        strong_corr_mask = (corr_matrix.abs() > threshold)
        
        # Find factors and statistics with at least one strong correlation
        factors_with_strong = strong_corr_mask.any(axis=1)
        stats_with_strong = strong_corr_mask.any(axis=0)
        
        if not factors_with_strong.any() or not stats_with_strong.any():
            self.logger.warning(f"No correlations above threshold {threshold} found")
            return
            
        # Create filtered matrix
        filtered_matrix = corr_matrix.loc[factors_with_strong, stats_with_strong]
        
        self.logger.info(f"Heatmap dimensions: {filtered_matrix.shape}")
        
        # Create heatmap
        plt.figure(figsize=(max(12, len(filtered_matrix.columns) * 0.5), 
                          max(8, len(filtered_matrix.index) * 0.3)))
        
        sns.heatmap(filtered_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm', 
                   vmin=-1, vmax=1,
                   center=0,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'Correlation Heatmap (|r| > {threshold})', fontsize=14, fontweight='bold')
        plt.xlabel('Experiment Statistics', fontsize=12)
        plt.ylabel('Factors', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / f'correlation_heatmap_threshold_{threshold}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved heatmap to: {output_file}")
        
        # Save the correlation matrix as CSV too
        matrix_file = self.results_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(matrix_file)
        self.logger.info(f"Saved full correlation matrix to: {matrix_file}")
        
    def analyze_imu_data(self, combined_df: pd.DataFrame, top_n: int = 10) -> None:
        """Special analysis for IMU data (ANGULAR_RATE and LINEAR_ACCELERATION)"""
        self.logger.info("="*80)
        self.logger.info("Starting IMU-specific analysis...")
        
        # Get all statistic columns
        stat_columns = [col for col in combined_df.columns 
                       if col not in ['parameter_set_id'] + self.factor_columns]
        
        # Log all columns for debugging
        self.logger.debug(f"Total statistic columns: {len(stat_columns)}")
        self.logger.debug(f"Sample columns: {stat_columns[:10]}")
        
        # Find columns that match the specific IMU pattern
        # Looking for: GENERIC_IMU GENERIC_IMU_DATA_TLM [X/Y/Z]_[ANGULAR_RATE/LINEAR_ACCELERATION]_[mean/std]
        imu_columns = []
        
        for col in stat_columns:
            # Check if column contains GENERIC_IMU and either ANGULAR_RATE or LINEAR_ACCELERATION
            if 'GENERIC_IMU' in col and ('ANGULAR_RATE' in col or 'LINEAR_ACCELERATION' in col):
                imu_columns.append(col)
                self.logger.debug(f"Found IMU column: {col}")
        
        if not imu_columns:
            self.logger.warning("No IMU-related columns found in the data")
            self.logger.info("Looking for columns containing: 'GENERIC_IMU' and ('ANGULAR_RATE' or 'LINEAR_ACCELERATION')")
            # Show some actual column names for debugging
            sample_cols = [c for c in stat_columns if 'mean' in c or 'std' in c][:5]
            self.logger.info(f"Sample available columns: {sample_cols}")
            return
            
        self.logger.info(f"Found {len(imu_columns)} IMU-related statistics:")
        for col in imu_columns:
            self.logger.info(f"  - {col}")
        
        # Calculate correlations for IMU columns only
        imu_results = {}
        imu_corr_matrix = pd.DataFrame(index=self.factor_columns, columns=imu_columns)
        
        for imu_col in imu_columns:
            correlations = {}
            
            for factor in self.factor_columns:
                # Check if we have enough valid data
                valid_mask = combined_df[[factor, imu_col]].notna().all(axis=1)
                if valid_mask.sum() < 3:
                    continue
                    
                corr = combined_df[factor].corr(combined_df[imu_col])
                if pd.notna(corr):
                    # Store absolute correlation for ranking
                    correlations[factor] = abs(corr)
                    imu_corr_matrix.loc[factor, imu_col] = abs(corr)
                    
            # Get top N factors by absolute correlation
            if correlations:
                sorted_corrs = sorted(correlations.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:top_n]
                imu_results[imu_col] = sorted_corrs
                self.logger.debug(f"  {imu_col}: Top factor = {sorted_corrs[0][0]} (|r| = {sorted_corrs[0][1]:.4f})")
                
        if not imu_results:
            self.logger.warning("No valid IMU correlations found")
            return
            
        # Save IMU-specific top factors analysis
        self.save_imu_top_factors(imu_results)
        
        # Generate IMU-specific heatmap (only showing absolute correlations)
        self.generate_imu_heatmap(imu_corr_matrix)
        
        self.logger.info("IMU analysis completed successfully")
        
    def save_imu_top_factors(self, results: Dict) -> None:
        """Save IMU-specific top factors analysis"""
        output_file = self.results_dir / 'imu_top10_factors_analysis.csv'
        self.logger.info(f"Saving IMU top factors analysis to: {output_file}")
        
        # Convert to structured format
        rows = []
        for stat_col, top_factors in results.items():
            for rank, (factor, abs_corr) in enumerate(top_factors, 1):
                rows.append({
                    'imu_statistic': stat_col,
                    'rank': rank,
                    'factor': factor,
                    'abs_correlation': abs_corr,
                    'strength': 'Strong' if abs_corr > 0.7 else 'Moderate' if abs_corr > 0.4 else 'Weak'
                })
                
        if rows:
            # Save as CSV
            imu_factors_df = pd.DataFrame(rows)
            imu_factors_df.to_csv(output_file, index=False, float_format='%.4f')
            
            # Also save as JSON
            json_output = self.results_dir / 'imu_top10_factors_analysis.json'
            with open(json_output, 'w') as f:
                json.dump(results, f, indent=2, default=float)
                
            self.logger.info(f"Saved IMU analysis to {output_file} and {json_output}")
            
            # Print summary to log
            self.logger.info("\nIMU Analysis Summary:")
            for stat_col in results.keys():
                if results[stat_col]:
                    top_factor, top_corr = results[stat_col][0]
                    self.logger.info(f"  {stat_col}: Top factor = {top_factor} (|r| = {top_corr:.4f})")
        else:
            self.logger.warning("No IMU correlations found to save")
            
    def generate_imu_heatmap(self, corr_matrix: pd.DataFrame) -> None:
        """Generate heatmap specifically for IMU data (absolute correlations only)"""
        self.logger.info("Generating IMU correlation heatmap...")
        
        # Convert to numeric and take absolute values
        corr_matrix = corr_matrix.astype(float).abs()
        
        # Remove rows and columns with all NaN
        corr_matrix = corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        if corr_matrix.empty:
            self.logger.warning("No valid IMU correlations for heatmap")
            return
            
        # Find factors with at least one correlation > 0.3 (to reduce clutter)
        threshold = 0.3
        factors_to_show = corr_matrix.index[(corr_matrix > threshold).any(axis=1)]
        
        if len(factors_to_show) == 0:
            self.logger.warning(f"No IMU correlations above threshold {threshold}")
            return
            
        filtered_matrix = corr_matrix.loc[factors_to_show]
        
        # Create heatmap
        plt.figure(figsize=(max(14, len(filtered_matrix.columns) * 0.8), 
                          max(10, len(filtered_matrix.index) * 0.4)))
        
        # Use a custom colormap for absolute correlations (0 to 1)
        sns.heatmap(filtered_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',  # Yellow to Orange to Red for increasing correlation
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Absolute Correlation |r|'},
                   linewidths=0.5,
                   linecolor='gray')
        
        plt.title('IMU Data Correlation Heatmap (Absolute Values)\nANGULAR_RATE & LINEAR_ACCELERATION', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('IMU Statistics (mean/std)', fontsize=12)
        plt.ylabel('Factors', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        output_file = self.results_dir / 'imu_correlation_heatmap.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved IMU heatmap to: {output_file}")
        
        # Also save the correlation matrix as CSV
        matrix_file = self.results_dir / 'imu_correlation_matrix.csv'
        corr_matrix.to_csv(matrix_file, float_format='%.4f')
        self.logger.info(f"Saved IMU correlation matrix to: {matrix_file}")
        
    def generate_summary_report(self, combined_df: pd.DataFrame) -> None:
        """Generate a summary report of the analysis"""
        report_file = self.results_dir / 'analysis_summary.txt'
        self.logger.info(f"Generating summary report: {report_file}")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA ANALYSIS SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"1. DATA OVERVIEW\n")
            f.write(f"   - Parameter configurations loaded: {len(self.param_df)}\n")
            f.write(f"   - Experiment files matched: {len(self.matched_configs)}\n")
            f.write(f"   - Valid configurations processed: {len(self.statistics_data)}\n")
            f.write(f"   - Number of factors: {len(self.factor_columns)}\n")
            f.write(f"   - Number of statistics calculated: {len([c for c in combined_df.columns if c not in self.factor_columns + ['parameter_set_id']])}\n\n")
            
            f.write(f"2. TOP CORRELATIONS\n")
            for stat_col, top_factors in self.correlation_results.items():
                if top_factors:
                    top_factor, top_corr = top_factors[0]
                    f.write(f"   - {stat_col}:\n")
                    f.write(f"     Best factor: {top_factor} (r = {top_corr:.4f})\n")
                    
            f.write(f"\n3. FILES GENERATED\n")
            f.write(f"   - General Analysis:\n")
            f.write(f"     * Top factors analysis: top_factors_analysis.csv\n")
            f.write(f"     * Correlation heatmap: correlation_heatmap_threshold_*.png\n")
            f.write(f"     * Full correlation matrix: correlation_matrix.csv\n")
            f.write(f"   - IMU-Specific Analysis:\n")
            f.write(f"     * IMU top 10 factors: imu_top10_factors_analysis.csv\n")
            f.write(f"     * IMU correlation heatmap: imu_correlation_heatmap.png\n")
            f.write(f"     * IMU correlation matrix: imu_correlation_matrix.csv\n")
            f.write(f"   - Processing log: processing_log_*.log\n")
            
        self.logger.info("Summary report generated")
        
    def run(self, top_n: int = 5, heatmap_threshold: float = 0.5) -> None:
        """Run the complete analysis pipeline"""
        self.logger.info("Starting data analysis pipeline...")
        
        try:
            # Step 1: Load parameter summary
            self.load_parameter_summary()
            
            # Step 2: Scan and match experiment files
            self.scan_and_match_experiments()
            
            if not self.matched_configs:
                self.logger.error("No matched configurations found. Exiting.")
                return
                
            # Step 3: Process all experiments
            combined_df = self.process_all_experiments()
            
            # Step 4: Calculate correlations
            results = self.calculate_correlations(combined_df, top_n=top_n)
            
            # Step 5: Save results
            self.save_top_factors(results)
            
            # Step 6: Generate heatmap
            self.generate_correlation_heatmap(combined_df, threshold=heatmap_threshold)
            
            # Step 7: IMU-specific analysis (with top_n=10 for IMU)
            self.analyze_imu_data(combined_df, top_n=10)
            
            # Step 8: Generate summary report
            self.generate_summary_report(combined_df)
            
            self.logger.info("="*80)
            self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results saved to: {self.results_dir}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    try:
        # Run analysis
        analyzer.run(top_n=5, heatmap_threshold=0.5)
        return 0
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        print("Please check the processing log for details.")
        
        # Try to provide diagnostic information
        if hasattr(analyzer, 'matched_configs') and analyzer.matched_configs:
            print(f"\nDiagnostic: Found {len(analyzer.matched_configs)} matched files")
            # Sample one file to check format
            if analyzer.matched_configs:
                sample_config = list(analyzer.matched_configs.keys())[0]
                sample_file = analyzer.matched_configs[sample_config]
                print(f"Checking sample file: {sample_file.name}")
                
                try:
                    df_sample = pd.read_csv(sample_file, nrows=5)
                    print(f"Sample shape: {df_sample.shape}")
                    print(f"Sample columns (first 10): {list(df_sample.columns)[:10]}")
                    print(f"\nSample data types:")
                    for col in list(df_sample.columns)[:5]:
                        sample_val = df_sample[col].dropna().iloc[0] if not df_sample[col].dropna().empty else 'N/A'
                        print(f"  {col}: {type(sample_val).__name__} (example: {sample_val})")
                except Exception as diag_error:
                    print(f"Could not read sample file: {diag_error}")
        
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())