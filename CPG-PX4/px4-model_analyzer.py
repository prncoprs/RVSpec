"""
PX4 Drone Simulation Random Forest Model Analysis Module
=======================================================

This module trains Random Forest models to analyze relationships between
environmental factors and physical flight states for PX4 simulation data.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

# Import from preprocessor
from data_preprocessor import ENV_FACTORS, PX4_CONFIG


# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis configuration for PX4
PX4_ANALYSIS_CONFIG = {
    "analysis_output": Path("<DATA_DIR>/PGFuzzVMShared"),
    "filtered_data_path": Path("<DATA_DIR>/PGFuzzVMShared"),
    "models_dir": "px4_random_forest_models",
    "max_workers": 8
}

# Improved Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": 'sqrt',
    "random_state": 42,
    "n_jobs": -1,
    "bootstrap": True
}

# Cross-validation parameters
CV_FOLDS = 5

# Data quality thresholds
MIN_SAMPLES_THRESHOLD = 20  # Minimum samples required for training
VARIANCE_THRESHOLD = 1e-8   # Minimum variance for features
OUTLIER_Z_THRESHOLD = 5     # Z-score threshold for outlier detection


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_directory(output_dir):
    """Ensure the output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def extract_phase_from_column(column_name):
    """
    Extract phase number from column name.
    Expected format: {topic}_{column}_Phase{X}_{statistic}
    
    Returns:
        int: Phase number or None if not found
    """
    import re
    match = re.search(r'_Phase(\d+)_', column_name)
    return int(match.group(1)) if match else None


def get_base_column_name(column_name):
    """
    Extract base column name without phase and statistic suffixes.
    Expected format: {topic}_{column}_Phase{X}_{statistic}
    
    Returns:
        str: Base column name
    """
    import re
    # Remove _Phase{X}_{statistic} pattern
    base_name = re.sub(r'_Phase\d+_[a-z]+$', '', column_name)
    return base_name


def detect_and_remove_outliers(X, y, z_threshold=OUTLIER_Z_THRESHOLD):
    """
    Detect and remove outliers using Z-score method.
    
    Args:
        X: Feature matrix
        y: Target vector
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        X_clean, y_clean: Cleaned data without outliers
    """
    # Calculate Z-scores for target variable
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    
    # Keep samples below threshold
    mask = z_scores < z_threshold
    
    if mask.sum() < len(mask) * 0.8:  # Don't remove more than 20% of data
        print(f"    Warning: Would remove {(~mask).sum()} outliers ({(~mask).sum()/len(mask)*100:.1f}%), keeping all data")
        return X, y
    
    removed_count = (~mask).sum()
    if removed_count > 0:
        print(f"    Removed {removed_count} outliers ({removed_count/len(mask)*100:.1f}%)")
    
    return X[mask], y[mask]


def check_data_quality(X, y, feature_names):
    """
    Check data quality and return diagnostics.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        dict: Data quality metrics
    """
    diagnostics = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "target_mean": np.mean(y),
        "target_std": np.std(y),
        "target_min": np.min(y),
        "target_max": np.max(y),
        "has_inf": np.any(np.isinf(X)) or np.any(np.isinf(y)),
        "has_nan": np.any(np.isnan(X)) or np.any(np.isnan(y)),
        "zero_variance_features": []
    }
    
    # Check for zero variance features
    for i, feature in enumerate(feature_names):
        if np.var(X[:, i]) < VARIANCE_THRESHOLD:
            diagnostics["zero_variance_features"].append(feature)
    
    return diagnostics


def save_model_multiple_formats(pipeline, model_info, model_path_base):
    """
    Save model in multiple formats for compatibility with different Python versions.
    
    Args:
        pipeline: Trained sklearn pipeline
        model_info: Dictionary with model metadata
        model_path_base: Base path without extension
        
    Returns:
        dict: Paths to saved model files
    """
    saved_paths = {}
    
    try:
        # 1. Modern joblib format (Python 3 only)
        joblib_path = f"{model_path_base}.joblib"
        joblib.dump({
            "pipeline": pipeline,
            **model_info
        }, joblib_path)
        saved_paths["joblib"] = joblib_path
        
        # 2. Legacy pickle format (Python 2 compatible)
        pickle_path = f"{model_path_base}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                "pipeline": pipeline,
                **model_info
            }, f, protocol=2)  # Protocol 2 is compatible with Python 2
        saved_paths["pickle"] = pickle_path
        
        # 3. Extract Random Forest model directly
        rf_model = pipeline.named_steps['rf']
        rf_pickle_path = f"{model_path_base}_rf_only.pkl"
        with open(rf_pickle_path, 'wb') as f:
            pickle.dump(rf_model, f, protocol=2)
        saved_paths["rf_pickle"] = rf_pickle_path
        
        # 4. Save model parameters as JSON (for manual reconstruction)
        json_path = f"{model_path_base}_params.json"
        
        # Extract preprocessing info
        try:
            variance_selector = pipeline.named_steps['variance_threshold']
            selected_features = variance_selector.get_support().tolist()
            scaler = pipeline.named_steps['scaler']
            scaler_center = scaler.center_.tolist()
            scaler_scale = scaler.scale_.tolist()
        except:
            selected_features = [True] * len(model_info["env_factors"])
            scaler_center = [0.0] * len(model_info["env_factors"])
            scaler_scale = [1.0] * len(model_info["env_factors"])
        
        # Create JSON-serializable model info
        json_data = {
            "model_type": "RandomForest",
            "sklearn_version": "compatible",
            "env_factors": model_info["env_factors"],
            "physical_state": model_info["physical_state"],
            "phase": model_info["phase"],
            "rf_params": model_info["rf_params"],
            
            # Preprocessing info
            "selected_features": selected_features,
            "scaler_center": scaler_center,
            "scaler_scale": scaler_scale,
            
            # Model structure
            "n_estimators": rf_model.n_estimators,
            "max_depth": rf_model.max_depth,
            "min_samples_split": rf_model.min_samples_split,
            "min_samples_leaf": rf_model.min_samples_leaf,
            "max_features": rf_model.max_features,
            "random_state": rf_model.random_state,
            
            # Feature importance
            "feature_importances": rf_model.feature_importances_.tolist(),
            
            # Instructions for Python 2
            "python2_usage": {
                "load_command": f"import pickle; model = pickle.load(open('{rf_pickle_path}', 'rb'))",
                "predict_steps": [
                    "1. Load model using pickle",
                    "2. Apply feature selection using 'selected_features'",
                    "3. Apply scaling: (X - scaler_center) / scaler_scale",
                    "4. Use model.predict(X_processed)"
                ]
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_paths["json"] = json_path
        
        # 5. Save a simple Python 2 compatible prediction function
        py2_script_path = f"{model_path_base}_predict_py2.py"
        create_python2_prediction_script(json_data, py2_script_path)
        saved_paths["py2_script"] = py2_script_path
        
    except Exception as e:
        print(f"    Warning: Error saving some model formats: {e}")
    
    return saved_paths


def create_python2_prediction_script(model_info, script_path):
    """
    Create a Python 2 compatible prediction script.
    
    Args:
        model_info: Model information dictionary
        script_path: Path to save the Python script
    """
    script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 2 Compatible Prediction Script for PX4 Random Forest Model
Physical State: {model_info["physical_state"]}
Phase: {model_info["phase"]}

Generated automatically by PX4 Model Analyzer
"""

import pickle
import numpy as np

class PX4RandomForestPredictor(object):
    """Python 2 compatible Random Forest predictor."""
    
    def __init__(self, model_path):
        """Load the Random Forest model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Model metadata
        self.env_factors = {model_info["env_factors"]}
        self.selected_features = {model_info["selected_features"]}
        self.scaler_center = np.array({model_info["scaler_center"]})
        self.scaler_scale = np.array({model_info["scaler_scale"]})
        self.physical_state = "{model_info["physical_state"]}"
        self.phase = {model_info["phase"]}
    
    def preprocess_features(self, X):
        """Apply preprocessing steps."""
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply feature selection
        if len(self.selected_features) == X.shape[1]:
            X_selected = X[:, [i for i, selected in enumerate(self.selected_features) if selected]]
        else:
            X_selected = X  # Skip if dimensions don't match
        
        # Apply robust scaling
        X_scaled = (X_selected - self.scaler_center) / self.scaler_scale
        
        return X_scaled
    
    def predict(self, X):
        """Make predictions."""
        X_processed = self.preprocess_features(X)
        return self.model.predict(X_processed)
    
    def predict_single(self, env_factor_values):
        """Predict for a single set of environmental factor values."""
        if len(env_factor_values) != len(self.env_factors):
            raise ValueError("Input must have {{}} values for: {{}}".format(
                len(self.env_factors), self.env_factors))
        
        X = np.array([env_factor_values])
        return self.predict(X)[0]


# Example usage:
if __name__ == "__main__":
    # Load model
    model_path = "{script_path.replace('_predict_py2.py', '_rf_only.pkl')}"
    predictor = PX4RandomForestPredictor(model_path)
    
    # Example prediction (replace with actual environmental factor values)
    # env_values = [0.1, 0.2, 0.3, ...]  # Must match env_factors order
    # prediction = predictor.predict_single(env_values)
    # print("Predicted {{}}:".format(predictor.physical_state), prediction)
    
    print("Model loaded successfully!")
    print("Physical State:", predictor.physical_state)
    print("Phase:", predictor.phase)
    print("Environmental Factors:", len(predictor.env_factors))
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    try:
        import os
        os.chmod(script_path, 0o755)
    except:
        pass


# ============================================================================
# IMPROVED RANDOM FOREST MODEL TRAINER
# ============================================================================

class ImprovedRandomForestTrainer:
    """Enhanced trainer for Random Forest models with data preprocessing."""
    
    def __init__(self, env_factors, output_dir, rf_params=None):
        self.env_factors = env_factors
        self.output_dir = output_dir
        self.rf_params = rf_params or RF_PARAMS
        ensure_output_directory(output_dir)
    
    def prepare_data(self, df_filtered, phys_state):
        """Prepare and clean data for model training."""
        required_columns = self.env_factors + [phys_state]
        df_valid = df_filtered[required_columns].dropna()
        
        if df_valid.empty:
            return None, None, None
        
        X = df_valid[self.env_factors].values
        y = df_valid[phys_state].values
        
        # Check minimum sample requirement
        if len(X) < MIN_SAMPLES_THRESHOLD:
            print(f"    Insufficient samples: {len(X)} < {MIN_SAMPLES_THRESHOLD}")
            return None, None, None
        
        # Data quality check
        diagnostics = check_data_quality(X, y, self.env_factors)
        
        # Handle infinite or NaN values
        if diagnostics["has_inf"] or diagnostics["has_nan"]:
            print(f"    Warning: Found inf/nan values, removing...")
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            
            if len(X) < MIN_SAMPLES_THRESHOLD:
                return None, None, None
        
        # Remove outliers
        X, y = detect_and_remove_outliers(X, y)
        
        if len(X) < MIN_SAMPLES_THRESHOLD:
            return None, None, None
        
        return X, y, diagnostics
    
    def create_preprocessing_pipeline(self, X):
        """Create preprocessing pipeline based on data characteristics."""
        steps = []
        
        # Remove zero-variance features
        variance_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        steps.append(('variance_threshold', variance_selector))
        
        # Scale features - use RobustScaler to handle outliers better
        scaler = RobustScaler()
        steps.append(('scaler', scaler))
        
        return steps
    
    def evaluate_model(self, pipeline, X, y, cv=CV_FOLDS):
        """Evaluate model using cross-validation with error handling."""
        try:
            # Use negative mean squared error as backup if R² fails
            r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
            
            # Check for reasonable R² scores
            if np.any(r2_scores < -100):  # Extremely negative R² indicates problems
                # Try with RMSE instead
                neg_mse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
                mse_scores = -neg_mse_scores
                
                # Convert to R² equivalent (pseudo R²)
                y_var = np.var(y)
                pseudo_r2_scores = 1 - (mse_scores / y_var)
                return pseudo_r2_scores.mean(), pseudo_r2_scores.std()
            
            return r2_scores.mean(), r2_scores.std()
            
        except Exception as e:
            print(f"    Error in cross-validation: {e}")
            return -1.0, 0.0
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train an improved Random Forest model for a specific physical state and phase."""
        result = self.prepare_data(df_filtered, phys_state)
        if result[0] is None:
            return None
        
        X, y, diagnostics = result
        
        # Create preprocessing pipeline
        preprocessing_steps = self.create_preprocessing_pipeline(X)
        
        # Create full pipeline with Random Forest
        pipeline_steps = preprocessing_steps + [('rf', RandomForestRegressor(**self.rf_params))]
        pipeline = Pipeline(pipeline_steps)
        
        # Cross-validation evaluation
        r2_mean, r2_std = self.evaluate_model(pipeline, X, y)
        
        # Skip models with very poor performance
        if r2_mean < -10:  # Extremely poor models
            print(f"    Skipping {phys_state}: R² = {r2_mean:.3f} (too poor)")
            return None
        
        # Fit pipeline on full data
        try:
            pipeline.fit(X, y)
        except Exception as e:
            print(f"    Error fitting model for {phys_state}: {e}")
            return None
        
        # Save model in multiple formats
        model_dir = self.output_dir / f"phase_{phase}"
        ensure_output_directory(model_dir)
        
        model_path_base = model_dir / f"{phys_state}"
        
        # Model metadata
        model_info = {
            "env_factors": self.env_factors,
            "rf_params": self.rf_params,
            "physical_state": phys_state,
            "phase": phase,
            "diagnostics": diagnostics
        }
        
        # Save in multiple formats for compatibility
        saved_paths = save_model_multiple_formats(pipeline, model_info, str(model_path_base))
        
        # Get feature importance from the trained Random Forest
        rf_model = pipeline.named_steps['rf']
        
        # Get feature importance (accounting for preprocessing)
        try:
            # Get the features that survived preprocessing
            variance_selector = pipeline.named_steps['variance_threshold']
            selected_features = variance_selector.get_support()
            selected_env_factors = [self.env_factors[i] for i in range(len(self.env_factors)) if selected_features[i]]
            
            feature_importance = rf_model.feature_importances_
        except:
            selected_env_factors = self.env_factors
            feature_importance = rf_model.feature_importances_
        
        return {
            "physical_state": phys_state,
            "phase": phase,
            "r2_cv_mean": r2_mean,
            "r2_cv_std": r2_std,
            "model_paths": saved_paths,
            "feature_importance": feature_importance.tolist(),
            "selected_features": selected_env_factors,
            "n_samples": len(X),
            "target_mean": diagnostics["target_mean"],
            "target_std": diagnostics["target_std"],
            "outliers_removed": diagnostics["n_samples"] - len(X)
        }


# ============================================================================
# MODEL TRAINING PIPELINE
# ============================================================================

def process_single_state(args):
    """Process a single physical state with improved Random Forest trainer."""
    trainer, df_filtered, phys_state, phase = args
    try:
        return trainer.train_model(df_filtered, phys_state, phase)
    except Exception as e:
        print(f"    Error processing {phys_state}: {e}")
        return None


def train_random_forest_models(df_filtered, output_dir):
    """
    Train improved Random Forest models for all physical states across all phases.
    
    Args:
        df_filtered: Filtered DataFrame with environmental factors and physical states
        output_dir: Directory to save models and results
        
    Returns:
        pd.DataFrame: Summary of all trained models
    """
    print("Starting improved Random Forest model training...")
    
    # Data preprocessing and quality check
    print(f"Initial data shape: {df_filtered.shape}")
    
    # Remove any remaining NaN/inf values in environmental factors
    env_factor_mask = df_filtered[ENV_FACTORS].notna().all(axis=1)
    env_factor_mask &= np.isfinite(df_filtered[ENV_FACTORS]).all(axis=1)
    df_cleaned = df_filtered[env_factor_mask].copy()
    
    print(f"After cleaning environmental factors: {df_cleaned.shape}")
    
    if df_cleaned.empty:
        print(" No valid data after cleaning environmental factors")
        return pd.DataFrame()
    
    # Group physical states by phase
    physical_states_by_phase = {}
    
    for col in df_cleaned.columns:
        if col not in ENV_FACTORS and col != 'exp_id':
            phase = extract_phase_from_column(col)
            if phase is not None:
                # Additional check: ensure this physical state has sufficient variation
                if df_cleaned[col].notna().sum() >= MIN_SAMPLES_THRESHOLD:
                    non_na_values = df_cleaned[col].dropna()
                    if len(non_na_values) > 0 and non_na_values.std() > VARIANCE_THRESHOLD:
                        if phase not in physical_states_by_phase:
                            physical_states_by_phase[phase] = []
                        physical_states_by_phase[phase].append(col)
    
    print(f"Found physical states for phases: {sorted(physical_states_by_phase.keys())}")
    for phase, states in physical_states_by_phase.items():
        print(f"  Phase {phase}: {len(states)} valid physical states")
    
    # Initialize improved trainer
    models_dir = output_dir / PX4_ANALYSIS_CONFIG["models_dir"]
    trainer = ImprovedRandomForestTrainer(ENV_FACTORS, models_dir)
    
    all_results = []
    total_models = sum(len(states) for states in physical_states_by_phase.values())
    processed_count = 0
    
    print(f"Training {total_models} improved Random Forest models...")
    
    # Process each phase
    for phase in sorted(physical_states_by_phase.keys()):
        phys_states = physical_states_by_phase[phase]
        print(f"\nPhase {phase}: Training {len(phys_states)} models")
        
        phase_results = []
        phase_r2_values = []
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=PX4_ANALYSIS_CONFIG["max_workers"]) as executor:
            # Submit all jobs for this phase
            args_list = [(trainer, df_cleaned, state, phase) for state in phys_states]
            futures = [executor.submit(process_single_state, args) for args in args_list]
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    phase_results.append(result)
                    phase_r2_values.append(result["r2_cv_mean"])
                
                processed_count += 1
                
                # Progress update
                if processed_count % 20 == 0 or processed_count == total_models:
                    progress_pct = (processed_count / total_models) * 100
                    print(f"  Progress: {processed_count}/{total_models} ({progress_pct:.1f}%)")
        
        # Phase summary
        if phase_results:
            # Filter out extremely poor models for statistics
            valid_r2_values = [r2 for r2 in phase_r2_values if r2 > -10]
            
            if valid_r2_values:
                phase_mean_r2 = np.mean(valid_r2_values)
                phase_median_r2 = np.median(valid_r2_values)
                phase_high_r2_count = sum(1 for r2 in valid_r2_values if r2 >= 0.8)
                phase_good_r2_count = sum(1 for r2 in valid_r2_values if r2 >= 0.5)
                
                print(f"   Phase {phase} complete:")
                print(f"    - Models trained: {len(phase_results)} ({len(valid_r2_values)} with R² > -10)")
                print(f"    - Mean R²: {phase_mean_r2:.3f}")
                print(f"    - Median R²: {phase_median_r2:.3f}")
                print(f"    - Models with R² ≥ 0.8: {phase_high_r2_count}")
                print(f"    - Models with R² ≥ 0.5: {phase_good_r2_count}")
            else:
                print(f"   Phase {phase}: No models with reasonable performance")
            
            all_results.extend(phase_results)
        else:
            print(f"   Phase {phase}: No models trained successfully")
    
    print(f"\n Training complete: {len(all_results)}/{total_models} models trained successfully")
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def analyze_model_performance(results_df, output_dir):
    """
    Analyze and summarize improved Random Forest model performance.
    
    Args:
        results_df: DataFrame with model training results
        output_dir: Directory to save analysis results
    """
    if results_df.empty:
        print("No model results to analyze.")
        return
    
    print("\n" + "="*60)
    print("IMPROVED RANDOM FOREST MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Filter out extremely poor models for main statistics
    reasonable_results = results_df[results_df['r2_cv_mean'] > -10].copy()
    poor_models = len(results_df) - len(reasonable_results)
    
    if poor_models > 0:
        print(f"Note: Excluded {poor_models} models with extremely poor performance (R² < -10)")
    
    if reasonable_results.empty:
        print(" All models had extremely poor performance.")
        return
    
    # Overall statistics
    total_models = len(reasonable_results)
    mean_r2 = reasonable_results['r2_cv_mean'].mean()
    median_r2 = reasonable_results['r2_cv_mean'].median()
    std_r2 = reasonable_results['r2_cv_mean'].std()
    min_r2 = reasonable_results['r2_cv_mean'].min()
    max_r2 = reasonable_results['r2_cv_mean'].max()
    
    # Performance thresholds
    excellent_models = (reasonable_results['r2_cv_mean'] >= 0.9).sum()
    good_models = (reasonable_results['r2_cv_mean'] >= 0.8).sum()
    fair_models = (reasonable_results['r2_cv_mean'] >= 0.7).sum()
    decent_models = (reasonable_results['r2_cv_mean'] >= 0.5).sum()
    poor_models_filtered = (reasonable_results['r2_cv_mean'] < 0.2).sum()
    
    print(f"Overall Performance Summary (Reasonable Models Only):")
    print(f"  Total Models Analyzed: {total_models}")
    print(f"  Mean R² Score: {mean_r2:.4f}")
    print(f"  Median R² Score: {median_r2:.4f}")
    print(f"  Std R² Score: {std_r2:.4f}")
    print(f"  Min R² Score: {min_r2:.4f}")
    print(f"  Max R² Score: {max_r2:.4f}")
    print()
    print(f"Performance Categories:")
    print(f"  Excellent (R² ≥ 0.9): {excellent_models} ({excellent_models/total_models*100:.1f}%)")
    print(f"  Good (R² ≥ 0.8): {good_models} ({good_models/total_models*100:.1f}%)")
    print(f"  Fair (R² ≥ 0.7): {fair_models} ({fair_models/total_models*100:.1f}%)")
    print(f"  Decent (R² ≥ 0.5): {decent_models} ({decent_models/total_models*100:.1f}%)")
    print(f"  Poor (R² < 0.2): {poor_models_filtered} ({poor_models_filtered/total_models*100:.1f}%)")
    
    # Phase-wise analysis
    print(f"\nPhase-wise Performance:")
    phase_summary = []
    
    for phase in sorted(reasonable_results['phase'].unique()):
        phase_df = reasonable_results[reasonable_results['phase'] == phase]
        phase_mean_r2 = phase_df['r2_cv_mean'].mean()
        phase_median_r2 = phase_df['r2_cv_mean'].median()
        phase_count = len(phase_df)
        phase_good = (phase_df['r2_cv_mean'] >= 0.8).sum()
        phase_decent = (phase_df['r2_cv_mean'] >= 0.5).sum()
        
        print(f"  Phase {phase}: {phase_count} models, Mean R²: {phase_mean_r2:.3f}, "
              f"Median R²: {phase_median_r2:.3f}, Good: {phase_good}, Decent: {phase_decent}")
        
        phase_summary.append({
            'phase': phase,
            'model_count': phase_count,
            'mean_r2': phase_mean_r2,
            'median_r2': phase_median_r2,
            'good_models_count': phase_good,
            'decent_models_count': phase_decent,
            'good_models_pct': phase_good / phase_count * 100,
            'decent_models_pct': phase_decent / phase_count * 100
        })
    
    # Save detailed results (all models including poor ones)
    results_path = output_dir / "improved_random_forest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n Detailed results saved to: {results_path}")
    
    # Save phase summary
    phase_summary_df = pd.DataFrame(phase_summary)
    phase_summary_path = output_dir / "improved_random_forest_phase_summary.csv"
    phase_summary_df.to_csv(phase_summary_path, index=False)
    print(f" Phase summary saved to: {phase_summary_path}")
    
    # Top performing models
    top_models = reasonable_results.nlargest(15, 'r2_cv_mean')[
        ['physical_state', 'phase', 'r2_cv_mean', 'r2_cv_std', 'n_samples', 'target_std']
    ]
    print(f"\nTop 15 Performing Models:")
    print(top_models.to_string(index=False))
    
    # Data quality insights
    print(f"\nData Quality Insights:")
    avg_samples = reasonable_results['n_samples'].mean()
    avg_target_std = reasonable_results['target_std'].mean()
    print(f"  Average samples per model: {avg_samples:.1f}")
    print(f"  Average target std deviation: {avg_target_std:.4f}")
    
    # Save feature importance analysis
    analyze_feature_importance(reasonable_results, output_dir)
    
    print("="*60)


def analyze_feature_importance(results_df, output_dir):
    """
    Analyze feature importance across all Random Forest models.
    
    Args:
        results_df: DataFrame with model training results
        output_dir: Directory to save analysis results
    """
    print(f"\nFeature Importance Analysis:")
    
    # Aggregate feature importance across all models
    importance_data = []
    
    for _, row in results_df.iterrows():
        if 'feature_importance' in row and row['feature_importance']:
            for i, importance in enumerate(row['feature_importance']):
                if i < len(ENV_FACTORS):  # Safety check
                    importance_data.append({
                        'environmental_factor': ENV_FACTORS[i],
                        'importance': importance,
                        'phase': row['phase'],
                        'r2_score': row['r2_cv_mean']
                    })
    
    if not importance_data:
        print("  No feature importance data available.")
        return
    
    importance_df = pd.DataFrame(importance_data)
    
    # Overall feature importance (mean across all models)
    overall_importance = importance_df.groupby('environmental_factor')['importance'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    print(f"  Top 10 Most Important Environmental Factors (Overall):")
    top_features = overall_importance.head(10)
    for factor, stats in top_features.iterrows():
        print(f"    {factor}: {stats['mean']:.4f} ± {stats['std']:.4f} (from {stats['count']} models)")
    
    # Phase-wise feature importance
    phase_importance = importance_df.groupby(['phase', 'environmental_factor'])['importance'].mean().unstack(fill_value=0)
    
    # Save feature importance results
    overall_importance_path = output_dir / "feature_importance_overall.csv"
    overall_importance.to_csv(overall_importance_path)
    
    phase_importance_path = output_dir / "feature_importance_by_phase.csv"
    phase_importance.to_csv(phase_importance_path)
    
    print(f"   Feature importance saved to: {overall_importance_path}")
    print(f"   Phase-wise importance saved to: {phase_importance_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to train Random Forest models for PX4 simulation data.
    """
    print("PX4 Random Forest Model Training Pipeline")
    print("="*60)
    
    # Check if filtered data exists
    filtered_data_path = PX4_ANALYSIS_CONFIG["filtered_data_path"]
    if not filtered_data_path.exists():
        print(f" Filtered data not found: {filtered_data_path}")
        print("Please run data preprocessing first:")
        print("python data_preprocessor.py")
        return False
    
    # Load filtered data
    print(f"Loading filtered data from: {filtered_data_path}")
    try:
        df_filtered = pd.read_csv(filtered_data_path)
        print(f" Data loaded successfully: {df_filtered.shape}")
    except Exception as e:
        print(f" Error loading data: {e}")
        return False
    
    # Verify environmental factors are present
    missing_env_factors = [factor for factor in ENV_FACTORS if factor not in df_filtered.columns]
    if missing_env_factors:
        print(f"  Warning: Missing environmental factors: {missing_env_factors}")
    
    present_env_factors = [factor for factor in ENV_FACTORS if factor in df_filtered.columns]
    print(f" Found {len(present_env_factors)} environmental factors")
    
    # Count physical states
    physical_state_columns = [col for col in df_filtered.columns 
                            if col not in ENV_FACTORS and col != 'exp_id']
    print(f" Found {len(physical_state_columns)} physical state columns")
    
    # Set up output directory
    output_dir = PX4_ANALYSIS_CONFIG["analysis_output"]
    ensure_output_directory(output_dir)
    
    # Train Random Forest models
    print(f"\nStarting Random Forest model training...")
    results_df = train_random_forest_models(df_filtered, output_dir)
    
    if results_df.empty:
        print(" No models were trained successfully.")
        return False
    
    # Analyze performance
    analyze_model_performance(results_df, output_dir)
    
    # Final summary
    print(f"\n PX4 Random Forest analysis complete!")
    print(f"   Models saved to: {output_dir / PX4_ANALYSIS_CONFIG['models_dir']}")
    print(f"   Results saved to: {output_dir}")
    print(f"   Total models trained: {len(results_df)}")
    print(f"   Mean performance: R² = {results_df['r2_cv_mean'].mean():.3f}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)