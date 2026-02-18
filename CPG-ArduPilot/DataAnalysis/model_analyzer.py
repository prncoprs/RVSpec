"""
Drone Simulation Model Analysis Module
=====================================

This module handles machine learning model training, evaluation, and comparison
for analyzing relationships between environmental factors and physical flight states.
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import warnings

# Suppress Ridge regression warnings about singular matrices
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.linear_model._ridge")
# Suppress Gaussian Process convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process")
warnings.filterwarnings("ignore", message=".*optimal value found.*upper bound.*")

# Machine Learning Libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from xgboost import XGBRegressor
import lightgbm as lgb

# Import from preprocessor
from data_preprocessor import ENV_FACTORS, preprocess_dataframe, check_file_exists


# ============================================================================
# CONFIGURATION
# ============================================================================

N_TOP = 200  # Number of top correlated factors
REPLICATION_COLUMN = "Rep"

# Simulator configurations for dual-simulator support
SIMULATOR_CONFIGS = {
    "SITL": {
        "data_root": "./data_sitl",
        "analysis_output": "./analysis_output_sitl",
        "display_name": "SITL Simulator"
    },
    "Gazebo": {
        "data_root": "./data_gazebo",
        "analysis_output": "./analysis_output_gazebo",
        "display_name": "Gazebo Simulator"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_directory(output_dir):
    """Ensure the analysis output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def configure_simulator_paths(sitl_data_path=None, gazebo_data_path=None):
    """Configure the data paths for both simulators."""
    if sitl_data_path:
        SIMULATOR_CONFIGS["SITL"]["data_root"] = sitl_data_path
        print(f" SITL data path set to: {sitl_data_path}")
    
    if gazebo_data_path:
        SIMULATOR_CONFIGS["Gazebo"]["data_root"] = gazebo_data_path
        print(f" Gazebo data path set to: {gazebo_data_path}")
    
    # Verify paths exist
    for sim_name, config in SIMULATOR_CONFIGS.items():
        if not os.path.exists(config["data_root"]):
            print(f"  Warning: {sim_name} data path does not exist: {config['data_root']}")
        else:
            print(f" {sim_name} data path verified: {config['data_root']}")


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_top_correlations(df, env_factor, n=N_TOP):
    """
    Compute the top `n` correlated non-environmental factors for a given environmental factor.
    """
    if env_factor not in df.columns:
        print(f"Warning: {env_factor} not found in dataframe. Skipping...")
        return None

    corr_matrix = df.corr(numeric_only=True)
    excluded = set(ENV_FACTORS)
    correlation_values = corr_matrix[env_factor].drop(labels=excluded, errors="ignore").abs()
    return correlation_values.nlargest(n)


def analyze_top_correlations(df, env_factors=ENV_FACTORS, n=N_TOP, output_path="Top_Correlated_Factors.csv"):
    """Analyze the top N correlated factors for each environmental factor."""
    df_preprocessed = preprocess_dataframe(df)
    
    top_correlations = {}
    for env_factor in env_factors:
        top_n_correlations = compute_top_correlations(df_preprocessed, env_factor, n)
        if top_n_correlations is not None:
            top_correlations[env_factor] = top_n_correlations
            print(f"\nTop {n} most correlated factors with {env_factor}:")
            print(top_n_correlations.head(10))  # Show top 10

    # Save results
    df_top_correlations = pd.DataFrame.from_dict(top_correlations, orient='index').T
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_top_correlations.to_csv(output_path, index=True)
    print(f"Top correlated factors saved to {output_path}")


# ============================================================================
# MACHINE LEARNING MODEL CLASSES
# ============================================================================

class ModelTrainer:
    """Base class for training different types of models."""
    
    def __init__(self, env_factors, output_dir):
        self.env_factors = env_factors
        self.output_dir = output_dir
        
    def prepare_data(self, df_filtered, phys_state):
        """Prepare data for model training."""
        df_valid = df_filtered[self.env_factors + [phys_state]].dropna()
        if df_valid.empty:
            return None, None
        
        X = df_valid[self.env_factors].values
        y = df_valid[phys_state].values
        return X, y
    
    def evaluate_model(self, model, X, y, cv=3):
        """Evaluate model using cross-validation."""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            return scores.mean()
        except Exception as e:
            return -1.0  # Return poor score if CV fails


class PolynomialRidgeTrainer(ModelTrainer):
    """Trainer for Polynomial + Ridge regression models with fixed degree."""
    
    def __init__(self, env_factors, output_dir, degree=3):
        super().__init__(env_factors, output_dir)
        self.degree = degree
        
    def train_model(self, df_filtered, phys_state, phase):
        """Train a polynomial ridge model with fixed degree for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Use fixed degree (no automatic selection)
        pipeline = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            Ridge(alpha=1.0)
        )

        # Cross-validation to get average R²
        r2_cv_mean = self.evaluate_model(pipeline, X, y)

        # Refit model on full data
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = Ridge(alpha=1.0).fit(X_poly, y)

        # Save model and transformer
        model_dir = os.path.join(self.output_dir, f"models_phase{phase}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": model,
            "poly": poly,
            "env_factors": self.env_factors,
            "degree": self.degree
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path,
            "degree": self.degree,
            **{f"coef_{i}": v for i, v in enumerate(model.coef_)}
        }


class GaussianProcessTrainer(ModelTrainer):
    """Trainer for Gaussian Process Regression models."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train a Gaussian Process model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Define kernel and model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(X.shape[1]), 
            length_scale_bounds=(1e-2, 1e2)
        )
        model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=1e-2, 
            normalize_y=True
        )

        # Cross-validation for generalization performance
        r2_cv_mean = self.evaluate_model(model, X, y)

        # Refit on full data
        model.fit(X, y)

        # Save model
        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_gpr")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": model,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path
        }


class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost models - represents gradient boosting methods."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train an XGBoost model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        model = XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            verbosity=0, 
            random_state=0
        )
        
        r2_cv_mean = self.evaluate_model(model, X, y)
        model.fit(X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_xgb")
        os.makedirs(model_dir, exist_ok=True)

        # Save in JSON format for cross-version compatibility
        model_path = os.path.join(model_dir, f"{phys_state}.json")
        model.get_booster().save_model(model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path,
            "feature_importance": model.feature_importances_.tolist()
        }


class NeuralNetworkTrainer(ModelTrainer):
    """Trainer for Neural Network models - represents deep learning methods."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train a Neural Network model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Neural networks require feature scaling
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(100, 50),  # Two hidden layers
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                learning_rate='adaptive',
                max_iter=500,
                random_state=0,
                early_stopping=True,
                validation_fraction=0.1
            )
        )
        
        r2_cv_mean = self.evaluate_model(pipeline, X, y)
        pipeline.fit(X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_nn")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": pipeline,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path
        }


class LightGBMTrainer(ModelTrainer):
    """Trainer for LightGBM models - represents gradient boosting with quantile regression."""
    
    def train_quantile_model(self, X, y, quantile_alpha=None):
        """Train a quantile regression model."""
        if quantile_alpha is None:
            model = lgb.LGBMRegressor(
                objective='regression', 
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=0,
                verbose=-1  # Suppress warnings
            )
        else:
            model = lgb.LGBMRegressor(
                objective='quantile', 
                alpha=quantile_alpha, 
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=0,
                verbose=-1
            )
        model.fit(X, y)
        return model
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train LightGBM models for median and quantile predictions."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        # Train three models: median, lower quantile, upper quantile
        model_median = self.train_quantile_model(X, y, quantile_alpha=None)
        model_lower = self.train_quantile_model(X, y, quantile_alpha=0.025)
        model_upper = self.train_quantile_model(X, y, quantile_alpha=0.975)

        r2_cv_mean = self.evaluate_model(model_median, X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_lgb")
        os.makedirs(model_dir, exist_ok=True)

        # Save models in txt format
        model_median_path = os.path.join(model_dir, f"{phys_state}_median.txt")
        model_lower_path = os.path.join(model_dir, f"{phys_state}_lower.txt")
        model_upper_path = os.path.join(model_dir, f"{phys_state}_upper.txt")

        model_median.booster_.save_model(model_median_path)
        model_lower.booster_.save_model(model_lower_path)
        model_upper.booster_.save_model(model_upper_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_median_path": model_median_path,
            "model_lower_path": model_lower_path,
            "model_upper_path": model_upper_path,
            "feature_importance": model_median.feature_importances_.tolist()
        }


class RandomForestTrainer(ModelTrainer):
    """Trainer for Random Forest models - represents ensemble tree-based methods."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train a Random Forest model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1
        )
        
        r2_cv_mean = self.evaluate_model(model, X, y)
        model.fit(X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_rf")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": model,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path,
            "feature_importance": model.feature_importances_.tolist()
        }


class SVRTrainer(ModelTrainer):
    """Trainer for Support Vector Regression models - represents kernel-based methods."""
    
    def train_model(self, df_filtered, phys_state, phase):
        """Train an SVR model for a specific physical state and phase."""
        X, y = self.prepare_data(df_filtered, phys_state)
        if X is None:
            return None

        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        
        # SVR requires feature scaling
        pipeline = make_pipeline(
            StandardScaler(),
            SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        )
        
        r2_cv_mean = self.evaluate_model(pipeline, X, y)
        pipeline.fit(X, y)

        model_dir = os.path.join(self.output_dir, f"models_phase{phase}_svr")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{phys_state}.joblib")
        joblib.dump({
            "model": pipeline,
            "env_factors": self.env_factors
        }, model_path)

        return {
            "state": phys_state,
            "r2_cv": r2_cv_mean,
            "model_path": model_path
        }


# ============================================================================
# MODEL TRAINING PIPELINE
# ============================================================================

def process_one_state_with_trainer(args):
    """Process a single physical state with a given trainer."""
    trainer, df_filtered, phys_state, phase = args
    try:
        return trainer.train_model(df_filtered, phys_state, phase)
    except Exception as e:
        return None


def process_multivariate_analysis_with_trainer(
    df_filtered, 
    trainer_class, 
    analysis_output,
    phase_range=range(1, 13), 
    suffix="",
    **trainer_kwargs
):
    """Process multivariate analysis using a specific trainer class."""
    all_phase_r2_summary = []
    trainer = trainer_class(ENV_FACTORS, analysis_output, **trainer_kwargs)

    for phase in phase_range:
        phase_suffix = f"_Phase{phase}"
        phys_states = [
            col for col in df_filtered.columns
            if col not in ENV_FACTORS and col.endswith(phase_suffix)
        ]

        if not phys_states:
            print(f"Phase {phase}: No physical states found, skipping...")
            continue

        print(f"Training {trainer_class.__name__} models for Phase {phase} ({len(phys_states)} states)")

        records = []
        r2_values = []
        
        # Use clean progress tracking
        processed_count = 0
        total_states = len(phys_states)
        print(f"  Starting training for {total_states} states...")

        with ThreadPoolExecutor(max_workers=8) as executor:
            args_list = [
                (trainer, df_filtered, state, phase)
                for state in phys_states
            ]
            futures = [executor.submit(process_one_state_with_trainer, args) for args in args_list]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    r2_values.append(result["r2_cv"])
                
                processed_count += 1
                progress_pct = (processed_count / total_states) * 100
                successful_count = len(records)
                
                # Progress update every 10 models or when complete
                if processed_count % 10 == 0 or processed_count == total_states:
                    status = f"  Progress: {processed_count}/{total_states} ({progress_pct:.1f}%) | "
                    status += f"Successful: {successful_count} | "
                    status += f"Mean R²: {np.mean(r2_values):.3f}" if r2_values else "Mean R²: N/A"
                    print(status)

        # Clear progress line and show completion
        if records:
            final_msg = f"   Phase {phase} complete: {len(records)}/{total_states} models (Mean R²: {np.mean(r2_values):.3f})"
            print(final_msg)
            
            # Save per-model performance
            coef_df = pd.DataFrame(records)
            csv_path = os.path.join(analysis_output, f"multivariate_coefficients_phase{phase}{suffix}.csv")
            coef_df.to_csv(csv_path, index=False)

            # Generate histogram
            plot_r2_histogram(r2_values, phase, analysis_output, suffix)

            # Save summary
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
            all_phase_r2_summary.append(summary)
        else:
            print(f"   Phase {phase}: No models generated")

    if all_phase_r2_summary:
        summary_df = pd.DataFrame(all_phase_r2_summary)
        summary_csv_path = os.path.join(analysis_output, f"multivariate_r2_summary{suffix}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f" Saved {trainer_class.__name__} R² summary to {summary_csv_path}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_r2_histogram(r2_values, phase, output_dir, suffix=""):
    """Plot R² distribution histogram for a given phase."""
    plt.figure(figsize=(6, 4))
    plt.hist(r2_values, bins=np.linspace(0, 1, 21), color="skyblue", edgecolor="black")
    plt.title(f"R² Distribution for Phase {phase}")
    plt.xlabel("R² Score")
    plt.ylabel("Number of Models")
    plt.grid(True)
    plt.tight_layout()
    filename = f"r2_histogram_phase{phase}{suffix}.pdf"
    plt.savefig(os.path.join(output_dir, filename), format="pdf")
    plt.close()


def plot_all_r2_histograms(analysis_output_dir, phase_range=range(1, 13), suffix=""):
    """Plot R² histograms for all phases in a single figure."""
    cols = 4
    rows = 3
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()

    for idx, phase in enumerate(phase_range):
        coef_path = os.path.join(analysis_output_dir, f"multivariate_coefficients_phase{phase}{suffix}.csv")
        if not os.path.exists(coef_path):
            continue

        df = pd.read_csv(coef_path)
        if "r2_cv" not in df.columns:
            continue

        r2_values = df["r2_cv"].values
        ax = axs[idx]
        ax.hist(r2_values, bins=np.linspace(0, 1, 21), color="skyblue", edgecolor="black")
        ax.set_title(f"Phase {phase}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("R²")
        ax.set_ylabel("Count")
        ax.grid(True)

    # Hide unused subplots
    for j in range(len(phase_range), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    output_pdf = os.path.join(analysis_output_dir, f"r2_histograms_all_phases{suffix}.pdf")
    plt.savefig(output_pdf, format="pdf")
    plt.close()
    print(f"Saved R² histograms to {output_pdf}")


def plot_model_performance_summary(summary_csv_path, output_dir, model_name=""):
    """Visualize model R² performance summary per phase."""
    os.makedirs(output_dir, exist_ok=True)
    df_summary = pd.read_csv(summary_csv_path)

    # Sort by phase number
    if "Phase" in df_summary.columns:
        df_summary["PhaseNum"] = df_summary["Phase"]
        if df_summary["Phase"].dtype == object:
            df_summary["PhaseNum"] = df_summary["Phase"].str.extract(r"(\d+)").astype(int)
        df_summary = df_summary.sort_values("PhaseNum")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_summary, x="PhaseNum", y="Mean_R2_CV", color="skyblue", label="Mean R² (CV)")
    plt.plot(df_summary["PhaseNum"], df_summary["Max_R2_CV"], label="Max R²", marker="o", linestyle="--", color="green")
    plt.plot(df_summary["PhaseNum"], df_summary["Min_R2_CV"], label="Min R²", marker="x", linestyle="--", color="red")
    plt.plot(df_summary["PhaseNum"], df_summary["Median_R2_CV"], label="Median R²", marker="^", linestyle="--", color="orange")

    plt.title(f"{model_name} Model Performance (R² CV per Phase)")
    plt.xlabel("Phase")
    plt.ylabel("R² Score")
    plt.ylim(-0.5, 1.05)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"r2_summary_plot_{model_name.lower().replace(' ', '_')}.pdf")
    plt.savefig(plot_path, format="pdf")
    plt.close()
    print(f" Saved {model_name} R² summary plot to: {plot_path}")


# ============================================================================
# PERFORMANCE COMPARISON FUNCTIONS
# ============================================================================

def generate_model_performance_comparison_table(output_dir, phase_range=range(1, 13)):
    """Generate a comprehensive comparison table of all model performances."""
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
        
        summary_path = os.path.join(output_dir, f"multivariate_r2_summary{suffix}.csv")
        
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found for {display_name}: {summary_path}")
            continue
            
        df_summary = pd.read_csv(summary_path)
        
        # Calculate overall statistics across all phases
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
        output_path = os.path.join(output_dir, "model_performance_comparison.csv")
        df_comparison.to_csv(output_path, index=False)
        print(f" Model performance comparison saved to: {output_path}")
        
        # Create paper-ready table
        paper_table = df_comparison[[
            "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", 
            "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7", 
            "Best_Phase_R2", "R2_Std_Across_Phases"
        ]].copy()
        
        # Rename columns for paper
        paper_table.columns = [
            "Model", "Total Models", "Mean R²", "Median R²", 
            "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7", 
            "Best Phase R²", "R² Std Dev"
        ]
        
        paper_output_path = os.path.join(output_dir, "model_performance_paper_table.csv")
        paper_table.to_csv(paper_output_path, index=False)
        print(f" Paper-ready table saved to: {paper_output_path}")
        
        # Print summary to console
        simulator_name = os.path.basename(output_dir).replace("analysis_output_", "").upper()
        print(f"\n" + "="*80)
        print(f"MODEL PERFORMANCE COMPARISON SUMMARY - {simulator_name}")
        print("="*80)
        print(paper_table.to_string(index=False))
        print("="*80)
        
    return df_comparison


def generate_cross_simulator_comparison_table():
    """Generate a comparison table between SITL and Gazebo simulators."""
    cross_comparison_results = []
    
    for sim_name, config in SIMULATOR_CONFIGS.items():
        analysis_output = config["analysis_output"]
        display_name = config["display_name"]
        
        comparison_path = os.path.join(analysis_output, "model_performance_comparison.csv")
        if not os.path.exists(comparison_path):
            print(f"Warning: No performance comparison found for {display_name}: {comparison_path}")
            continue
            
        df_sim_comparison = pd.read_csv(comparison_path)
        df_sim_comparison["Simulator"] = sim_name
        cross_comparison_results.append(df_sim_comparison)
    
    if not cross_comparison_results:
        print("No simulator comparisons found.")
        return None
    
    # Combine all simulator results
    df_cross_comparison = pd.concat(cross_comparison_results, ignore_index=True)
    
    # Reorder columns
    cols = df_cross_comparison.columns.tolist()
    cols = ["Simulator"] + [col for col in cols if col != "Simulator"]
    df_cross_comparison = df_cross_comparison[cols]
    
    # Save cross-simulator comparison
    cross_output_dir = "./analysis_output_combined"
    os.makedirs(cross_output_dir, exist_ok=True)
    
    output_path = os.path.join(cross_output_dir, "cross_simulator_model_comparison.csv")
    df_cross_comparison.to_csv(output_path, index=False)
    print(f" Cross-simulator comparison saved to: {output_path}")
    
    # Create paper-ready table
    paper_cross_table = df_cross_comparison[[
        "Simulator", "Model", "Total_Models", "Overall_Mean_R2", "Overall_Median_R2", 
        "Pct_R2_GTE_0.9", "Pct_R2_GTE_0.8", "Pct_R2_GTE_0.7"
    ]].copy()
    
    paper_cross_table.columns = [
        "Simulator", "Model", "Total Models", "Mean R²", "Median R²", 
        "% R² ≥ 0.9", "% R² ≥ 0.8", "% R² ≥ 0.7"
    ]
    
    paper_cross_output_path = os.path.join(cross_output_dir, "cross_simulator_paper_table.csv")
    paper_cross_table.to_csv(paper_cross_output_path, index=False)
    print(f" Cross-simulator paper table saved to: {paper_cross_output_path}")
    
    # Print summary
    print(f"\n" + "="*100)
    print("CROSS-SIMULATOR MODEL PERFORMANCE COMPARISON")
    print("="*100)
    print(paper_cross_table.to_string(index=False))
    print("="*100)
    
    return df_cross_comparison


def plot_cross_simulator_comparison(cross_comparison_df=None):
    """Create visualizations comparing model performance across simulators."""
    if cross_comparison_df is None:
        cross_output_dir = "./analysis_output_combined"
        cross_path = os.path.join(cross_output_dir, "cross_simulator_model_comparison.csv")
        if os.path.exists(cross_path):
            cross_comparison_df = pd.read_csv(cross_path)
        else:
            print("No cross-simulator comparison data found.")
            return
    
    cross_output_dir = "./analysis_output_combined"
    os.makedirs(cross_output_dir, exist_ok=True)
    
    # Mean R² comparison bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cross_comparison_df, x="Model", y="Overall_Mean_R2", hue="Simulator")
    plt.title("Model Performance Comparison: SITL vs Gazebo\n(Mean R² Score)", fontsize=14)
    plt.xlabel("Model Type")
    plt.ylabel("Mean R² Score")
    plt.xticks(rotation=45)
    plt.legend(title="Simulator")
    plt.tight_layout()
    
    plot_path = os.path.join(cross_output_dir, "cross_simulator_mean_r2_comparison.pdf")
    plt.savefig(plot_path, format="pdf", dpi=300)
    plt.close()
    print(f" Cross-simulator mean R² comparison saved to: {plot_path}")


# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_env_factor_influence_via_pca(
    df_filtered,
    env_factors,
    phase_range=range(1, 13),
    n_components=3,
    output_dir="./analysis_output",
    top_k=5
):
    """Analyze environmental factor influence using PCA on physical states."""
    os.makedirs(output_dir, exist_ok=True)
    phase_records = []

    for phase in phase_range:
        suffix = f"_Phase{phase}"
        phys_state_cols = [col for col in df_filtered.columns if col.endswith(suffix) and col not in env_factors]

        if not phys_state_cols:
            continue

        cols_needed = env_factors + phys_state_cols
        df_phase = df_filtered[cols_needed].dropna()
        if df_phase.empty:
            continue

        # Normalize physical states
        X_phys = df_phase[phys_state_cols]
        X_phys_std = (X_phys - X_phys.mean()) / X_phys.std()

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_phys_std)

        # Correlate ENV_FACTORS with each PC
        for comp_idx in range(n_components):
            pc_values = X_pca[:, comp_idx]
            for factor in env_factors:
                corr = np.corrcoef(df_phase[factor], pc_values)[0, 1]
                phase_records.append({
                    "Phase": f"Phase{phase}",
                    "PC": f"PC{comp_idx+1}",
                    "ENV_FACTOR": factor,
                    "Correlation": corr,
                    "AbsCorr": abs(corr),
                    "ExplainedVar": pca.explained_variance_ratio_[comp_idx]
                })

    # Save results
    df_result = pd.DataFrame(phase_records)
    result_path = os.path.join(output_dir, "pca_env_factor_influence.csv")
    df_result.to_csv(result_path, index=False)
    print(f" Saved PCA influence table to {result_path}")

    # Create heatmap for PC1
    df_pc1 = df_result[df_result["PC"] == "PC1"]
    df_pc1["PhaseNum"] = df_pc1["Phase"].str.extract(r"Phase(\d+)").astype(int)

    heatmap_df = (
        df_pc1.sort_values(["PhaseNum", "AbsCorr"], ascending=[True, False])
        .groupby("PhaseNum")
        .head(top_k)
        .pivot(index="PhaseNum", columns="ENV_FACTOR", values="AbsCorr")
        .fillna(0)
    )

    heatmap_df = heatmap_df.sort_index()

    plt.figure(figsize=(min(heatmap_df.shape[1]*0.6, 20), 6))
    sns.heatmap(heatmap_df, annot=True, cmap="viridis", cbar=True, linewidths=0.5)
    plt.title(f"Top {top_k} ENV_FACTORS by Abs(Correlation) with PC1 (Per Phase)", fontsize=12)
    plt.ylabel("Phase")
    plt.xlabel("ENV Factor")
    plt.xticks(rotation=90)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "pca_env_factor_pc1_top_heatmap.pdf")
    plt.savefig(heatmap_path, format="pdf")
    plt.close()
    print(f" Saved PC1 ENV_FACTOR heatmap to {heatmap_path}")

    return df_result


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def process_simulator_analysis(simulator_name_or_path, custom_output_dir=None):
    """
    Process analysis for a single simulator or custom data path.
    
    Args:
        simulator_name_or_path: Either "SITL"/"Gazebo" or a custom data path
        custom_output_dir: Custom output directory (optional)
        
    Returns:
        pd.DataFrame: Filtered data for this simulator
    """
    # Determine if it's a predefined simulator or custom path
    if simulator_name_or_path in SIMULATOR_CONFIGS:
        config = SIMULATOR_CONFIGS[simulator_name_or_path]
        data_root = config["data_root"]
        analysis_output = config["analysis_output"]
        display_name = config["display_name"]
    else:
        # Custom data path
        data_root = simulator_name_or_path
        if custom_output_dir:
            analysis_output = custom_output_dir
        else:
            basename = os.path.basename(data_root.rstrip('/'))
            analysis_output = f"./analysis_output_{basename}"
        display_name = f"Custom Data ({os.path.basename(data_root)})"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {display_name.upper()}")
    print(f"Data: {data_root}")
    print(f"Output: {analysis_output}")
    print(f"{'='*60}")
    
    # Check if filtered data exists
    filtered_path = os.path.join(analysis_output, "filtered_data.csv")
    if not os.path.exists(filtered_path):
        print(f" Filtered data not found: {filtered_path}")
        print(f"Please run data preprocessing first:")
        print(f"python data_preprocessor.py {data_root} {analysis_output}")
        return None
    
    # Load filtered data
    print("Loading filtered data...")
    df_filtered = pd.read_csv(filtered_path)
    print(f" Loaded data shape: {df_filtered.shape}")
    
    # Correlation Analysis
    print(f"\n--- Correlation Analysis ---")
    correlation_file = os.path.join(analysis_output, "Top_Correlated_Factors.csv")
    if check_file_exists(correlation_file, "correlation analysis"):
        print("Skipping correlation analysis...")
    else:
        analyze_top_correlations(df_filtered, output_path=correlation_file)
    
    # Model Training
    print(f"\n--- Training Models ---")
    models_to_train = [
        {"trainer": PolynomialRidgeTrainer, "suffix": "", "name": "Polynomial + Ridge", "kwargs": {"degree": 3}},
        {"trainer": GaussianProcessTrainer, "suffix": "_gpr", "name": "Gaussian Process", "kwargs": {}},
        {"trainer": XGBoostTrainer, "suffix": "_xgb", "name": "XGBoost", "kwargs": {}},
        {"trainer": LightGBMTrainer, "suffix": "_lgb", "name": "LightGBM", "kwargs": {}},
        {"trainer": NeuralNetworkTrainer, "suffix": "_nn", "name": "Neural Network", "kwargs": {}},
        {"trainer": RandomForestTrainer, "suffix": "_rf", "name": "Random Forest", "kwargs": {}},
        {"trainer": SVRTrainer, "suffix": "_svr", "name": "Support Vector Regression", "kwargs": {}}
    ]
    
    for model_config in models_to_train:
        print(f"\n--- Training {model_config['name']} Models ---")
        summary_file = os.path.join(analysis_output, f"multivariate_r2_summary{model_config['suffix']}.csv")
        
        if check_file_exists(summary_file, f"{model_config['name']} results"):
            print(f"Skipping {model_config['name']} training...")
        else:
            process_multivariate_analysis_with_trainer(
                df_filtered, 
                model_config["trainer"],
                analysis_output,
                suffix=model_config["suffix"],
                **model_config["kwargs"]
            )
    
    # Performance Analysis
    print(f"\n--- Performance Analysis ---")
    generate_model_performance_comparison_table(analysis_output)
    
    # Visualizations
    print(f"\n--- Generating Visualizations ---")
    # R² histograms
    suffixes = ["", "_gpr", "_xgb", "_lgb", "_nn", "_rf", "_svr"]
    for suffix in suffixes:
        hist_file = f"r2_histograms_all_phases{suffix}.pdf"
        if not os.path.exists(os.path.join(analysis_output, hist_file)):
            plot_all_r2_histograms(analysis_output, phase_range=range(1, 13), suffix=suffix)
    
    # Performance summary plots
    performance_plots = [
        {"file": "multivariate_r2_summary.csv", "name": "Polynomial Ridge"},
        {"file": "multivariate_r2_summary_gpr.csv", "name": "Gaussian Process"},
        {"file": "multivariate_r2_summary_xgb.csv", "name": "XGBoost"},
        {"file": "multivariate_r2_summary_lgb.csv", "name": "LightGBM"},
        {"file": "multivariate_r2_summary_nn.csv", "name": "Neural Network"},
        {"file": "multivariate_r2_summary_rf.csv", "name": "Random Forest"},
        {"file": "multivariate_r2_summary_svr.csv", "name": "Support Vector Regression"}
    ]
    
    for plot_config in performance_plots:
        summary_path = os.path.join(analysis_output, plot_config["file"])
        if os.path.exists(summary_path):
            plot_model_performance_summary(summary_path, analysis_output, plot_config["name"])
    
    # Advanced Analysis
    print(f"\n--- Advanced Analysis ---")
    pca_file = os.path.join(analysis_output, "pca_env_factor_influence.csv")
    if not check_file_exists(pca_file, "PCA analysis"):
        analyze_env_factor_influence_via_pca(
            df_filtered,
            ENV_FACTORS,
            output_dir=analysis_output,
            top_k=10
        )
    
    print(f" Analysis complete for {display_name}!")
    return df_filtered


def main_dual_simulator_analysis(sitl_data_path=None, gazebo_data_path=None):
    """Main execution pipeline for dual simulator analysis."""
    print("Starting Dual-Simulator Model Analysis Pipeline...")
    print("="*80)
    
    # Configure simulator paths
    if sitl_data_path or gazebo_data_path:
        configure_simulator_paths(sitl_data_path, gazebo_data_path)
    
    # Process each simulator
    simulator_results = {}
    
    for sim_name in ["SITL", "Gazebo"]:
        config = SIMULATOR_CONFIGS[sim_name]
        if not os.path.exists(config["data_root"]):
            print(f"  Skipping {sim_name}: Data directory not found - {config['data_root']}")
            continue
            
        df_filtered = process_simulator_analysis(sim_name)
        if df_filtered is not None:
            simulator_results[sim_name] = df_filtered
    
    # Cross-simulator comparison
    if len(simulator_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-SIMULATOR COMPARISON")
        print(f"{'='*80}")
        
        cross_comparison_df = generate_cross_simulator_comparison_table()
        if cross_comparison_df is not None:
            plot_cross_simulator_comparison(cross_comparison_df)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    for sim_name, config in SIMULATOR_CONFIGS.items():
        if sim_name in simulator_results:
            print(f" {config['display_name']} results: {config['analysis_output']}")
    
    if len(simulator_results) > 1:
        print(f" Cross-simulator comparison: ./analysis_output_combined")


def main_single_analysis(data_path, output_dir=None):
    """Main execution pipeline for single dataset analysis."""
    print("Starting Single Dataset Model Analysis Pipeline...")
    print("="*60)
    
    df_filtered = process_simulator_analysis(data_path, output_dir)
    
    if df_filtered is not None:
        final_output = output_dir if output_dir else f"./analysis_output_{os.path.basename(data_path.rstrip('/'))}"
        print(f"\n Analysis complete! Results saved to: {final_output}")
        print(f"\n Key Output Files:")
        print(f"  • Model Comparison: {os.path.join(final_output, 'model_performance_paper_table.csv')}")
        print(f"  • R² Summaries: {final_output}/multivariate_r2_summary*.csv")
        print(f"  • Visualizations: {final_output}/*.pdf")
    else:
        print(" Analysis failed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single dataset: python model_analyzer.py <data_path> [output_dir]")
        print("  Dual simulator: python model_analyzer.py --dual <sitl_path> <gazebo_path>")
        print("\nExamples:")
        print("  python model_analyzer.py ./analysis_output_sitl")
        print("  python model_analyzer.py --dual ./data_sitl ./data_gazebo")
        sys.exit(1)
    
    if sys.argv[1] == "--dual":
        if len(sys.argv) < 4:
            print("Error: Dual simulator mode requires both SITL and Gazebo paths")
            sys.exit(1)
        sitl_path = sys.argv[2]
        gazebo_path = sys.argv[3]
        main_dual_simulator_analysis(sitl_path, gazebo_path)
    else:
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        main_single_analysis(data_path, output_dir)