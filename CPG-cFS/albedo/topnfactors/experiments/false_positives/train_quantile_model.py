#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantile Random Forest Model for Albedo Error Reduction 95th Percentile Prediction
Trains a model to predict the 95th percentile of error_reduction
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import quantile-forest, if not available, provide installation instructions
try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    print("Error: quantile-forest library not installed.")
    print("Please install it using: pip install quantile-forest --break-system-packages")
    print("Or in a virtual environment: pip install quantile-forest")
    sys.exit(1)

# Set matplotlib parameters
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 100,
})

class QuantileRFPredictor:
    def __init__(self, data_path, output_dir, quantile=0.95):
        """
        Initialize the Quantile RF predictor
        
        Args:
            data_path: Path to the analysis_data.csv file
            output_dir: Directory to save outputs
            quantile: Quantile to predict (default 0.95 for 95th percentile)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.quantile = quantile
        
        # Create output directories
        self.model_dir = self.output_dir / "models_quantile"
        self.results_dir = self.output_dir / "results_quantile"
        self.figures_dir = self.output_dir / "figures_quantile"
        
        for dir_path in [self.model_dir, self.results_dir, self.figures_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Define feature columns
        self.feature_columns = [
            'MASS', 'MOI_XX', 'MOI_YY', 'MOI_ZZ',
            'MTB_0_SATURATION', 'MTB_1_SATURATION', 'MTB_2_SATURATION',
            'MAG_SATURATION', 'MAG_QUANTIZATION', 'MAG_NOISE',
            'GYRO_MAX_RATE', 'GYRO_SCALE_FACTOR_ERROR', 'GYRO_QUANTIZATION', 'GYRO_ANGLE_NOISE',
            'ORB_PERIAPSIS_ALT', 'ORB_APOAPSIS_ALT', 'ORB_INCLINATION', 
            'ORB_RAAN', 'ORB_ARG_PERIAPSIS', 'ORB_TRUE_ANOMALY',
            'F10_7_FLUX', 'AP_INDEX',
            'global_phase_angle', 'is_daylight'
        ]
        
        self.target_column = 'error_reduction'
        
        # Initialize model and data holders
        self.model = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set random seed
        self.random_state = 42
        np.random.seed(self.random_state)
    
    def load_and_prepare_data(self):
        """Load and prepare the data for training"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows")
        
        # Remove rows with NaN in target
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[self.target_column])
        removed_target_nan = initial_rows - len(self.df)
        print(f"Removed {removed_target_nan} rows with NaN in target")
        
        # Remove rows with NaN in features
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.feature_columns)
        removed_feature_nan = initial_rows - len(self.df)
        print(f"Removed {removed_feature_nan} rows with NaN in features")
        
        print(f"Final dataset size: {len(self.df)} rows")
        
        # Calculate actual 95th percentile of error_reduction
        actual_p95 = np.percentile(self.df[self.target_column], 95)
        
        print("\nTarget variable statistics:")
        print(f"Mean error_reduction: {self.df[self.target_column].mean():.6f}")
        print(f"Median error_reduction: {self.df[self.target_column].median():.6f}")
        print(f"95th percentile error_reduction: {actual_p95:.6f}")
        print(f"Max error_reduction: {self.df[self.target_column].max():.6f}")
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
    
    def train_quantile_model(self, tune_hyperparameters=True):
        """Train Quantile Random Forest model"""
        print(f"\nTraining Quantile Random Forest for {self.quantile*100}th percentile...")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter combinations to try
            param_combinations = [
                {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10},
                {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5},
                {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 2},
                {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5},
            ]
            
            best_score = float('inf')
            best_params = None
            
            # Use cross-validation to evaluate each combination
            kf = KFold(n_splits=3, random_state=self.random_state, shuffle=True)
            
            for params in param_combinations:
                print(f"  Testing: {params}")
                
                scores = []
                for train_idx, val_idx in kf.split(self.X_train):
                    X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                    
                    # Train model
                    qrf = RandomForestQuantileRegressor(
                        random_state=self.random_state,
                        n_jobs=-1,
                        **params
                    )
                    qrf.fit(X_tr, y_tr)
                    
                    # Predict 95th percentile
                    y_pred = qrf.predict(X_val, quantiles=[self.quantile])
                    
                    # Calculate quantile loss (pinball loss)
                    errors = y_val.values - y_pred.flatten()
                    quantile_loss = np.mean(np.maximum(self.quantile * errors, (self.quantile - 1) * errors))
                    scores.append(quantile_loss)
                
                avg_score = np.mean(scores)
                print(f"    Average quantile loss: {avg_score:.6f}")
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
            
            print(f"\nBest parameters: {best_params}")
            print(f"Best quantile loss: {best_score:.6f}")
            
            # Train final model with best parameters
            self.model = RandomForestQuantileRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **best_params
            )
            
            # Save best parameters
            with open(self.results_dir / "best_parameters_quantile.json", 'w') as f:
                json.dump(best_params, f, indent=2)
        
        else:
            print("Training with default parameters...")
            self.model = RandomForestQuantileRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete!")
    
    def evaluate_model(self):
        """Evaluate quantile model performance"""
        print("\nEvaluating quantile model...")
        
        # Make predictions for multiple quantiles for comparison
        quantiles_to_predict = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Training predictions
        y_train_pred_all = self.model.predict(self.X_train, quantiles=quantiles_to_predict)
        y_train_pred_95 = y_train_pred_all[:, -1]  # 95th percentile
        
        # Test predictions
        y_test_pred_all = self.model.predict(self.X_test, quantiles=quantiles_to_predict)
        y_test_pred_95 = y_test_pred_all[:, -1]  # 95th percentile
        
        # Calculate coverage (what percentage of actual values are below predicted 95th percentile)
        train_coverage = np.mean(self.y_train.values <= y_train_pred_95)
        test_coverage = np.mean(self.y_test.values <= y_test_pred_95)
        
        # Calculate quantile loss (pinball loss) for 95th percentile
        def quantile_loss(y_true, y_pred, quantile):
            errors = y_true - y_pred
            return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
        
        train_qloss = quantile_loss(self.y_train.values, y_train_pred_95, self.quantile)
        test_qloss = quantile_loss(self.y_test.values, y_test_pred_95, self.quantile)
        
        # Calculate interval width (difference between 95th and 5th percentiles)
        train_interval_width = np.mean(y_train_pred_all[:, -1] - y_train_pred_all[:, 0])
        test_interval_width = np.mean(y_test_pred_all[:, -1] - y_test_pred_all[:, 0])
        
        # Calculate MAE for median prediction as reference
        y_test_pred_median = y_test_pred_all[:, 2]  # 50th percentile (median)
        median_mae = mean_absolute_error(self.y_test, y_test_pred_median)
        
        # Print metrics
        print("\nQuantile Model Performance:")
        print("-" * 50)
        print(f"{'Metric':<25} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        print(f"{'95% Coverage':<25} {train_coverage:<15.2%} {test_coverage:<15.2%}")
        print(f"{'Quantile Loss (95%)':<25} {train_qloss:<15.6f} {test_qloss:<15.6f}")
        print(f"{'90% Interval Width':<25} {train_interval_width:<15.4f} {test_interval_width:<15.4f}")
        print(f"{'Median MAE':<25} {'N/A':<15} {median_mae:<15.4f}")
        
        # Analyze conditional coverage (coverage by different conditions)
        print("\nConditional Coverage Analysis (Test Set):")
        
        # By daylight condition
        daylight_mask = self.X_test['is_daylight'] > 0.5
        daylight_coverage = np.mean(self.y_test[daylight_mask].values <= y_test_pred_95[daylight_mask])
        eclipse_coverage = np.mean(self.y_test[~daylight_mask].values <= y_test_pred_95[~daylight_mask])
        
        print(f"  Daylight: {daylight_coverage:.2%}")
        print(f"  Eclipse: {eclipse_coverage:.2%}")
        
        # Save metrics
        metrics = {
            'quantile': self.quantile,
            'train': {
                'coverage': train_coverage,
                'quantile_loss': train_qloss,
                'interval_width': train_interval_width
            },
            'test': {
                'coverage': test_coverage,
                'quantile_loss': test_qloss,
                'interval_width': test_interval_width,
                'median_mae': median_mae,
                'daylight_coverage': daylight_coverage,
                'eclipse_coverage': eclipse_coverage
            }
        }
        
        with open(self.results_dir / "quantile_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        test_results = pd.DataFrame({
            'actual': self.y_test,
            'predicted_p05': y_test_pred_all[:, 0],
            'predicted_p25': y_test_pred_all[:, 1],
            'predicted_p50': y_test_pred_all[:, 2],
            'predicted_p75': y_test_pred_all[:, 3],
            'predicted_p95': y_test_pred_all[:, 4],
            'is_daylight': self.X_test['is_daylight'].values
        })
        test_results.to_csv(self.results_dir / "test_predictions_quantile.csv", index=False)
        
        return metrics, y_test_pred_all
    
    def analyze_features(self):
        """Analyze feature importance for quantile model"""
        print("\nAnalyzing feature importance...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save feature importances
        feature_importance_df.to_csv(self.results_dir / "feature_importances_quantile.csv", index=False)
        
        # Print top 10 features
        print("\nTop 10 Most Important Features (Quantile Model):")
        print("-" * 50)
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.6f}")
        
        return feature_importance_df
    
    def generate_visualizations(self, metrics, y_test_pred_all, feature_importance_df):
        """Generate visualization plots for quantile model"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. Coverage Plot - Show actual vs predicted quantiles
        ax = axes[0, 0]
        y_test_sorted_idx = np.argsort(self.y_test.values)
        y_test_sorted = self.y_test.values[y_test_sorted_idx]
        
        # Plot predicted quantiles
        for i, q in enumerate([0.05, 0.25, 0.5, 0.75, 0.95]):
            y_pred_sorted = y_test_pred_all[y_test_sorted_idx, i]
            ax.plot(range(len(y_test_sorted)), y_pred_sorted, 
                   label=f'{q*100:.0f}th percentile', alpha=0.7)
        
        # Plot actual values
        ax.scatter(range(len(y_test_sorted)), y_test_sorted, 
                  alpha=0.3, s=1, c='black', label='Actual')
        
        ax.set_xlabel('Sample (sorted by actual value)')
        ax.set_ylabel('Error Reduction (degrees)')
        ax.set_title('Quantile Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Feature Importance Plot
        ax = axes[0, 1]
        top_features = feature_importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Features (Quantile Model)')
        ax.grid(True, alpha=0.3)
        
        # 3. Coverage by Percentile
        ax = axes[0, 2]
        percentiles = [5, 25, 50, 75, 95]
        coverages = []
        for i, p in enumerate(percentiles):
            pred = y_test_pred_all[:, i]
            coverage = np.mean(self.y_test.values <= pred) * 100
            coverages.append(coverage)
        
        ax.plot(percentiles, coverages, 'o-', label='Actual Coverage')
        ax.plot(percentiles, percentiles, 'r--', label='Ideal Coverage')
        ax.set_xlabel('Predicted Percentile')
        ax.set_ylabel('Actual Coverage (%)')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 95th Percentile Prediction vs Actual
        ax = axes[1, 0]
        y_test_pred_95 = y_test_pred_all[:, -1]
        ax.scatter(self.y_test, y_test_pred_95, alpha=0.5, s=10)
        
        # Add diagonal line
        min_val = min(self.y_test.min(), y_test_pred_95.min())
        max_val = max(self.y_test.max(), y_test_pred_95.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Add 95th percentile line
        actual_p95 = np.percentile(self.y_test, 95)
        ax.axhline(y=actual_p95, color='g', linestyle=':', label=f'Actual 95th: {actual_p95:.3f}')
        
        ax.set_xlabel('Actual Error Reduction (°)')
        ax.set_ylabel('Predicted 95th Percentile (°)')
        ax.set_title(f'95th Percentile Predictions (Coverage: {metrics["test"]["coverage"]:.1%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Prediction Interval Width
        ax = axes[1, 1]
        interval_width = y_test_pred_all[:, -1] - y_test_pred_all[:, 0]
        ax.hist(interval_width, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(interval_width), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(interval_width):.3f}')
        ax.set_xlabel('90% Interval Width (°)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Interval Widths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Conditional Coverage
        ax = axes[1, 2]
        
        # Group by daylight condition
        daylight_mask = self.X_test['is_daylight'] > 0.5
        conditions = ['All', 'Daylight', 'Eclipse']
        coverages = [
            metrics['test']['coverage'] * 100,
            metrics['test']['daylight_coverage'] * 100,
            metrics['test']['eclipse_coverage'] * 100
        ]
        
        bars = ax.bar(conditions, coverages, alpha=0.7)
        ax.axhline(y=95, color='r', linestyle='--', label='Target: 95%')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('95th Percentile Coverage by Condition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cov in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{cov:.1f}%', ha='center', va='bottom')
        
        # 7. Exceedance Analysis
        ax = axes[2, 0]
        exceedances = self.y_test.values > y_test_pred_95
        exceed_pct = np.mean(exceedances) * 100
        
        ax.scatter(self.y_test[~exceedances], y_test_pred_95[~exceedances], 
                  alpha=0.5, s=10, label=f'Within bound ({100-exceed_pct:.1f}%)', color='blue')
        ax.scatter(self.y_test[exceedances], y_test_pred_95[exceedances], 
                  alpha=0.7, s=20, label=f'Exceeds bound ({exceed_pct:.1f}%)', color='red')
        
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Actual Error Reduction (°)')
        ax.set_ylabel('Predicted 95th Percentile (°)')
        ax.set_title('Exceedance Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Quantile Loss by Percentile
        ax = axes[2, 1]
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        q_losses = []
        
        for i, q in enumerate(quantiles):
            y_pred = y_test_pred_all[:, i]
            errors = self.y_test.values - y_pred
            qloss = np.mean(np.maximum(q * errors, (q - 1) * errors))
            q_losses.append(qloss)
        
        ax.plot(quantiles, q_losses, 'o-')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Quantile Loss')
        ax.set_title('Quantile Loss Across Percentiles')
        ax.grid(True, alpha=0.3)
        
        # 9. Summary Statistics Table
        ax = axes[2, 2]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Model Type', 'Quantile RF (95%)'],
            ['Test Coverage', f'{metrics["test"]["coverage"]:.1%}'],
            ['Quantile Loss', f'{metrics["test"]["quantile_loss"]:.4f}'],
            ['Mean Interval Width', f'{metrics["test"]["interval_width"]:.3f}°'],
            ['Daylight Coverage', f'{metrics["test"]["daylight_coverage"]:.1%}'],
            ['Eclipse Coverage', f'{metrics["test"]["eclipse_coverage"]:.1%}'],
            ['Median MAE', f'{metrics["test"]["median_mae"]:.4f}°']
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Format header
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('Quantile Random Forest Model Analysis (95th Percentile)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.figures_dir / "quantile_rf_analysis.pdf"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def save_model(self):
        """Save the trained quantile model"""
        print("\nSaving quantile model...")
        
        # Save model
        model_path = self.model_dir / "quantile_rf_model_95.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestQuantileRegressor',
            'quantile': self.quantile,
            'training_date': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'n_training_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'model_params': self.model.get_params()
        }
        
        with open(self.model_dir / "quantile_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def generate_report(self, metrics):
        """Generate comprehensive text report"""
        report_path = self.results_dir / "quantile_training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("QUANTILE RANDOM FOREST MODEL TRAINING REPORT\n")
            f.write(f"Target: {self.quantile*100}th Percentile of Error Reduction\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Path: {self.data_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-"*60 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Number of features: {len(self.feature_columns)}\n\n")
            
            f.write("TARGET VARIABLE STATISTICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean: {self.df[self.target_column].mean():.6f}\n")
            f.write(f"Median: {self.df[self.target_column].median():.6f}\n")
            f.write(f"95th Percentile: {np.percentile(self.df[self.target_column], 95):.6f}\n")
            f.write(f"Max: {self.df[self.target_column].max():.6f}\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-"*60 + "\n")
            f.write(f"Test Set Coverage: {metrics['test']['coverage']:.1%}\n")
            f.write(f"  (Should be close to 95% for well-calibrated model)\n")
            f.write(f"Quantile Loss: {metrics['test']['quantile_loss']:.6f}\n")
            f.write(f"90% Interval Width: {metrics['test']['interval_width']:.4f}°\n\n")
            
            f.write("CONDITIONAL COVERAGE\n")
            f.write("-"*60 + "\n")
            f.write(f"Daylight: {metrics['test']['daylight_coverage']:.1%}\n")
            f.write(f"Eclipse: {metrics['test']['eclipse_coverage']:.1%}\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-"*60 + "\n")
            f.write("This model predicts the 95th percentile of error reduction.\n")
            f.write("In practical terms:\n")
            f.write("- 95% of actual error reductions should be BELOW the predicted value\n")
            f.write("- Only 5% of cases should exceed the predicted bound\n")
            f.write("- This provides a conservative upper bound for system design\n\n")
            
            if metrics['test']['coverage'] > 0.93 and metrics['test']['coverage'] < 0.97:
                f.write(" Model is well-calibrated (coverage close to 95%)\n")
            elif metrics['test']['coverage'] < 0.93:
                f.write(" Model is overconfident (coverage < 93%)\n")
            else:
                f.write(" Model is too conservative (coverage > 97%)\n")
            
        print(f"Report saved to {report_path}")
    
    def run(self, tune_hyperparameters=True):
        """Run the complete training pipeline"""
        print("="*80)
        print("Starting Quantile RF Training Pipeline (95th Percentile)")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Split data
        self.split_data()
        
        # Train model
        self.train_quantile_model(tune_hyperparameters=tune_hyperparameters)
        
        # Evaluate model
        metrics, y_test_pred_all = self.evaluate_model()
        
        # Analyze features
        feature_importance_df = self.analyze_features()
        
        # Generate visualizations
        self.generate_visualizations(metrics, y_test_pred_all, feature_importance_df)
        
        # Save model
        self.save_model()
        
        # Generate report
        self.generate_report(metrics)
        
        print("\n" + "="*80)
        print("Quantile RF Pipeline Complete!")
        print(f"95th percentile model coverage: {metrics['test']['coverage']:.1%}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train Quantile RF model for 95th percentile prediction')
    parser.add_argument('--data-path',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/results/analysis_data.csv',
                       help='Path to analysis_data.csv')
    parser.add_argument('--output-dir',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/false_positives',
                       help='Output directory')
    parser.add_argument('--quantile', type=float, default=0.95,
                       help='Quantile to predict (default: 0.95 for 95th percentile)')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Create predictor and run
    predictor = QuantileRFPredictor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        quantile=args.quantile
    )
    
    predictor.run(tune_hyperparameters=not args.no_tuning)


if __name__ == "__main__":
    main()