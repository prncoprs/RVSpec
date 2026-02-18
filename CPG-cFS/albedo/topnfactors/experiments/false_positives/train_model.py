#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Model for Albedo Error Reduction Prediction
Trains a model to predict error_reduction based on spacecraft and orbital parameters
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.preprocessing import StandardScaler

# Set matplotlib parameters
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 100,
})

class AlbedoErrorPredictor:
    def __init__(self, data_path, output_dir):
        """
        Initialize the predictor
        
        Args:
            data_path: Path to the analysis_data.csv file
            output_dir: Directory to save outputs
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.model_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.figures_dir = self.output_dir / "figures"
        
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
        
        # Set random seed for reproducibility
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
        
        # Check for missing features
        missing_features = []
        for col in self.feature_columns:
            if col not in self.df.columns:
                missing_features.append(col)
        
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Remove rows with NaN in features
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.feature_columns)
        removed_feature_nan = initial_rows - len(self.df)
        print(f"Removed {removed_feature_nan} rows with NaN in features")
        
        print(f"Final dataset size: {len(self.df)} rows")
        
        # Print basic statistics
        print("\nTarget variable statistics:")
        print(f"Mean error_reduction: {self.df[self.target_column].mean():.6f}")
        print(f"Std error_reduction: {self.df[self.target_column].std():.6f}")
        print(f"Min error_reduction: {self.df[self.target_column].min():.6f}")
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
    
    def train_model(self, tune_hyperparameters=True):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3]
            }
            
            # Create base model
            rf_base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            # Grid search
            grid_search = GridSearchCV(
                rf_base, param_grid, 
                cv=5, scoring='neg_mean_absolute_error',
                verbose=1, n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Best model
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV MAE: {-grid_search.best_score_:.6f}")
            
            # Save best parameters
            with open(self.results_dir / "best_parameters.json", 'w') as f:
                json.dump(grid_search.best_params_, f, indent=2)
        
        else:
            print("Training with default parameters...")
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
        
        print("Model training complete!")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'train': {
                'mae': mean_absolute_error(self.y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'r2': r2_score(self.y_train, y_train_pred),
                'median_ae': median_absolute_error(self.y_train, y_train_pred)
            },
            'test': {
                'mae': mean_absolute_error(self.y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'r2': r2_score(self.y_test, y_test_pred),
                'median_ae': median_absolute_error(self.y_test, y_test_pred)
            }
        }
        
        # Print metrics
        print("\nModel Performance:")
        print("-" * 50)
        print(f"{'Metric':<15} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        for metric in ['mae', 'rmse', 'r2', 'median_ae']:
            print(f"{metric.upper():<15} {metrics['train'][metric]:<15.6f} {metrics['test'][metric]:<15.6f}")
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(self.results_dir / "performance_metrics.csv")
        
        # Save predictions
        test_results = pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_test_pred,
            'error': y_test_pred - self.y_test
        })
        test_results.to_csv(self.results_dir / "test_predictions.csv", index=False)
        
        return metrics, y_test_pred
    
    def analyze_features(self):
        """Analyze feature importance"""
        print("\nAnalyzing feature importance...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save feature importances
        feature_importance_df.to_csv(self.results_dir / "feature_importances.csv", index=False)
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        print("-" * 50)
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.6f}")
        
        return feature_importance_df
    
    def generate_visualizations(self, metrics, y_test_pred, feature_importance_df):
        """Generate visualization plots"""
        print("\nGenerating visualizations...")
        
        # 1. Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Most Important Features')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "feature_importance.pdf")
        plt.close()
        
        # 2. Prediction Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.y_test, y_test_pred, alpha=0.5, s=10)
        
        # Add diagonal line
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('Actual Error Reduction (degrees)')
        ax.set_ylabel('Predicted Error Reduction (degrees)')
        ax.set_title(f'Predictions vs Actual (Test R² = {metrics["test"]["r2"]:.4f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "prediction_scatter.pdf")
        plt.close()
        
        # 3. Residual Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        residuals = y_test_pred - self.y_test
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_test_pred, residuals, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual Distribution
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error by Magnitude
        axes[1, 1].scatter(self.y_test, np.abs(residuals), alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Actual Error Reduction')
        axes[1, 1].set_ylabel('Absolute Residual')
        axes[1, 1].set_title('Absolute Error vs Actual Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Residual Analysis', fontsize=14)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "residual_plots.pdf")
        plt.close()
        
        print(f"Visualizations saved to {self.figures_dir}")
    
    def save_model(self):
        """Save the trained model and metadata"""
        print("\nSaving model...")
        
        # Save model
        model_path = self.model_dir / "rf_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'n_training_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'model_params': self.model.get_params()
        }
        
        with open(self.model_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_path = self.results_dir / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RANDOM FOREST MODEL TRAINING REPORT\n")
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
            f.write(f"Std: {self.df[self.target_column].std():.6f}\n")
            f.write(f"Min: {self.df[self.target_column].min():.6f}\n")
            f.write(f"Max: {self.df[self.target_column].max():.6f}\n\n")
            
            f.write("MODEL PARAMETERS\n")
            f.write("-"*60 + "\n")
            for param, value in self.model.get_params().items():
                f.write(f"{param}: {value}\n")
            
        print(f"Report saved to {report_path}")
    
    def run(self, tune_hyperparameters=True):
        """Run the complete training pipeline"""
        print("="*80)
        print("Starting Albedo Error Reduction Prediction Pipeline")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Split data
        self.split_data()
        
        # Train model
        self.train_model(tune_hyperparameters=tune_hyperparameters)
        
        # Evaluate model
        metrics, y_test_pred = self.evaluate_model()
        
        # Analyze features
        feature_importance_df = self.analyze_features()
        
        # Generate visualizations
        self.generate_visualizations(metrics, y_test_pred, feature_importance_df)
        
        # Save model
        self.save_model()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*80)
        print("Pipeline Complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest model for error reduction prediction')
    parser.add_argument('--data-path',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/results/analysis_data.csv',
                       help='Path to analysis_data.csv')
    parser.add_argument('--output-dir',
                       default='<RVSPEC_ROOT>/CPG-cFS/albedo/topnfactors/experiments/false_positives',
                       help='Output directory')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Create predictor and run
    predictor = AlbedoErrorPredictor(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    predictor.run(tune_hyperparameters=not args.no_tuning)


if __name__ == "__main__":
    main()