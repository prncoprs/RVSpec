#!/usr/bin/env python3
"""
Random Forest Model for Required Tau Prediction - Complete Version
Includes all fixes and optimizations for overfitting prevention
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           mean_absolute_percentage_error, explained_variance_score)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class RequiredTauPredictor:
    """
    A Random Forest based predictor for required_tau in vehicle braking scenarios.
    With overfitting prevention and outlier handling.
    """
    
    def __init__(self, config=None):
        """
        Initialize the predictor with configuration parameters.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config or self.get_default_config()
        self.model = None
        self.scaler = None
        self.feature_names = [
            'applied_mass',
            'applied_drag_coefficient', 
            'applied_tire_friction',
            'applied_brake_torque_front',
            'applied_brake_torque_rear',
            'applied_road_friction'
        ]
        self.target_name = 'required_tau'
        self.metrics = {}
        self.feature_importance = None
        
    @staticmethod
    def get_default_config():
        """Get default configuration for the model (optimized for preventing overfitting)."""
        return {
            'n_estimators': 150,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'max_samples': 0.9,  # Subsample for each tree
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
            'test_size': 0.25,  # Larger test set
            'use_scaler': True,
            'scaler_type': 'robust',
            'perform_grid_search': False,
            'cross_validation_folds': 5,
            'remove_outliers': True,
            'outlier_contamination': 0.05,
            'compare_models': True  # Compare multiple configurations
        }
    
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the CSV data.
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            tuple: (X_features, y_target, full_dataframe)
        """
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Data quality checks
        print(f"Initial data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Filter only successful runs
        if 'success' in df.columns:
            df_success = df[df['success'] == True].copy()
            print(f"Successful runs: {len(df_success)}/{len(df)} ({100*len(df_success)/len(df):.1f}%)")
        else:
            df_success = df.copy()
        
        # Check for missing values
        missing_counts = df_success[self.feature_names + [self.target_name]].isnull().sum()
        if missing_counts.any():
            print("\nMissing values detected:")
            print(missing_counts[missing_counts > 0])
            df_success = df_success.dropna(subset=self.feature_names + [self.target_name])
            print(f"Shape after removing missing values: {df_success.shape}")
        
        # Check for outliers using IQR method
        outlier_info = self._detect_outliers(df_success)
        
        # Extract features and target
        X = df_success[self.feature_names].values
        y = df_success[self.target_name].values
        
        # Remove outliers if configured
        if self.config['remove_outliers'] and len(outlier_info) > 0:
            X, y = self._remove_outliers(X, y)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Target value range: [{y.min():.4f}, {y.max():.4f}]")
        
        return X, y, df_success
    
    def _detect_outliers(self, df):
        """Detect and report outliers using IQR method."""
        print("\nOutlier Detection (IQR method):")
        outlier_info = {}
        for col in self.feature_names + [self.target_name]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)} outliers detected")
                outlier_info[col] = len(outliers)
        return outlier_info
    
    def _remove_outliers(self, X, y):
        """Remove outliers using Isolation Forest."""
        print(f"\nRemoving outliers (contamination={self.config['outlier_contamination']})...")
        iso_forest = IsolationForest(
            contamination=self.config['outlier_contamination'], 
            random_state=RANDOM_STATE
        )
        outliers = iso_forest.fit_predict(X)
        
        mask = outliers != -1
        X_clean = X[mask]
        y_clean = y[mask]
        
        n_removed = len(X) - len(X_clean)
        print(f"  Removed {n_removed} outlier samples ({100*n_removed/len(X):.1f}%)")
        print(f"  Remaining samples: {len(X_clean)}")
        
        return X_clean, y_clean
    
    def split_data(self, X, y):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            shuffle=True
        )
        
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if not self.config['use_scaler']:
            return X_train, X_test
        
        if self.config['scaler_type'] == 'standard':
            self.scaler = StandardScaler()
        else:  # robust
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nFeature scaling applied: {self.config['scaler_type']}")
        
        return X_train_scaled, X_test_scaled
    
    def perform_grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            dict: Best parameters found
        """
        if not self.config['perform_grid_search']:
            return None
        
        print("\nPerforming Grid Search for hyperparameter optimization...")
        
        # Conservative parameter ranges to prevent overfitting
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [4, 6, 8],
            'max_features': ['sqrt', 2, 3],
            'max_samples': [0.8, 0.9, 1.0]
        }
        
        rf = RandomForestRegressor(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=self.config['cross_validation_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=self.config['n_jobs'],
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
        
        # Update config with best parameters
        for param, value in grid_search.best_params_.items():
            self.config[param] = value
        
        return grid_search.best_params_
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """
        Compare multiple model configurations to find the best one.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Best model configuration and results
        """
        if not self.config['compare_models']:
            return None
        
        print("\nComparing multiple model configurations...")
        
        model_configs = [
            {
                'name': 'Conservative',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 20,
                    'min_samples_leaf': 8,
                    'max_features': 2,
                    'max_samples': 0.8,
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1
                }
            },
            {
                'name': 'Moderate',
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'min_samples_split': 15,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'max_samples': 0.9,
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1
                }
            },
            {
                'name': 'Balanced',
                'params': {
                    'n_estimators': 120,
                    'max_depth': 7,
                    'min_samples_split': 18,
                    'min_samples_leaf': 6,
                    'max_features': 3,
                    'max_samples': 0.85,
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1
                }
            }
        ]
        
        best_model = None
        best_score = float('inf')
        best_result = None
        all_results = []
        
        print(f"\n{'Model':15s} {'Train R2':10s} {'Test R2':10s} {'Overfit':10s} {'RMSE':10s}")
        print("-" * 60)
        
        for config in model_configs:
            model = RandomForestRegressor(**config['params'])
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            overfitting = train_r2 - test_r2
            
            result = {
                'name': config['name'],
                'model': model,
                'config': config['params'],
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'overfitting': overfitting
            }
            all_results.append(result)
            
            # Print results
            status = "OK" if overfitting < 0.1 else "OVERFIT"
            print(f"{config['name']:15s} {train_r2:10.4f} {test_r2:10.4f} "
                  f"{overfitting:10.4f} {test_rmse:10.4f} {status}")
            
            # Score: balance between overfitting and performance
            score = overfitting * 2 + test_rmse
            if score < best_score:
                best_score = score
                best_model = model
                best_result = result
        
        print(f"\nBest model selected: {best_result['name']}")
        
        # Update self.model with the best model
        self.model = best_model
        
        # Update config with best parameters
        for param, value in best_result['config'].items():
            if param not in ['random_state', 'n_jobs']:
                self.config[param] = value
        
        return best_result
    
    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("\nTraining Random Forest model...")
        
        self.model = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 150),
            max_depth=self.config.get('max_depth', 8),
            min_samples_split=self.config.get('min_samples_split', 15),
            min_samples_leaf=self.config.get('min_samples_leaf', 5),
            max_features=self.config.get('max_features', 'sqrt'),
            max_samples=self.config.get('max_samples', 0.9),
            bootstrap=self.config.get('bootstrap', True),
            oob_score=self.config.get('oob_score', True),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', RANDOM_STATE),
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        if self.config.get('oob_score', True) and hasattr(self.model, 'oob_score_'):
            print(f"Out-of-bag R2 score: {self.model.oob_score_:.4f}")
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in self.feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional, for comparison)
            y_train: Training target (optional, for comparison)
        """
        print("\nModel Evaluation:")
        
        # Test set predictions
        y_pred_test = self.model.predict(X_test)
        
        # Calculate test metrics
        self.metrics['test'] = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'explained_variance': explained_variance_score(y_test, y_pred_test)
        }
        
        print("\nTest Set Performance:")
        for metric, value in self.metrics['test'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Training set predictions (to check for overfitting)
        if X_train is not None and y_train is not None:
            y_pred_train = self.model.predict(X_train)
            self.metrics['train'] = {
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            }
            
            print("\nTraining Set Performance:")
            for metric, value in self.metrics['train'].items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            # Check for overfitting
            r2_diff = self.metrics['train']['r2'] - self.metrics['test']['r2']
            if r2_diff > 0.15:
                print(f"\nWarning: Significant overfitting detected (R2 difference: {r2_diff:.4f})")
                print("Recommendations:")
                print("  - Reduce max_depth (current: {})".format(self.config.get('max_depth', 'N/A')))
                print("  - Increase min_samples_split (current: {})".format(self.config.get('min_samples_split', 'N/A')))
                print("  - Increase min_samples_leaf (current: {})".format(self.config.get('min_samples_leaf', 'N/A')))
                print("  - Collect more training data")
            elif r2_diff > 0.1:
                print(f"\nNote: Mild overfitting detected (R2 difference: {r2_diff:.4f})")
            else:
                print(f"\nGood generalization! (R2 difference: {r2_diff:.4f})")
        
        return y_pred_test
    
    def cross_validate(self, X, y):
        """
        Perform cross-validation to assess model stability.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print(f"\nPerforming {self.config['cross_validation_folds']}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=self.config['cross_validation_folds'],
            scoring='r2',
            n_jobs=self.config['n_jobs']
        )
        
        print(f"Cross-validation R2 scores: {cv_scores}")
        print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
    
    def plot_results(self, y_test, y_pred, save_path=None):
        """
        Create visualization plots for model results.
        
        Args:
            y_test: True values
            y_pred: Predicted values
            save_path: Path to save plots (optional)
        """
        # Ensure save directory exists
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Random Forest Model Performance Analysis', fontsize=16)
        
        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual required_tau')
        ax.set_ylabel('Predicted required_tau')
        ax.set_title(f'Actual vs Predicted (R2 = {self.metrics["test"]["r2"]:.4f})')
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted required_tau')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax = axes[0, 2]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.grid(True, alpha=0.3)
        
        # 4. Feature Importance
        ax = axes[1, 0]
        if self.feature_importance is not None:
            ax.barh(self.feature_importance['feature'], self.feature_importance['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3)
        
        # 5. Prediction Error Distribution
        ax = axes[1, 1]
        percent_error = 100 * (y_pred - y_test) / y_test
        ax.hist(percent_error, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Percentage Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Error Distribution')
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.grid(True, alpha=0.3)
        
        # 6. Q-Q Plot
        ax = axes[1, 2]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"\nPlots saved to: {save_path}")
        
        plt.close()
    
    def save_model(self, save_dir):
        """
        Save the trained model, scaler, and metadata.
        
        Args:
            save_dir: Directory to save model artifacts
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = save_dir / f"rf_model_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler if used
        if self.scaler is not None:
            scaler_path = save_dir / f"scaler_{timestamp}.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")
        
        # Save configuration and metrics
        metadata = {
            'timestamp': timestamp,
            'config': self.config,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        metadata_path = save_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler (optional)
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (can be DataFrame or numpy array)
            
        Returns:
            numpy array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)


def main():
    """Main execution function."""
    
    # Define paths
    data_path = "<RVSPEC_ROOT>/CPG-openpilot/core_experiment_results/core_factors_brake_results.csv"
    save_dir = "<RVSPEC_ROOT>/CPG-openpilot/false_positives/models"
    
    # Initialize predictor with optimized configuration
    config = RequiredTauPredictor.get_default_config()
    config.update({
        'perform_grid_search': False,  # Set to True for hyperparameter optimization
        'compare_models': True,  # Compare multiple configurations
        'remove_outliers': True,  # Remove outliers
        'outlier_contamination': 0.05,  # 5% contamination rate
    })
    
    print("="*70)
    print("Random Forest Model Training - Complete Version")
    print("="*70)
    
    predictor = RequiredTauPredictor(config)
    
    # Load and preprocess data
    X, y, df = predictor.load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
    
    # Compare multiple models or perform grid search
    if config['compare_models']:
        best_result = predictor.compare_models(X_train_scaled, y_train, X_test_scaled, y_test)
    elif config['perform_grid_search']:
        best_params = predictor.perform_grid_search(X_train_scaled, y_train)
        predictor.train_model(X_train_scaled, y_train)
    else:
        predictor.train_model(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = predictor.evaluate_model(X_test_scaled, y_test, X_train_scaled, y_train)
    
    # Cross-validation
    predictor.cross_validate(X_train_scaled, y_train)
    
    # Visualize results
    plot_path = Path(save_dir) / "model_performance.png"
    predictor.plot_results(y_test, y_pred, save_path=plot_path)
    
    # Save model
    model_path, metadata_path = predictor.save_model(save_dir)
    
    # Example predictions
    print("\n" + "="*70)
    print("Example Predictions on Test Data (first 5 samples):")
    print("="*70)
    
    for i in range(min(5, len(X_test))):
        sample = X_test[i:i+1]
        prediction = predictor.predict(sample)[0]
        actual = y_test[i]
        error = abs(prediction - actual)
        error_percent = 100 * error / actual
        
        print(f"\nSample {i+1}:")
        print(f"  Predicted required_tau: {prediction:.4f}")
        print(f"  Actual required_tau: {actual:.4f}")
        print(f"  Error: {error:.4f} ({error_percent:.2f}%)")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()