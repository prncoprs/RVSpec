#!/usr/bin/env python3
"""
ArduPilot EKF vs Ground Truth Analysis
Compare estimated EKF results with simulation ground truth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class ArduPilotAnalyzer:
    def __init__(self, data_dir):
        """
        Initialize analyzer with data directory containing flight log folders
        
        Args:
            data_dir: Path to directory containing flight log folders
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def load_flight_data(self, flight_folder):
        """
        Load and synchronize data from a single flight folder
        
        Args:
            flight_folder: Path to flight folder containing CSV files
            
        Returns:
            dict: Synchronized dataframes for analysis
        """
        folder_path = Path(flight_folder)
        
        try:
            # Load ground truth data
            sim_df = pd.read_csv(folder_path / 'SIM.csv')
            sim2_df = pd.read_csv(folder_path / 'SIM2.csv')
            
            # Load EKF estimate data
            xkf1_df = pd.read_csv(folder_path / 'XKF1.csv')
            xkq_df = pd.read_csv(folder_path / 'XKQ.csv')
            
            # Filter for primary EKF core (C=0)
            xkf1_primary = xkf1_df[xkf1_df['C'] == 0].copy()
            xkq_primary = xkq_df[xkq_df['C'] == 0].copy()
            
            # Synchronize timestamps using TimeUS (microseconds)
            data = self.synchronize_data(sim_df, sim2_df, xkf1_primary, xkq_primary)
            
            return data
            
        except FileNotFoundError as e:
            print(f"Missing file in {flight_folder}: {e}")
            return None
        except Exception as e:
            print(f"Error processing {flight_folder}: {e}")
            return None
    
    def synchronize_data(self, sim_df, sim2_df, xkf1_df, xkq_df):
        """
        Synchronize all dataframes by timestamp for comparison
        """
        # Use TimeUS as the common timestamp
        dfs = {
            'sim': sim_df[['TimeUS', 'Roll', 'Pitch', 'Yaw', 'Q1', 'Q2', 'Q3', 'Q4']],
            'sim2': sim2_df[['TimeUS', 'PN', 'PE', 'PD', 'VN', 'VE', 'VD']],
            'xkf1': xkf1_df[['TimeUS', 'Roll', 'Pitch', 'Yaw', 'VN', 'VE', 'VD', 'PN', 'PE', 'PD']],
            'xkq': xkq_df[['TimeUS', 'Q1', 'Q2', 'Q3', 'Q4']]
        }
        
        # Find common time range
        min_time = max(df['TimeUS'].min() for df in dfs.values())
        max_time = min(df['TimeUS'].max() for df in dfs.values())
        
        # Filter to common time range
        for key in dfs:
            dfs[key] = dfs[key][(dfs[key]['TimeUS'] >= min_time) & 
                               (dfs[key]['TimeUS'] <= max_time)]
        
        # Create synchronized dataset using nearest neighbor interpolation
        base_time = dfs['xkf1']['TimeUS'].values
        
        synchronized_data = pd.DataFrame({'TimeUS': base_time})
        
        # Add ground truth data (interpolated to EKF timestamps)
        for col in ['Roll', 'Pitch', 'Yaw']:
            synchronized_data[f'GT_{col}'] = np.interp(base_time, dfs['sim']['TimeUS'], dfs['sim'][col])
            synchronized_data[f'EKF_{col}'] = dfs['xkf1'][col].values
        
        # Add position data
        for col in ['PN', 'PE', 'PD']:
            synchronized_data[f'GT_{col}'] = np.interp(base_time, dfs['sim2']['TimeUS'], dfs['sim2'][col])
            synchronized_data[f'EKF_{col}'] = dfs['xkf1'][col].values
        
        # Add velocity data
        for col in ['VN', 'VE', 'VD']:
            synchronized_data[f'GT_{col}'] = np.interp(base_time, dfs['sim2']['TimeUS'], dfs['sim2'][col])
            synchronized_data[f'EKF_{col}'] = dfs['xkf1'][col].values
        
        # Add quaternion data
        for col in ['Q1', 'Q2', 'Q3', 'Q4']:
            synchronized_data[f'GT_{col}'] = np.interp(base_time, dfs['sim']['TimeUS'], dfs['sim'][col])
            synchronized_data[f'EKF_{col}'] = np.interp(base_time, dfs['xkq']['TimeUS'], dfs['xkq'][col])
        
        return synchronized_data
    
    def calculate_metrics(self, data):
        """
        Calculate comparison metrics between EKF estimates and ground truth
        """
        metrics = {}
        
        # Define variable groups
        variables = {
            'Attitude': ['Roll', 'Pitch', 'Yaw'],
            'Position': ['PN', 'PE', 'PD'],
            'Velocity': ['VN', 'VE', 'VD'],
            'Quaternion': ['Q1', 'Q2', 'Q3', 'Q4']
        }
        
        for group, vars_list in variables.items():
            metrics[group] = {}
            
            for var in vars_list:
                gt_col = f'GT_{var}'
                ekf_col = f'EKF_{var}'
                
                if gt_col in data.columns and ekf_col in data.columns:
                    gt_data = data[gt_col].values
                    ekf_data = data[ekf_col].values
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(gt_data) | np.isnan(ekf_data))
                    gt_clean = gt_data[mask]
                    ekf_clean = ekf_data[mask]
                    
                    if len(gt_clean) > 0:
                        metrics[group][var] = {
                            'RMSE': np.sqrt(mean_squared_error(gt_clean, ekf_clean)),
                            'MAE': mean_absolute_error(gt_clean, ekf_clean),
                            'Correlation': stats.pearsonr(gt_clean, ekf_clean)[0],
                            'Mean_Error': np.mean(ekf_clean - gt_clean),
                            'Std_Error': np.std(ekf_clean - gt_clean),
                            'Max_Error': np.max(np.abs(ekf_clean - gt_clean))
                        }
        
        return metrics
    
    def plot_comparison(self, data, flight_name, save_path=None):
        """
        Create comparison plots for all variables
        """
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle(f'EKF vs Ground Truth Comparison - {flight_name}', fontsize=16)
        
        # Time vector (convert to seconds)
        time_s = (data['TimeUS'] - data['TimeUS'].iloc[0]) / 1e6
        
        # Plot attitude
        ax = axes[0]
        for var in ['Roll', 'Pitch', 'Yaw']:
            ax.plot(time_s, data[f'GT_{var}'], label=f'GT {var}', linestyle='--', alpha=0.7)
            ax.plot(time_s, data[f'EKF_{var}'], label=f'EKF {var}', linewidth=2)
        ax.set_title('Attitude Comparison')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot position
        ax = axes[1]
        for var in ['PN', 'PE', 'PD']:
            ax.plot(time_s, data[f'GT_{var}'], label=f'GT {var}', linestyle='--', alpha=0.7)
            ax.plot(time_s, data[f'EKF_{var}'], label=f'EKF {var}', linewidth=2)
        ax.set_title('Position Comparison (NED)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot velocity
        ax = axes[2]
        for var in ['VN', 'VE', 'VD']:
            ax.plot(time_s, data[f'GT_{var}'], label=f'GT {var}', linestyle='--', alpha=0.7)
            ax.plot(time_s, data[f'EKF_{var}'], label=f'EKF {var}', linewidth=2)
        ax.set_title('Velocity Comparison (NED)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot quaternions
        ax = axes[3]
        for var in ['Q1', 'Q2', 'Q3', 'Q4']:
            ax.plot(time_s, data[f'GT_{var}'], label=f'GT {var}', linestyle='--', alpha=0.7)
            ax.plot(time_s, data[f'EKF_{var}'], label=f'EKF {var}', linewidth=2)
        ax.set_title('Quaternion Comparison')
        ax.set_ylabel('Quaternion Components')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_single_flight(self, flight_folder, plot=True):
        """
        Analyze a single flight and return metrics
        """
        print(f"Analyzing flight: {flight_folder}")
        
        # Load and synchronize data
        data = self.load_flight_data(flight_folder)
        if data is None:
            return None
        
        # Calculate metrics
        metrics = self.calculate_metrics(data)
        
        # Create plots if requested
        if plot:
            flight_name = Path(flight_folder).name
            self.plot_comparison(data, flight_name)
        
        return {
            'flight_name': Path(flight_folder).name,
            'metrics': metrics,
            'data_points': len(data)
        }
    
    def export_all_relationships(self, output_dir="relationship_data", pattern="*gazebo*"):
        """
        Export all flight relationships to CSV files and fit models
        
        Args:
            output_dir: Directory to save CSV files and models
            pattern: Pattern to match flight folders
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        flight_folders = list(self.data_dir.glob(pattern))
        print(f"Processing {len(flight_folders)} flights matching pattern '{pattern}'")
        
        # Dictionary to store all data points across flights
        all_relationships = {
            'Attitude': {'Roll': {'GT': [], 'EKF': [], 'Flight': []},
                        'Pitch': {'GT': [], 'EKF': [], 'Flight': []},
                        'Yaw': {'GT': [], 'EKF': [], 'Flight': []}},
            'Position': {'PN': {'GT': [], 'EKF': [], 'Flight': []},
                        'PE': {'GT': [], 'EKF': [], 'Flight': []},
                        'PD': {'GT': [], 'EKF': [], 'Flight': []}},
            'Velocity': {'VN': {'GT': [], 'EKF': [], 'Flight': []},
                        'VE': {'GT': [], 'EKF': [], 'Flight': []},
                        'VD': {'GT': [], 'EKF': [], 'Flight': []}},
            'Quaternion': {'Q1': {'GT': [], 'EKF': [], 'Flight': []},
                          'Q2': {'GT': [], 'EKF': [], 'Flight': []},
                          'Q3': {'GT': [], 'EKF': [], 'Flight': []},
                          'Q4': {'GT': [], 'EKF': [], 'Flight': []}}
        }
        
        # Process each flight
        processed_count = 0
        for i, folder in enumerate(flight_folders):
            print(f"Processing flight {i+1}/{len(flight_folders)}: {folder.name}")
            
            # Load flight data
            data = self.load_flight_data(folder)
            if data is None:
                continue
            
            flight_name = folder.name
            
            # Collect data points for each variable
            for group in all_relationships:
                for var in all_relationships[group]:
                    gt_col = f'GT_{var}'
                    ekf_col = f'EKF_{var}'
                    
                    if gt_col in data.columns and ekf_col in data.columns:
                        # Remove NaN values
                        mask = ~(np.isnan(data[gt_col]) | np.isnan(data[ekf_col]))
                        gt_values = data[gt_col][mask].values
                        ekf_values = data[ekf_col][mask].values
                        
                        # Store data points
                        all_relationships[group][var]['GT'].extend(gt_values)
                        all_relationships[group][var]['EKF'].extend(ekf_values)
                        all_relationships[group][var]['Flight'].extend([flight_name] * len(gt_values))
            
            processed_count += 1
        
        print(f"Successfully processed {processed_count} flights")
        
        # Export to CSV and fit models
        models = {}
        for group in all_relationships:
            models[group] = {}
            group_path = output_path / group.lower()
            group_path.mkdir(exist_ok=True)
            
            for var in all_relationships[group]:
                if len(all_relationships[group][var]['GT']) > 0:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Ground_Truth': all_relationships[group][var]['GT'],
                        'EKF_Estimate': all_relationships[group][var]['EKF'],
                        'Flight_Name': all_relationships[group][var]['Flight']
                    })
                    
                    # Save to CSV
                    csv_path = group_path / f"{var}_relationship.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"Saved {len(df)} data points to {csv_path}")
                    
                    # Fit models
                    models[group][var] = self.fit_relationship_models(df, var, group_path)
        
        # Save model summary
        self.save_model_summary(models, output_path)
        
        return models, all_relationships
    
    def fit_relationship_models(self, df, var_name, save_path):
        """
        Fit fast linear models to find relationship f(EKF_estimate) = Ground_Truth
        
        Args:
            df: DataFrame with Ground_Truth and EKF_Estimate columns
            var_name: Variable name for saving
            save_path: Path to save model files
            
        Returns:
            dict: Fitted models and their performance metrics
        """
        print(f"  Fitting linear models for {var_name}...")
        
        X = df['EKF_Estimate'].values.reshape(-1, 1)
        y = df['Ground_Truth'].values
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {}
        
        try:
            # 1. Simple Linear: f(x) = ax + b
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred = linear_model.predict(X_test)
            
            models['linear'] = {
                'model': linear_model,
                'scaler': None,
                'formula': f"f(x) = {linear_model.coef_[0]:.6f}*x + {linear_model.intercept_:.6f}",
                'r2_score': linear_model.score(X_test, y_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'cv_score': np.mean(cross_val_score(linear_model, X_train, y_train, cv=5))
            }
        except Exception as e:
            print(f"    Warning: Linear model failed for {var_name}: {e}")
        
        try:
            # 2. Polynomial degree 2: f(x) = ax² + bx + c
            poly2_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            poly2_model.fit(X_train, y_train)
            y_pred = poly2_model.predict(X_test)
            
            # Extract coefficients
            poly_features = PolynomialFeatures(2)
            X_poly = poly_features.fit_transform(X_train)
            lr = LinearRegression()
            lr.fit(X_poly, y_train)
            coefs = lr.coef_
            intercept = lr.intercept_
            
            formula = f"f(x) = {coefs[2]:.6f}*x² + {coefs[1]:.6f}*x + {intercept:.6f}"
            
            models['polynomial_2'] = {
                'model': poly2_model,
                'scaler': None,
                'formula': formula,
                'r2_score': poly2_model.score(X_test, y_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'cv_score': np.mean(cross_val_score(poly2_model, X_train, y_train, cv=5))
            }
        except Exception as e:
            print(f"    Warning: Polynomial model failed for {var_name}: {e}")
        
        try:
            # 3. Polynomial degree 3: f(x) = ax³ + bx² + cx + d
            poly3_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly3_model.fit(X_train, y_train)
            y_pred = poly3_model.predict(X_test)
            
            # Extract coefficients
            poly_features = PolynomialFeatures(3)
            X_poly = poly_features.fit_transform(X_train)
            lr = LinearRegression()
            lr.fit(X_poly, y_train)
            coefs = lr.coef_
            intercept = lr.intercept_
            
            formula = f"f(x) = {coefs[3]:.6f}*x³ + {coefs[2]:.6f}*x² + {coefs[1]:.6f}*x + {intercept:.6f}"
            
            models['polynomial_3'] = {
                'model': poly3_model,
                'scaler': None,
                'formula': formula,
                'r2_score': poly3_model.score(X_test, y_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'cv_score': np.mean(cross_val_score(poly3_model, X_train, y_train, cv=5))
            }
        except Exception as e:
            print(f"    Warning: Polynomial degree 3 model failed for {var_name}: {e}")
        
        # 4. Identity model for comparison
        y_pred_identity = X_test.flatten()
        models['identity'] = {
            'model': None,
            'scaler': None,
            'formula': "f(x) = x",
            'r2_score': stats.pearsonr(y_test, y_pred_identity)[0]**2 if len(y_test) > 1 else 0,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_identity)),
            'mae': mean_absolute_error(y_test, y_pred_identity),
            'cv_score': 0
        }
        
        # Save models
        model_path = save_path / f"{var_name}_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"    Saved {len(models)} models to {model_path}")
        
        # Create comparison plot
        self.plot_model_comparison(df, models, var_name, save_path)
        
        return models
    
    def plot_model_comparison(self, df, models, var_name, save_path):
        """
        Plot model comparisons and relationships - save as PDF
        """
        # Set up matplotlib for PDF output
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 8
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Advanced Model Comparison for {var_name}', fontsize=14, fontweight='bold')
        
        X = df['EKF_Estimate'].values
        y = df['Ground_Truth'].values
        
        # Sample data for plotting if too many points
        if len(X) > 10000:
            indices = np.random.choice(len(X), 10000, replace=False)
            X_plot = X[indices]
            y_plot = y[indices]
        else:
            X_plot = X
            y_plot = y
        
        # Create prediction range
        X_range = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        
        # Plot 1: Scatter plot with best models
        ax = axes[0, 0]
        ax.scatter(X_plot, y_plot, alpha=0.1, s=0.5, color='lightgray', label=f'Data ({len(X):,} points)')
        
        # Sort models by R² score and plot top 5
        sorted_models = sorted(models.items(), key=lambda x: x[1]['r2_score'], reverse=True)[:5]
        colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_models)))
        
        for i, (model_name, model_info) in enumerate(sorted_models):
            try:
                if model_info['model'] is not None:
                    if model_info['scaler'] is not None:
                        X_range_scaled = model_info['scaler'].transform(X_range)
                        y_pred_range = model_info['model'].predict(X_range_scaled)
                    else:
                        y_pred_range = model_info['model'].predict(X_range)
                    
                    ax.plot(X_range, y_pred_range, color=colors[i], linewidth=2, 
                           label=f"{model_name.replace('_', ' ').title()}: R²={model_info['r2_score']:.4f}")
                else:  # Identity model
                    ax.plot(X_range, X_range, color=colors[i], linewidth=2, 
                           label=f"{model_name.replace('_', ' ').title()}: R²={model_info['r2_score']:.4f}")
            except Exception as e:
                print(f"    Warning: Could not plot {model_name}: {e}")
        
        ax.set_xlabel('EKF Estimate')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Top 5 Model Fits')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals for best model
        best_model_name, best_model = sorted_models[0]
        
        ax = axes[0, 1]
        try:
            if best_model['model'] is not None:
                if best_model['scaler'] is not None:
                    X_scaled = best_model['scaler'].transform(X.reshape(-1, 1))
                    y_pred = best_model['model'].predict(X_scaled)
                else:
                    y_pred = best_model['model'].predict(X.reshape(-1, 1))
            else:
                y_pred = X
            
            residuals = y - y_pred
            
            # Sample residuals for plotting
            if len(residuals) > 10000:
                indices = np.random.choice(len(residuals), 10000, replace=False)
                y_pred_plot = y_pred[indices]
                residuals_plot = residuals[indices]
            else:
                y_pred_plot = y_pred
                residuals_plot = residuals
            
            ax.scatter(y_pred_plot, residuals_plot, alpha=0.1, s=0.5, color='blue')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals - {best_model_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Add residual statistics
            residual_std = np.std(residuals)
            ax.text(0.05, 0.95, f'Residual Std: {residual_std:.6f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting residuals: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
        
        # Plot 3: Model performance comparison (R² scores)
        ax = axes[1, 0]
        model_names = [name.replace('_', ' ').title() for name, _ in sorted_models]
        r2_scores = [info['r2_score'] for _, info in sorted_models]
        
        bars = ax.barh(model_names, r2_scores, alpha=0.7)
        ax.set_xlabel('R² Score')
        ax.set_title('Model Performance (R² Score)')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', ha='left', va='center', fontsize=8)
        
        # Plot 4: Model performance comparison (RMSE)
        ax = axes[1, 1]
        rmse_scores = [info['rmse'] for _, info in sorted_models]
        
        bars = ax.barh(model_names, rmse_scores, alpha=0.7, color='orange')
        ax.set_xlabel('RMSE')
        ax.set_title('Model Performance (RMSE - Lower is Better)')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, rmse_scores):
            ax.text(bar.get_width() + max(rmse_scores)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', ha='left', va='center', fontsize=8)
        
        # Plot 5: Cross-validation scores
        ax = axes[2, 0]
        cv_scores = [info['cv_score'] for _, info in sorted_models]
        
        bars = ax.barh(model_names, cv_scores, alpha=0.7, color='green')
        ax.set_xlabel('CV Score (5-fold)')
        ax.set_title('Cross-Validation Performance')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, cv_scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', ha='left', va='center', fontsize=8)
        
        # Plot 6: Model formulas and statistics
        ax = axes[2, 1]
        ax.axis('off')
        
        formula_text = f"Best Models for {var_name}:\n" + "="*50 + "\n\n"
        
        for i, (model_name, model_info) in enumerate(sorted_models[:3]):  # Top 3 models
            formula_text += f"{i+1}. {model_name.replace('_', ' ').title()}:\n"
            formula_text += f"   Formula: {model_info['formula']}\n"
            formula_text += f"   R² = {model_info['r2_score']:.6f}\n"
            formula_text += f"   RMSE = {model_info['rmse']:.6f}\n"
            formula_text += f"   MAE = {model_info['mae']:.6f}\n"
            formula_text += f"   CV Score = {model_info['cv_score']:.6f}\n\n"
        
        # Add data statistics
        formula_text += f"Data Statistics:\n"
        formula_text += f"   Total Points: {len(X):,}\n"
        formula_text += f"   EKF Range: [{X.min():.3f}, {X.max():.3f}]\n"
        formula_text += f"   GT Range: [{y.min():.3f}, {y.max():.3f}]\n"
        formula_text += f"   Correlation: {np.corrcoef(X, y)[0,1]:.6f}\n"
        
        ax.text(0.05, 0.95, formula_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save as PDF
        pdf_path = save_path / f"{var_name}_model_comparison.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved model comparison plot to {pdf_path}")
        
        # Reset matplotlib settings
        plt.rcParams.update(plt.rcParamsDefault)
    
    def save_model_summary(self, models, output_path):
        """
        Save a summary of all fitted models to CSV
        """
        summary_data = []
        
        for group in models:
            for var in models[group]:
                for model_name, model_info in models[group][var].items():
                    summary_data.append({
                        'Group': group,
                        'Variable': var,
                        'Model_Type': model_name,
                        'Formula': model_info['formula'],
                        'R2_Score': model_info['r2_score'],
                        'RMSE': model_info['rmse'],
                        'MAE': model_info['mae'],
                        'CV_Score': model_info['cv_score']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_path / "model_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nModel summary saved to {summary_path}")
        print("\nBest models by R² score:")
        print("-" * 100)
        
        # Show best model for each variable
        for group in models:
            for var in models[group]:
                best_model_name = max(models[group][var].keys(), 
                                    key=lambda k: models[group][var][k]['r2_score'])
                best_model = models[group][var][best_model_name]
                
                print(f"{group:12} {var:4}: {best_model_name:18} - "
                      f"R²={best_model['r2_score']:.6f} - CV={best_model['cv_score']:.6f}")
        
        # Create additional model loading utility
        self.create_model_loader(output_path)
    
    def create_model_loader(self, output_path):
        """
        Create a utility script for loading and using the trained models
        """
        loader_script = '''#!/usr/bin/env python3
"""
Model Loader Utility for ArduPilot EKF Correction

This script loads the trained models and provides functions to correct EKF estimates.
"""

import pickle
import numpy as np
from pathlib import Path

class EKFCorrector:
    def __init__(self, model_dir):
        """
        Initialize the EKF corrector with trained models
        
        Args:
            model_dir: Directory containing the trained model files
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models from pickle files"""
        model_files = list(self.model_dir.glob("*/*_models.pkl"))
        
        for model_file in model_files:
            group = model_file.parent.name
            var_name = model_file.stem.replace("_models", "")
            
            if group not in self.models:
                self.models[group] = {}
            
            with open(model_file, 'rb') as f:
                self.models[group][var_name] = pickle.load(f)
            
            print(f"Loaded models for {group}/{var_name}")
    
    def get_best_model(self, group, variable):
        """Get the best performing model for a variable"""
        if group not in self.models or variable not in self.models[group]:
            raise ValueError(f"No models found for {group}/{variable}")
        
        models = self.models[group][variable]
        best_model_name = max(models.keys(), key=lambda k: models[k]['r2_score'])
        return best_model_name, models[best_model_name]
    
    def correct_estimate(self, group, variable, ekf_value, use_best=True, model_name=None):
        """
        Correct an EKF estimate using the trained model
        
        Args:
            group: Variable group ('attitude', 'position', 'velocity', 'quaternion')
            variable: Variable name ('Roll', 'PN', 'VN', etc.)
            ekf_value: EKF estimated value to correct
            use_best: Whether to use the best performing model
            model_name: Specific model to use (if use_best=False)
            
        Returns:
            Corrected value
        """
        if use_best:
            model_name, model_info = self.get_best_model(group, variable)
        else:
            if model_name not in self.models[group][variable]:
                raise ValueError(f"Model {model_name} not found for {group}/{variable}")
            model_info = self.models[group][variable][model_name]
        
        # Prepare input
        X = np.array([[ekf_value]])
        
        # Apply scaling if needed
        if model_info['scaler'] is not None:
            X = model_info['scaler'].transform(X)
        
        # Make prediction
        if model_info['model'] is not None:
            corrected_value = model_info['model'].predict(X)[0]
        else:  # Identity model
            corrected_value = ekf_value
        
        return corrected_value
    
    def correct_batch(self, group, variable, ekf_values, use_best=True, model_name=None):
        """
        Correct a batch of EKF estimates
        
        Args:
            group: Variable group
            variable: Variable name
            ekf_values: Array of EKF estimated values
            use_best: Whether to use the best performing model
            model_name: Specific model to use
            
        Returns:
            Array of corrected values
        """
        ekf_values = np.array(ekf_values)
        corrected_values = np.zeros_like(ekf_values)
        
        for i, value in enumerate(ekf_values):
            corrected_values[i] = self.correct_estimate(group, variable, value, use_best, model_name)
        
        return corrected_values
    
    def list_available_models(self):
        """List all available models and their performance"""
        print("Available Models:")
        print("=" * 80)
        
        for group in self.models:
            print(f"\n{group.upper()}:")
            for variable in self.models[group]:
                print(f"  {variable}:")
                for model_name, model_info in self.models[group][variable].items():
                    print(f"    {model_name:20}: R²={model_info['r2_score']:.6f}, "
                          f"RMSE={model_info['rmse']:.6f}")

# Example usage
if __name__ == "__main__":
    # Initialize corrector
    corrector = EKFCorrector("relationship_data")
    
    # List available models
    corrector.list_available_models()
    
    # Example corrections
    # corrected_roll = corrector.correct_estimate('attitude', 'Roll', 1.5)
    # corrected_position = corrector.correct_estimate('position', 'PN', 100.0)
    
    print("\\nEKF Corrector ready for use!")
'''
        
        loader_path = output_path / "ekf_corrector.py"
        with open(loader_path, 'w') as f:
            f.write(loader_script)
        
        print(f"Model loader utility saved to {loader_path}")

    def analyze_all_flights(self, pattern="*gazebo*", max_flights=None):
        """
        Analyze multiple flights and generate summary statistics
        """
        flight_folders = list(self.data_dir.glob(pattern))
        if max_flights:
            flight_folders = flight_folders[:max_flights]
            
        print(f"Found {len(flight_folders)} flight folders matching pattern '{pattern}'")
        
        all_results = []
        
        for folder in flight_folders:
            result = self.analyze_single_flight(folder, plot=False)
            if result:
                all_results.append(result)
        
        # Generate summary statistics
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """
        Generate summary statistics across all flights
        """
        if not results:
            print("No valid results to summarize")
            return
        
        print("\n" + "="*80)
        print("SUMMARY REPORT - EKF vs Ground Truth Analysis")
        print("="*80)
        
        # Collect metrics across all flights
        all_metrics = {}
        
        for result in results:
            for group, group_metrics in result['metrics'].items():
                if group not in all_metrics:
                    all_metrics[group] = {}
                
                for var, var_metrics in group_metrics.items():
                    if var not in all_metrics[group]:
                        all_metrics[group][var] = {metric: [] for metric in var_metrics.keys()}
                    
                    for metric_name, metric_value in var_metrics.items():
                        all_metrics[group][var][metric_name].append(metric_value)
        
        # Print summary statistics
        for group in all_metrics:
            print(f"\n{group.upper()} METRICS:")
            print("-" * 60)
            
            for var in all_metrics[group]:
                print(f"\n{var}:")
                for metric in ['RMSE', 'MAE', 'Correlation', 'Max_Error']:
                    if metric in all_metrics[group][var]:
                        values = all_metrics[group][var][metric]
                        print(f"  {metric:12}: Mean={np.mean(values):.6f}, "
                              f"Std={np.std(values):.6f}, "
                              f"Min={np.min(values):.6f}, "
                              f"Max={np.max(values):.6f}")

# Example usage
def main():
    """
    Main analysis function
    """
    # Set your data directory path here
    data_dir = "<RVSPEC_ROOT>/CPG-ArduPilot/DataAnalysis/data_gazebo"  # Update this path
    
    # Initialize analyzer
    analyzer = ArduPilotAnalyzer(data_dir)
    
    # Option 1: Analyze a single flight
    # single_flight = "20250520-gazebo-multi-00000001"
    # result = analyzer.analyze_single_flight(data_dir / single_flight)
    
    # Option 2: Export all relationships and fit models
    print("Exporting all flight relationships and fitting models...")
    models, relationships = analyzer.export_all_relationships(
        output_dir="relationship_data", 
        pattern="*single*"  # Process all gazebo flights
    )
    
    # Option 3: Quick analysis summary (optional)
    # results = analyzer.analyze_all_flights(pattern="*gazebo-multi*", max_flights=3)
    
    print(f"\nAnalysis complete!")
    print(f"- Relationship data saved in 'relationship_data/' directory")
    print(f"- Model files and plots generated for each variable")
    print(f"- Check 'model_summary.csv' for best model formulas")

if __name__ == "__main__":
    main()