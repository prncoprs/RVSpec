#!/usr/bin/env python3
"""
Evaluation script for comparing theoretical braking time vs Random Forest tau predictions
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def theoretical_tau_pure_braking(initial_speed_kmh):
    """
    τ - 
    
    """
    v0 = initial_speed_kmh / 3.6  # km/hm/s
    STANDARD_DECELERATION = 7.0  # m/s²
    
    braking_time = v0 / STANDARD_DECELERATION
    return braking_time

def load_models_and_data():
    """"""
    base_path = Path("<RVSPEC_ROOT>/CPG-openpilot/false_positives")
    model_path = base_path / "models/rf_model_20250914_012014.pkl"
    scaler_path = base_path / "models/scaler_20250914_012014.pkl"
    data_path = base_path.parent / "core_experiment_results/core_factors_brake_results.csv"
    
    print("Loading Random Forest model...")
    rf_model = joblib.load(model_path)
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)
    
    print("Loading test data...")
    df = pd.read_csv(data_path)
    df_success = df[df['success'] == True].copy()
    print(f"Loaded {len(df_success)} successful experiments")
    
    return rf_model, scaler, df_success

def evaluate_tau_predictions(df, rf_model, scaler):
    """tau"""
    feature_columns = [
        'applied_mass',
        'applied_drag_coefficient',
        'applied_tire_friction',
        'applied_brake_torque_front',
        'applied_brake_torque_rear',
        'applied_road_friction'
    ]
    
    results = []
    
    for idx, row in df.iterrows():
        # 1. 
        tau_theoretical = theoretical_tau_pure_braking(row['brake_start_speed'])
        
        # 2. Random Forest
        features = row[feature_columns].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        rf_tau = rf_model.predict(features_scaled)[0]
        
        # 3. required_tau
        actual_tau = row['required_tau']
        
        results.append({
            'brake_start_speed': row['brake_start_speed'],
            'actual_tau': actual_tau,
            'theoretical_pure_braking': tau_theoretical,
            'random_forest': rf_tau,
            'unsafe_theoretical': tau_theoretical < actual_tau,
            'unsafe_rf': rf_tau < actual_tau
        })
    
    return pd.DataFrame(results)

def analyze_results(results_df, output_dir):
    """"""
    n_samples = len(results_df)
    
    # 
    unsafe_theoretical = results_df['unsafe_theoretical'].sum()
    unsafe_rf = results_df['unsafe_rf'].sum()
    
    # 
    errors_theoretical = results_df['theoretical_pure_braking'] - results_df['actual_tau']
    errors_rf = results_df['random_forest'] - results_df['actual_tau']
    
    metrics = {
        'Theoretical (Pure Braking)': {
            'MAE': np.abs(errors_theoretical).mean(),
            'RMSE': np.sqrt((errors_theoretical**2).mean()),
            'Unsafe Count': unsafe_theoretical,
            'Unsafe Rate (%)': unsafe_theoretical / n_samples * 100,
            'Mean Error': errors_theoretical.mean(),
            'Std Error': errors_theoretical.std()
        },
        'Random Forest': {
            'MAE': np.abs(errors_rf).mean(),
            'RMSE': np.sqrt((errors_rf**2).mean()),
            'Unsafe Count': unsafe_rf,
            'Unsafe Rate (%)': unsafe_rf / n_samples * 100,
            'Mean Error': errors_rf.mean(),
            'Std Error': errors_rf.std()
        }
    }
    
    # 
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("TAU PREDICTION COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total samples evaluated: {n_samples}")
    report_lines.append("")
    report_lines.append("-"*80)
    report_lines.append(f"{'Method':<30} {'MAE':>8} {'RMSE':>8} {'Unsafe':>8} {'Unsafe%':>10}")
    report_lines.append("-"*80)
    
    for method, metric in metrics.items():
        report_lines.append(f"{method:<30} {metric['MAE']:>8.4f} {metric['RMSE']:>8.4f} "
                          f"{metric['Unsafe Count']:>8d} {metric['Unsafe Rate (%)']:>9.1f}%")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS:")
    report_lines.append("="*80)
    
    if unsafe_theoretical > 0:
        reduction = (unsafe_theoretical - unsafe_rf) / unsafe_theoretical * 100
        report_lines.append(f"\n1. Random Forest reduces unsafe predictions by {reduction:.1f}%")
        report_lines.append(f"   From {unsafe_theoretical} to {unsafe_rf} cases")
    
    report_lines.append(f"\n2. Theoretical pure braking time is unsafe in {unsafe_theoretical}/{n_samples} cases")
    report_lines.append(f"   This represents {unsafe_theoretical/n_samples*100:.1f}% of all scenarios")
    
    report_lines.append(f"\n3. Random Forest achieves {unsafe_rf}/{n_samples} unsafe predictions")
    report_lines.append(f"   Only {unsafe_rf/n_samples*100:.1f}% unsafe rate")
    
    report_lines.append("\n4. Random Forest advantages:")
    report_lines.append("   - Considers actual vehicle mass")
    report_lines.append("   - Accounts for tire and road friction")
    report_lines.append("   - Includes brake torque distribution")
    report_lines.append("   - Models aerodynamic effects")
    
    # 
    report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")
    
    return metrics

def create_visualizations(results_df, output_dir):
    """"""
    fig = plt.figure(figsize=(14, 8))
    
    # 1.  - 
    ax1 = plt.subplot(2, 3, 1)
    actual = results_df['actual_tau']
    theoretical = results_df['theoretical_pure_braking']
    unsafe_theo = results_df['unsafe_theoretical']
    
    ax1.scatter(actual[~unsafe_theo], theoretical[~unsafe_theo], alpha=0.5, label='Safe', color='blue', s=20)
    ax1.scatter(actual[unsafe_theo], theoretical[unsafe_theo], alpha=0.5, label='Unsafe', color='red', s=20)
    
    min_val = min(actual.min(), theoretical.min())
    max_val = max(actual.max(), theoretical.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
    
    ax1.set_xlabel('Actual Required τ (s)')
    ax1.set_ylabel('Predicted τ (s)')
    ax1.set_title(f'Theoretical Pure Braking\nUnsafe: {unsafe_theo.sum()}/{len(results_df)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2.  - Random Forest
    ax2 = plt.subplot(2, 3, 2)
    rf_pred = results_df['random_forest']
    unsafe_rf = results_df['unsafe_rf']
    
    ax2.scatter(actual[~unsafe_rf], rf_pred[~unsafe_rf], alpha=0.5, label='Safe', color='green', s=20)
    ax2.scatter(actual[unsafe_rf], rf_pred[unsafe_rf], alpha=0.5, label='Unsafe', color='red', s=20)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
    
    ax2.set_xlabel('Actual Required τ (s)')
    ax2.set_ylabel('Predicted τ (s)')
    ax2.set_title(f'Random Forest\nUnsafe: {unsafe_rf.sum()}/{len(results_df)}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 
    ax3 = plt.subplot(2, 3, 3)
    errors_theo = results_df['theoretical_pure_braking'] - results_df['actual_tau']
    errors_rf = results_df['random_forest'] - results_df['actual_tau']
    
    ax3.hist(errors_theo, bins=30, alpha=0.5, label='Theoretical', color='orange')
    ax3.hist(errors_rf, bins=30, alpha=0.5, label='Random Forest', color='green')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error (s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 
    ax4 = plt.subplot(2, 3, 4)
    unsafe_rates = [
        results_df['unsafe_theoretical'].mean() * 100,
        results_df['unsafe_rf'].mean() * 100
    ]
    methods = ['Theoretical\nPure Braking', 'Random\nForest']
    colors = ['orange', 'green']
    bars = ax4.bar(methods, unsafe_rates, color=colors)
    
    for bar, rate in zip(bars, unsafe_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    ax4.set_ylabel('Unsafe Prediction Rate (%)')
    ax4.set_title('Unsafe Predictions Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. vs
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(results_df['brake_start_speed'], errors_theo, alpha=0.3, label='Theoretical', color='orange', s=20)
    ax5.scatter(results_df['brake_start_speed'], errors_rf, alpha=0.3, label='RF', color='green', s=20)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax5.set_xlabel('Initial Speed (km/h)')
    ax5.set_ylabel('Prediction Error (s)')
    ax5.set_title('Error vs Initial Speed')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 
    ax6 = plt.subplot(2, 3, 6)
    error_data = pd.DataFrame({
        'Theoretical': errors_theo,
        'Random Forest': errors_rf
    })
    error_data.boxplot(ax=ax6)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax6.set_ylabel('Prediction Error (s)')
    ax6.set_title('Error Distribution Comparison')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 
    plot_path = output_dir / f"tau_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    plt.show()

def main():
    """"""
    # 
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("TAU PREDICTION EVALUATION: Theoretical vs Random Forest")
    print("="*80)
    
    # 
    rf_model, scaler, df = load_models_and_data()
    
    # 
    print("\nEvaluating tau predictions...")
    results_df = evaluate_tau_predictions(df, rf_model, scaler)
    
    # 
    metrics = analyze_results(results_df, output_dir)
    
    # 
    print("\nGenerating visualizations...")
    create_visualizations(results_df, output_dir)
    
    print("\n" + "="*80)
    print("Evaluation complete! Check ./results directory for results.")

if __name__ == "__main__":
    main()