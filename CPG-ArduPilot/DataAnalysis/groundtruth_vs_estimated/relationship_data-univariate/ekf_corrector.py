#!/usr/bin/env python3
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
            print(f"
{group.upper()}:")
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
    
    print("\nEKF Corrector ready for use!")
