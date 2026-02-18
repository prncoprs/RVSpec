#!/usr/bin/env python3
"""
cFS Parameter Generator using Latin Hypercube Sampling
Generates Inp_Sim.txt, Orb_LEO.txt, and SC_NOS3.txt files for cFS/NOS3 simulation studies
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import re

import numpy as np
import pandas as pd
from scipy.stats import qmc


class CFSParameterGenerator:
    def __init__(self, csv_file: str, templates_dir: str = "./templates", 
                 output_dir: str = "./parameter_output", seed: int = 42):
        """
        Initialize the cFS parameter generator
        
        Args:
            csv_file: Path to CSV file with parameter ranges
            templates_dir: Directory containing template files
            output_dir: Directory to save generated parameter sets
            seed: Random seed for reproducibility
        """
        self.csv_file = Path(csv_file)
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'parameter_sets').mkdir(exist_ok=True)
        
        # Load parameter ranges
        print(f"Loading parameter ranges from: {self.csv_file}")
        self.params_df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.params_df)} parameters")
        
        # Load template files
        self.load_templates()
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
    def load_templates(self):
        """Load template files"""
        template_files = {
            'inp_sim': self.templates_dir / 'Inp_Sim.txt',
            'orb_leo': self.templates_dir / 'Orb_LEO.txt', 
            'sc_nos3': self.templates_dir / 'SC_NOS3.txt'
        }
        
        self.templates = {}
        for name, file_path in template_files.items():
            if not file_path.exists():
                raise FileNotFoundError(f"Template file not found: {file_path}")
            
            with open(file_path, 'r') as f:
                self.templates[name] = f.read()
            print(f"Loaded template: {file_path}")
    
    def extract_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter ranges from CSV for Latin hypercube sampling"""
        param_info = {}
        
        print("Processing parameter ranges:")
        print("-" * 80)
        
        for _, row in self.params_df.iterrows():
            param_name = row['Parameter']
            range_str = row['Range']
            default_val = row['Default']
            
            # Parse range
            if '~' in range_str:
                min_val, max_val = range_str.split('~')
                min_val = float(min_val.strip())
                max_val = float(max_val.strip())
            else:
                try:
                    min_val = max_val = float(range_str.strip())
                except:
                    min_val = max_val = 0.0
            
            param_info[param_name] = {
                'min': min_val,
                'max': max_val,
                'description': row['Description'],
                'unit': row['Unit'],
                'default': default_val,
                'original_range': range_str
            }
            
            print(f"{param_name:35} | {min_val:12.6f} ~ {max_val:12.6f} | {row['Unit']:15}")
        
        print("-" * 80)
        print(f"Total parameters processed: {len(param_info)}")
        return param_info
    
    def generate_latin_hypercube_samples(self, n_samples: int, param_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, float]]:
        """Generate parameter sets using Latin hypercube sampling"""
        # Filter out duplicate parameters (use only first instance for unified parameters)
        filtered_params = {}
        for param_name, info in param_info.items():
            if param_name.startswith('MOI_'):
                if 'MOI' not in filtered_params:
                    filtered_params['MOI'] = info  # Use MOI_XX for all
            elif param_name.startswith('MTB_') and param_name.endswith('_SATURATION'):
                if 'MTB_SATURATION' not in filtered_params:
                    filtered_params['MTB_SATURATION'] = info  # Use MTB_0 for all
            else:
                filtered_params[param_name] = info
        
        param_names = list(filtered_params.keys())
        n_params = len(param_names)
        
        print(f"Generating {n_samples} parameter sets using Latin Hypercube Sampling...")
        print(f"Parameter space dimension: {n_params}")
        print(f"Unified parameters: MOI (XX/YY/ZZ), MTB_SATURATION (0/1/2)")
        
        # Create Latin hypercube sampler with fixed seed
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        
        # Generate samples in [0,1] space
        unit_samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        scaled_samples = np.zeros_like(unit_samples)
        
        for i, param_name in enumerate(param_names):
            min_val = filtered_params[param_name]['min']
            max_val = filtered_params[param_name]['max']
            if min_val == max_val:
                # Fixed parameter
                scaled_samples[:, i] = min_val
            else:
                scaled_samples[:, i] = min_val + (max_val - min_val) * unit_samples[:, i]
        
        # Convert to list of dictionaries with expanded parameter names
        parameter_sets = []
        for sample in scaled_samples:
            param_set = {}
            for i, param_name in enumerate(param_names):
                if param_name == 'MOI':
                    # Expand to all three MOI parameters
                    param_set['MOI_XX'] = sample[i]
                    param_set['MOI_YY'] = sample[i]
                    param_set['MOI_ZZ'] = sample[i]
                elif param_name == 'MTB_SATURATION':
                    # Expand to all three MTB parameters
                    param_set['MTB_0_SATURATION'] = sample[i]
                    param_set['MTB_1_SATURATION'] = sample[i]
                    param_set['MTB_2_SATURATION'] = sample[i]
                else:
                    param_set[param_name] = sample[i]
            parameter_sets.append(param_set)
        
        print(f"Generated {len(parameter_sets)} parameter sets")
        return parameter_sets
    
    def substitute_inp_sim_parameters(self, param_set: Dict[str, float]) -> str:
        """Substitute parameters in Inp_Sim.txt template"""
        content = self.templates['inp_sim']
        
        # F10.7 flux parameter
        if 'F10_7_FLUX' in param_set:
            # Replace USER-provided F10.7 value
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+USER-provided F10\.7',
                f'\\g<1>{param_set["F10_7_FLUX"]:.1f}                           !  USER-provided F10.7',
                content
            )
        
        # Ap index parameter
        if 'AP_INDEX' in param_set:
            # Replace USER-provided Ap value
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+USER-provided Ap',
                f'\\g<1>{param_set["AP_INDEX"]:.1f}                           !  USER-provided Ap',
                content
            )
        
        return content
    
    def substitute_orb_leo_parameters(self, param_set: Dict[str, float]) -> str:
        """Substitute parameters in Orb_LEO.txt template"""
        content = self.templates['orb_leo']
        
        # Periapsis and Apoapsis altitudes
        if 'ORB_PERIAPSIS_ALT' in param_set and 'ORB_APOAPSIS_ALT' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+(\d+\.?\d*)\s+!\s+Periapsis & Apoapsis Altitude, km',
                f'\\g<1>{param_set["ORB_PERIAPSIS_ALT"]:.1f}      {param_set["ORB_APOAPSIS_ALT"]:.1f}              !  Periapsis & Apoapsis Altitude, km',
                content
            )
        
        # Inclination
        if 'ORB_INCLINATION' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Inclination \(deg\)',
                f'\\g<1>{param_set["ORB_INCLINATION"]:.1f}                          !  Inclination (deg)',
                content
            )
        
        # Right Ascension of Ascending Node
        if 'ORB_RAAN' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Right Ascension of Ascending Node \(deg\)',
                f'\\g<1>{param_set["ORB_RAAN"]:.1f}                         !  Right Ascension of Ascending Node (deg)',
                content
            )
        
        # Argument of Periapsis
        if 'ORB_ARG_PERIAPSIS' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Argument of Periapsis \(deg\)',
                f'\\g<1>{param_set["ORB_ARG_PERIAPSIS"]:.1f}                           !  Argument of Periapsis (deg)',
                content
            )
        
        # True Anomaly
        if 'ORB_TRUE_ANOMALY' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+True Anomaly \(deg\)',
                f'\\g<1>{param_set["ORB_TRUE_ANOMALY"]:.1f}                           !  True Anomaly (deg)',
                content
            )
        
        return content
    
    def substitute_sc_nos3_parameters(self, param_set: Dict[str, float]) -> str:
        """Substitute parameters in SC_NOS3.txt template"""
        content = self.templates['sc_nos3']
        
        # Mass parameter
        if 'MASS' in param_set:
            content = re.sub(
                r'^(\s*)(\d+\.?\d*)\s+!\s+Mass$',
                f'\\g<1>{param_set["MASS"]:.1f}                         ! Mass',
                content,
                flags=re.MULTILINE
            )
        
        # Moments of Inertia (MOI_XX separate, MOI_YY and MOI_ZZ same value)
        if 'MOI_XX' in param_set and 'MOI_YY' in param_set:
            moi_xx = param_set['MOI_XX']
            moi_yz = param_set['MOI_YY']  # MOI_YY and MOI_ZZ use same value
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+!\s+Moments of Inertia \(kg-m\^2\)',
                f'\\g<1>{moi_xx:.6f}  {moi_yz:.6f}  {moi_yz:.6f}        ! Moments of Inertia (kg-m^2)',
                content
            )
        
        # MTB saturation parameters (all three using same value)
        if 'MTB_0_SATURATION' in param_set:
            mtb_value = param_set['MTB_0_SATURATION']
            # Replace all three MTB saturation values
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Saturation \(A-m\^2\)',
                f'\\g<1>{mtb_value:.2f}                        ! Saturation (A-m^2)',
                content
            )
        
        # Magnetometer parameters
        if 'MAG_SATURATION' in param_set:
            # Convert to scientific notation format used in template
            mag_sat = param_set['MAG_SATURATION']
            content = re.sub(
                r'(\s+)(\d+\.?\d*E?-?\d*)\s+!\s+Saturation, Tesla',
                f'\\g<1>{mag_sat:.0e}                    ! Saturation, Tesla',
                content
            )
        
        if 'MAG_QUANTIZATION' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*E?-?\d*)\s+!\s+Quantization, Tesla',
                f'\\g<1>{param_set["MAG_QUANTIZATION"]:.1e}                      ! Quantization, Tesla',
                content
            )
        
        if 'MAG_NOISE' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*E?-?\d*)\s+!\s+Noise, Tesla RMS',
                f'\\g<1>{param_set["MAG_NOISE"]:.1e}                         ! Noise, Tesla RMS',
                content
            )
        
        # Gyroscope parameters
        if 'GYRO_MAX_RATE' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Max Rate, deg/sec',
                f'\\g<1>{param_set["GYRO_MAX_RATE"]:.1f}                      ! Max Rate, deg/sec',
                content
            )
        
        if 'GYRO_SCALE_FACTOR_ERROR' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Scale Factor Error, ppm',
                f'\\g<1>{param_set["GYRO_SCALE_FACTOR_ERROR"]:.1f}                       ! Scale Factor Error, ppm',
                content
            )
        
        if 'GYRO_QUANTIZATION' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Quantization, arcsec',
                f'\\g<1>{param_set["GYRO_QUANTIZATION"]:.1f}                         ! Quantization, arcsec',
                content
            )
        
        if 'GYRO_ANGLE_NOISE' in param_set:
            content = re.sub(
                r'(\s+)(\d+\.?\d*)\s+!\s+Angle Noise, arcsec RMS',
                f'\\g<1>{param_set["GYRO_ANGLE_NOISE"]:.1f}                         ! Angle Noise, arcsec RMS',
                content
            )
        
        return content
    
    def save_parameter_set(self, run_id: int, param_set: Dict[str, float], 
                          configs: Dict[str, str]):
        """Save parameter set configuration to files"""
        param_dir = self.output_dir / 'parameter_sets' / f'config_{run_id:04d}'
        param_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the three configuration files
        config_files = {
            'Inp_Sim.txt': configs['inp_sim'],
            'Orb_LEO.txt': configs['orb_leo'],
            'SC_NOS3.txt': configs['sc_nos3']
        }
        
        for filename, content in config_files.items():
            config_file = param_dir / filename
            with open(config_file, 'w') as f:
                f.write(content)
        
        # Save metadata
        metadata = {
            'run_id': run_id,
            'parameter_set_id': f'config_{run_id:04d}',
            'generation_timestamp': time.time(),
            'parameters': param_set,
            'files': list(config_files.keys()),
            'seed': self.seed
        }
        
        metadata_file = param_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_parameter_study(self, n_samples: int):
        """Generate complete parameter study configuration"""
        print(f"Starting cFS parameter study generation for {n_samples} samples...")
        print("=" * 80)
        
        # Extract parameter information
        param_info = self.extract_parameter_ranges()
        
        # Generate parameter sets using Latin Hypercube Sampling
        parameter_sets = self.generate_latin_hypercube_samples(n_samples, param_info)
        
        # Generate configurations and save
        print("\nGenerating and saving configurations...")
        summary_data = []
        
        for i in range(n_samples):
            param_set = parameter_sets[i]
            
            # Generate all three configuration files
            configs = {
                'inp_sim': self.substitute_inp_sim_parameters(param_set),
                'orb_leo': self.substitute_orb_leo_parameters(param_set),
                'sc_nos3': self.substitute_sc_nos3_parameters(param_set)
            }
            
            # Save configuration files
            self.save_parameter_set(i, param_set, configs)
            
            # Add to summary
            summary_entry = {
                'run_id': i,
                'parameter_set_id': f'config_{i:04d}',
                'generation_timestamp': time.time(),
                **param_set  # Add all parameter values
            }
            summary_data.append(summary_entry)
            
            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/{n_samples} configurations...")
        
        # Save parameter summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'parameter_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Save generation metadata
        generation_metadata = {
            'generation_timestamp': time.time(),
            'n_samples': n_samples,
            'random_seed': self.seed,
            'parameter_count': len(param_info),
            'templates_used': list(self.templates.keys()),
            'output_structure': {
                'parameter_sets_dir': 'parameter_sets/',
                'summary_file': 'parameter_summary.csv',
                'metadata_file': 'generation_metadata.json'
            }
        }
        
        metadata_file = self.output_dir / 'generation_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(generation_metadata, f, indent=2)
        
        print("\n" + "=" * 80)
        print("cFS parameter study generation complete!")
        print(f"Generated {n_samples} parameter sets")
        print(f"Output directory: {self.output_dir}")
        print(f"Parameter summary: {summary_file}")
        print(f"Generation metadata: {metadata_file}")
        print("\nDirectory structure:")
        print(f"  {self.output_dir}/")
        print(f"     parameter_sets/")
        print(f"        config_0000/")
        print(f"           Inp_Sim.txt")
        print(f"           Orb_LEO.txt")
        print(f"           SC_NOS3.txt")
        print(f"           metadata.json")
        print(f"        ...")
        print(f"     parameter_summary.csv")
        print(f"     generation_metadata.json")
    
    def print_parameter_info(self):
        """Print information about parameters found in CSV"""
        param_info = self.extract_parameter_ranges()
        
        print("\nParameter Information Summary:")
        print("=" * 80)
        
        # Group parameters by type
        physical_params = ['MASS', 'MOI_XX', 'MOI_YY', 'MOI_ZZ']
        mtb_params = [p for p in param_info.keys() if 'MTB' in p]
        mag_params = [p for p in param_info.keys() if 'MAG' in p]
        gyro_params = [p for p in param_info.keys() if 'GYRO' in p]
        orbital_params = [p for p in param_info.keys() if 'ORB' in p]
        env_params = ['F10_7_FLUX', 'AP_INDEX']
        
        categories = [
            ("Physical Parameters", physical_params),
            ("Magnetorquer Parameters", mtb_params),
            ("Magnetometer Parameters", mag_params),
            ("Gyroscope Parameters", gyro_params),
            ("Orbital Parameters", orbital_params),
            ("Environmental Parameters", env_params)
        ]
        
        for category_name, param_list in categories:
            if param_list:
                print(f"\n{category_name}:")
                print("-" * 50)
                for param_name in param_list:
                    if param_name in param_info:
                        info = param_info[param_name]
                        range_str = f"{info['min']:.6f} ~ {info['max']:.6f}"
                        print(f"  {param_name:30} | {range_str:25} | {info['unit']}")


def main():
    parser = argparse.ArgumentParser(description='Generate cFS/NOS3 parameter study configurations')
    parser.add_argument('--csv', default='parameter_range.csv', help='CSV file with parameter ranges')
    parser.add_argument('--templates-dir', default='./templates', help='Templates directory')
    parser.add_argument('--output-dir', default='./parameter_output', help='Output directory')
    parser.add_argument('--samples', type=int, default=500, help='Number of parameter sets to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--info', action='store_true', help='Print parameter information and exit')
    
    args = parser.parse_args()
    
    try:
        generator = CFSParameterGenerator(
            csv_file=args.csv,
            templates_dir=args.templates_dir,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        if args.info:
            generator.print_parameter_info()
        else:
            generator.generate_parameter_study(args.samples)
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure all required files exist:")
        print(f"  - CSV file: {args.csv}")
        print(f"  - Templates directory: {args.templates_dir}")
        print("    - Inp_Sim.txt")
        print("    - Orb_LEO.txt") 
        print("    - SC_NOS3.txt")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()