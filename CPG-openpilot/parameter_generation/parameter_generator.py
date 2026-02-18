#!/usr/bin/env python3
"""
CARLA Parameter Generator using Latin Hypercube Sampling
Generates parameter configurations for CARLA-OpenPilot simulation studies
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc


class CarlaParameterGenerator:
    def __init__(self, csv_file: str, output_dir: str = "parameter_study_output", 
                 maps_config: str = None):
        """
        Initialize the parameter generator
        
        Args:
            csv_file: Path to CSV file with parameter ranges
            output_dir: Directory to save generated parameter sets
            maps_config: Path to maps configuration file (YAML format)
        """
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.maps_config_file = Path(maps_config) if maps_config else None
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'parameter_sets').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Load parameter ranges
        print(f"Loading parameter ranges from: {self.csv_file}")
        self.params_df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.params_df)} parameters")
        
        # Load maps configuration
        self.maps_config = self.load_maps_config()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def load_maps_config(self) -> Dict[str, List[str]]:
        """Load maps configuration from file or use defaults"""
        if self.maps_config_file and self.maps_config_file.exists():
            print(f"Loading maps configuration from: {self.maps_config_file}")
            try:
                # Simple YAML parser for our specific format
                config = {'highway_maps': [], 'intersection_maps': [], 'roundabout_maps': []}
                current_category = None
                
                with open(self.maps_config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.endswith(':') and not line.startswith('#'):
                            current_category = line.rstrip(':')
                        elif line.startswith('- ') and current_category in config:
                            map_name = line[2:].strip()
                            config[current_category].append(map_name)
                
                print(f"Loaded maps config: {config}")
                return config
            except Exception as e:
                print(f"Error loading maps config: {e}")
                print("Using default maps configuration")
        
        # Default configuration - using only Town03
        default_config = {
            'highway_maps': ['Town01'],
            'intersection_maps': ['Town01'],
            'roundabout_maps': ['Town01']
        }
        print(f"Using default maps configuration: {default_config}")
        return default_config
    
    def categorize_parameters(self, param_name: str) -> str:
        """Categorize parameters based on their names"""
        param_lower = param_name.lower()
        
        if any(x in param_lower for x in ['mass', 'drag', 'center_of_mass', 'rpm', 'moi', 'damping', 'gear', 'clutch']):
            return 'Vehicle_Engine'
        elif any(x in param_lower for x in ['tire_friction', 'wheel', 'brake', 'steer', 'radius']):
            return 'Vehicle_Wheels'
        elif any(x in param_lower for x in ['cloud', 'precipitation', 'wetness', 'wind', 'sun', 'fog', 'scattering', 'dust']):
            return 'Weather'
        elif any(x in param_lower for x in ['friction']):
            return 'Road_Surface'
        elif any(x in param_lower for x in ['camera', 'lidar', 'gnss', 'gps', 'imu']):
            return 'Sensors'
        else:
            return 'General'
    
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
            
            # Categorize parameter
            category = self.categorize_parameters(param_name)
            
            param_info[param_name] = {
                'min': min_val,
                'max': max_val,
                'description': row['Description'],
                'unit': row['Unit'],
                'default': default_val,
                'category': category,
                'original_range': range_str
            }
            
            print(f"{param_name:45} | {min_val:12.6f} ~ {max_val:12.6f} | {row['Unit']:15} | {category}")
        
        print("-" * 80)
        print(f"Total parameters processed: {len(param_info)}")
        return param_info
    
    def assign_maps_to_parameter_sets(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Assign maps and spawn points to parameter sets
        
        Args:
            n_samples: Number of parameter sets to generate
            
        Returns:
            List of map assignments with spawn points
        """
        map_assignments = []
        
        # Get all available maps
        all_maps = []
        for category, maps in self.maps_config.items():
            for map_name in maps:
                all_maps.append({
                    'map_name': map_name,
                    'category': category.replace('_maps', ''),
                    'scenario_type': category.replace('_maps', '')
                })
        
        if not all_maps:
            print("Warning: No maps configured. Using default Town03.")
            all_maps = [{'map_name': 'Town03', 'category': 'mixed', 'scenario_type': 'mixed'}]
        
        print(f"Map assignment strategy for {n_samples} parameter sets:")
        print(f"Available maps: {[m['map_name'] for m in all_maps]}")
        
        # Distribute maps across parameter sets
        # Strategy: Rotate through available maps to ensure even distribution
        for i in range(n_samples):
            # Select map using rotation
            map_assignment = all_maps[i % len(all_maps)].copy()
            map_assignment['parameter_set_id'] = i
            
            # Assign spawn point (will be validated later when we have actual map info)
            # For now, use a reasonable default range
            # map_assignment['spawn_point_index'] = np.random.randint(0, 50)  # Most maps have at least 50 spawn points
            map_assignment['spawn_point_index'] = 1

            # Add some randomization within scenario type for variety
            if len([m for m in all_maps if m['category'] == map_assignment['category']]) > 1:
                category_maps = [m for m in all_maps if m['category'] == map_assignment['category']]
                selected_map = np.random.choice(category_maps)
                map_assignment['map_name'] = selected_map['map_name']
            
            map_assignments.append(map_assignment)
        
        # Print distribution summary
        map_distribution = {}
        for assignment in map_assignments:
            map_name = assignment['map_name']
            map_distribution[map_name] = map_distribution.get(map_name, 0) + 1
        
        print("Map distribution:")
        for map_name, count in map_distribution.items():
            print(f"  {map_name}: {count} runs ({count/n_samples*100:.1f}%)")
        
        return map_assignments
    
    def generate_latin_hypercube_samples(self, n_samples: int, param_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, float]]:
        """Generate parameter sets using Latin hypercube sampling"""
        param_names = list(param_info.keys())
        n_params = len(param_names)
        
        print(f"Generating {n_samples} parameter sets using Latin Hypercube Sampling...")
        print(f"Parameter space dimension: {n_params}")
        
        # Create Latin hypercube sampler with fixed seed
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        
        # Generate samples in [0,1] space
        unit_samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        scaled_samples = np.zeros_like(unit_samples)
        
        for i, param_name in enumerate(param_names):
            min_val = param_info[param_name]['min']
            max_val = param_info[param_name]['max']
            if min_val == max_val:
                # Fixed parameter
                scaled_samples[:, i] = min_val
            else:
                scaled_samples[:, i] = min_val + (max_val - min_val) * unit_samples[:, i]
        
        # Convert to list of dictionaries
        parameter_sets = []
        for sample in scaled_samples:
            param_set = {}
            for i, param_name in enumerate(param_names):
                param_set[param_name] = sample[i]
            parameter_sets.append(param_set)
        
        print(f"Generated {len(parameter_sets)} parameter sets")
        return parameter_sets
    
    def create_run_configuration(self, run_id: int, param_set: Dict[str, float], 
                               map_assignment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create complete configuration for a single simulation run
        
        Args:
            run_id: Unique run identifier
            param_set: Generated parameter values
            map_assignment: Map and spawn point assignment
            
        Returns:
            Complete run configuration dictionary
        """
        
        # Create configuration structure
        config = {
            'run_metadata': {
                'run_id': run_id,
                'parameter_set_id': f'param_{run_id:04d}',
                'generation_timestamp': time.time(),
                'random_seed': 42,
                'simulation_duration': 300,  # 5 minutes default
                'description': f'CARLA-OpenPilot parameter study run {run_id}'
            },
            
            'map_config': {
                'map_name': map_assignment['map_name'],
                'scenario_type': map_assignment['scenario_type'],
                'spawn_point_index': map_assignment['spawn_point_index'],
                'weather_preset': 'ClearSunset'  # Default weather
            },
            
            'vehicle_config': {
                'vehicle_type': 'vehicle.tesla.model3',  # Default vehicle
                'physics': {
                    'mass': param_set.get('mass', 2000.0),
                    'drag_coefficient': param_set.get('drag_coefficient', 0.3),
                    'center_of_mass': [
                        param_set.get('center_of_mass.x', 0.0),
                        param_set.get('center_of_mass.y', 0.0),
                        param_set.get('center_of_mass.z', 0.0)
                    ],
                    'max_rpm': param_set.get('max_rpm', 6500.0),
                    'moi': param_set.get('moi', 1.0),
                    'damping_rate_full_throttle': param_set.get('damping_rate_full_throttle', 0.15),
                    'damping_rate_zero_throttle_clutch_engaged': param_set.get('damping_rate_zero_throttle_clutch_engaged', 2.0),
                    'damping_rate_zero_throttle_clutch_disengaged': param_set.get('damping_rate_zero_throttle_clutch_disengaged', 0.35),
                    'gear_switch_time': param_set.get('gear_switch_time', 0.5),
                    'clutch_strength': param_set.get('clutch_strength', 10.0),
                    'wheels': [
                        {
                            'tire_friction': param_set.get('wheels[0].tire_friction', 1.0),
                            'damping_rate': param_set.get('wheels[0].damping_rate', 0.25),
                            'max_steer_angle': param_set.get('wheels[0].max_steer_angle', 70.0),
                            'max_brake_torque': param_set.get('wheels[0].max_brake_torque', 1500.0),
                            'radius': param_set.get('wheels[0].radius', 0.30)
                        },
                        {
                            'tire_friction': param_set.get('wheels[1].tire_friction', 1.0),
                            'damping_rate': param_set.get('wheels[1].damping_rate', 0.25),
                            'max_steer_angle': param_set.get('wheels[1].max_steer_angle', 70.0),
                            'max_brake_torque': param_set.get('wheels[1].max_brake_torque', 1500.0),
                            'radius': param_set.get('wheels[1].radius', 0.30)
                        },
                        {
                            'tire_friction': param_set.get('wheels[2].tire_friction', 1.0),
                            'damping_rate': param_set.get('wheels[2].damping_rate', 0.25),
                            'max_brake_torque': param_set.get('wheels[2].max_brake_torque', 1500.0),
                            'radius': param_set.get('wheels[2].radius', 0.30)
                        },
                        {
                            'tire_friction': param_set.get('wheels[3].tire_friction', 1.0),
                            'damping_rate': param_set.get('wheels[3].damping_rate', 0.25),
                            'max_brake_torque': param_set.get('wheels[3].max_brake_torque', 1500.0),
                            'radius': param_set.get('wheels[3].radius', 0.30)
                        }
                    ]
                }
            },
            
            'weather_config': {
                'cloudiness': param_set.get('cloudiness', 10.0),
                'precipitation': param_set.get('precipitation', 0.0),
                'precipitation_deposits': param_set.get('precipitation_deposits', 0.0),
                'wetness': param_set.get('wetness', 0.0),
                'wind_intensity': param_set.get('wind_intensity', 0.0),
                'sun_azimuth_angle': param_set.get('sun_azimuth_angle', 180.0),
                'sun_altitude_angle': param_set.get('sun_altitude_angle', 75.0),
                'fog_density': param_set.get('fog_density', 0.0),
                'fog_distance': param_set.get('fog_distance', 500.0),
                'fog_falloff': param_set.get('fog_falloff', 0.2),
                'scattering_intensity': param_set.get('scattering_intensity', 1.0),
                'mie_scattering_scale': param_set.get('mie_scattering_scale', 0.03),
                'rayleigh_scattering_scale': param_set.get('rayleigh_scattering_scale', 0.0331),
                'dust_storm': param_set.get('dust_storm', 0.0)
            },
            
            'sensor_config': {
                'camera': {
                    'rgb_noise_stddev': param_set.get('sensor.camera.rgb.noise_stddev', 0.0),
                    'road_camera_fov': 40,  # Fixed values for now
                    'wide_camera_fov': 120,
                    'image_size_x': 1928,
                    'image_size_y': 1208
                },
                'lidar': {
                    'ray_cast_noise_stddev': param_set.get('sensor.lidar.ray_cast.noise_stddev', 0.0)
                },
                'gnss': {
                    'noise_alt_bias': param_set.get('sensor.other.gnss.noise_alt_bias', 0.0),
                    'noise_alt_stddev': param_set.get('sensor.other.gnss.noise_alt_stddev', 0.1),
                    'noise_lat_bias': param_set.get('sensor.other.gnss.noise_lat_bias', 0.0),
                    'noise_lat_stddev': param_set.get('sensor.other.gnss.noise_lat_stddev', 0.0001),
                    'noise_lon_bias': param_set.get('sensor.other.gnss.noise_lon_bias', 0.0),
                    'noise_lon_stddev': param_set.get('sensor.other.gnss.noise_lon_stddev', 0.0001)
                },
                'imu': {
                    'noise_accel_stddev': [
                        param_set.get('sensor.other.imu.noise_accel_stddev_x', 0.0),
                        param_set.get('sensor.other.imu.noise_accel_stddev_y', 0.0),
                        param_set.get('sensor.other.imu.noise_accel_stddev_z', 0.0)
                    ],
                    'noise_gyro_bias': [
                        param_set.get('sensor.other.imu.noise_gyro_bias_x', 0.0),
                        param_set.get('sensor.other.imu.noise_gyro_bias_y', 0.0),
                        param_set.get('sensor.other.imu.noise_gyro_bias_z', 0.0)
                    ],
                    'noise_gyro_stddev': [
                        param_set.get('sensor.other.imu.noise_gyro_stddev_x', 0.0),
                        param_set.get('sensor.other.imu.noise_gyro_stddev_y', 0.0),
                        param_set.get('sensor.other.imu.noise_gyro_stddev_z', 0.0)
                    ]
                }
            },
            
            'road_config': {
                'friction_trigger_coefficient': param_set.get('static.trigger.friction.friction', 1.0)
            },
            
            'bridge_config': {
                'control': {
                    'steer_ratio': 15.0,  # Fixed bridge parameters
                    'throttle_manual_multiplier': 0.7,
                    'brake_manual_multiplier': 0.7,
                    'steer_manual_multiplier': 45.0,
                    'steer_rate_limit': 0.5
                },
                'timing': {
                    'bridge_frequency': 100,
                    'world_tick_frequency': 20,
                    'print_decimation': 100
                }
            },
            
            'logging_config': {
                'enable_carla_logging': True,
                'enable_openpilot_logging': True,
                'log_vehicle_state': True,
                'log_sensor_data': True,
                'log_control_commands': True,
                'log_performance_metrics': True
            },
            
            'raw_parameters': param_set  # Store original parameter values
        }
        
        return config
    
    def save_parameter_set(self, run_id: int, config: Dict[str, Any]):
        """Save parameter set configuration to files"""
        param_dir = self.output_dir / 'parameter_sets' / f'param_{run_id:04d}'
        param_dir.mkdir(exist_ok=True, parents=True)
        
        # Save complete configuration as JSON
        config_file = param_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save bridge-specific configuration as YAML-like format
        bridge_config_file = param_dir / 'bridge_config.yaml'
        bridge_config = {
            'map_name': config['map_config']['map_name'],
            'spawn_point_index': config['map_config']['spawn_point_index'],
            'simulation_duration': config['run_metadata']['simulation_duration'],
            'vehicle_physics': config['vehicle_config']['physics'],
            'weather': config['weather_config'],
            'sensors': config['sensor_config'],
            'road': config['road_config'],
            'bridge_control': config['bridge_config']
        }
        
        # Simple YAML-like output
        with open(bridge_config_file, 'w') as f:
            f.write(f"# Bridge configuration for run {run_id}\n")
            f.write(f"# Generated at: {time.ctime()}\n\n")
            self._write_yaml_dict(f, bridge_config, 0)
        
        # Save metadata
        metadata_file = param_dir / 'metadata.json'
        metadata = {
            'run_id': run_id,
            'parameter_set_id': config['run_metadata']['parameter_set_id'],
            'generation_timestamp': config['run_metadata']['generation_timestamp'],
            'map_name': config['map_config']['map_name'],
            'scenario_type': config['map_config']['scenario_type'],
            'spawn_point': config['map_config']['spawn_point_index'],
            'parameter_count': len(config['raw_parameters']),
            'files': {
                'config': str(config_file.name),
                'bridge_config': str(bridge_config_file.name),
                'metadata': str(metadata_file.name)
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _write_yaml_dict(self, f, data: Dict[str, Any], indent: int):
        """Helper function to write dictionary in YAML-like format"""
        for key, value in data.items():
            indent_str = '  ' * indent
            if isinstance(value, dict):
                f.write(f"{indent_str}{key}:\n")
                self._write_yaml_dict(f, value, indent + 1)
            elif isinstance(value, list):
                f.write(f"{indent_str}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        f.write(f"{indent_str}  -\n")
                        self._write_yaml_dict(f, item, indent + 2)
                    else:
                        f.write(f"{indent_str}  - {item}\n")
            else:
                f.write(f"{indent_str}{key}: {value}\n")
    
    def generate_parameter_study(self, n_samples: int):
        """Generate complete parameter study configuration"""
        print(f"Starting parameter study generation for {n_samples} samples...")
        print("=" * 80)
        
        # Extract parameter information
        param_info = self.extract_parameter_ranges()
        
        # Generate parameter sets using Latin Hypercube Sampling
        parameter_sets = self.generate_latin_hypercube_samples(n_samples, param_info)
        
        # Assign maps to parameter sets
        map_assignments = self.assign_maps_to_parameter_sets(n_samples)
        
        # Generate configurations and save
        print("\nGenerating and saving configurations...")
        summary_data = []
        
        for i in range(n_samples):
            config = self.create_run_configuration(i, parameter_sets[i], map_assignments[i])
            self.save_parameter_set(i, config)
            
            # Add to summary
            summary_entry = {
                'run_id': i,
                'parameter_set_id': f'param_{i:04d}',
                'map_name': config['map_config']['map_name'],
                'scenario_type': config['map_config']['scenario_type'],
                'spawn_point': config['map_config']['spawn_point_index'],
                'generation_timestamp': config['run_metadata']['generation_timestamp']
            }
            
            # Add key parameter values to summary
            for param_name, param_value in parameter_sets[i].items():
                summary_entry[param_name] = param_value
            
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
            'random_seed': 42,
            'parameter_count': len(param_info),
            'maps_used': list(set([assignment['map_name'] for assignment in map_assignments])),
            'scenario_distribution': {
                scenario: len([a for a in map_assignments if a['scenario_type'] == scenario])
                for scenario in set([a['scenario_type'] for a in map_assignments])
            },
            'parameter_categories': {
                category: len([p for p in param_info.values() if p['category'] == category])
                for category in set([p['category'] for p in param_info.values()])
            },
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
        print("Parameter study generation complete!")
        print(f"Generated {n_samples} parameter sets")
        print(f"Output directory: {self.output_dir}")
        print(f"Parameter summary: {summary_file}")
        print(f"Generation metadata: {metadata_file}")
        print("\nDirectory structure:")
        print(f"  {self.output_dir}/")
        print(f"     parameter_sets/")
        print(f"        param_0000/")
        print(f"           config.json")
        print(f"           bridge_config.yaml")
        print(f"           metadata.json")
        print(f"        ...")
        print(f"     parameter_summary.csv")
        print(f"     generation_metadata.json")
    
    def print_parameter_info(self):
        """Print information about parameters found in CSV"""
        param_info = self.extract_parameter_ranges()
        
        print("\nParameter Information Summary:")
        print("=" * 80)
        
        # Group by category
        categories = {}
        for name, info in param_info.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, info))
        
        for category, params in categories.items():
            print(f"\n{category.upper()}:")
            print("-" * 50)
            for name, info in params:
                range_str = f"{info['min']:.6f} ~ {info['max']:.6f}"
                print(f"  {name:35} | {range_str:25} | {info['unit']}")


def main():
    parser = argparse.ArgumentParser(description='Generate CARLA-OpenPilot parameter study configurations')
    parser.add_argument('--csv', default='parameter_range-Carla.csv', help='CSV file with parameter ranges')
    parser.add_argument('--output-dir', default='parameter_output', help='Output directory')
    parser.add_argument('--maps-config', help='Maps configuration file (YAML format)')
    parser.add_argument('--samples', type=int, default=500, help='Number of parameter sets to generate')
    parser.add_argument('--info', action='store_true', help='Print parameter information and exit')
    
    args = parser.parse_args()
    
    try:
        generator = CarlaParameterGenerator(
            csv_file=args.csv,
            output_dir=args.output_dir,
            maps_config=args.maps_config
        )
        
        if args.info:
            generator.print_parameter_info()
        else:
            generator.generate_parameter_study(args.samples)
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure all required files exist:")
        print(f"  - CSV file: {args.csv}")
        if args.maps_config:
            print(f"  - Maps config: {args.maps_config}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()