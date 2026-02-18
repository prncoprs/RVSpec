#!/usr/bin/env python3
"""
CARLA Core Factors Brake Distance Experiment
Focus on 5 key factors: tire_friction, brake_torque, mass, drag_coefficient, road_friction
"""

import carla
import time
import math
import json
import csv
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class CarlaCoreBrakeExperiment:
    def __init__(self, parameter_sets_dir: str, output_dir: str = "core_experiment_results"):
        """Initialize the core brake experiment runner"""
        self.parameter_sets_dir = Path(parameter_sets_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # CARLA connection
        self.client = None
        self.world = None
        self.map = None
        self.spectator = None
        
        # Experiment settings
        self.TARGET_SPEED = 20.0  # m/s (72 km/h)
        self.BRAKE_THRESHOLD = 0.1  # m/s considered stopped
        self.EXPERIMENT_TIMEOUT = 60.0  # seconds max per experiment
        
        # Store applied factors for each experiment
        self.applied_factors = {}
        
        # Load parameter sets
        self.parameter_sets = self.load_parameter_sets()
        print(f"Loaded {len(self.parameter_sets)} parameter sets")
        
    def load_parameter_sets(self) -> List[Dict[str, Any]]:
        """Load all parameter sets from the generated configurations"""
        parameter_sets = []
        
        param_dirs = sorted([d for d in self.parameter_sets_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('param_')])
        
        for param_dir in param_dirs:
            config_file = param_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                parameter_sets.append(config)
            else:
                print(f"Warning: No config.json found in {param_dir}")
        
        return parameter_sets
    
    def connect_to_carla(self, host='localhost', port=2000, timeout=10.0):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA server at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spectator = self.world.get_spectator()
        
        print("Connected to CARLA successfully")
    
    def destroy_all_actors(self):
        """Clean up all actors in the world"""
        actors = self.world.get_actors()
        for actor in actors:
            if 'vehicle' in actor.type_id or actor.type_id == 'static.trigger.friction':
                try:
                    if actor.is_alive:
                        actor.destroy()
                except:
                    pass
        
        # Wait for cleanup
        time.sleep(0.5)
    
    def get_vehicle_speed(self, vehicle: carla.Vehicle) -> float:
        """Get vehicle speed in m/s"""
        velocity = vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    
    def setup_spectator_view(self, vehicle: carla.Vehicle):
        """Setup spectator camera closer to vehicle"""
        try:
            vehicle_location = vehicle.get_location()
            spectator_transform = carla.Transform(
                carla.Location(vehicle_location.x - 15, vehicle_location.y, vehicle_location.z + 8),
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )
            self.spectator.set_transform(spectator_transform)
        except:
            pass
    
    def update_spectator_view(self, vehicle: carla.Vehicle):
        """Update spectator camera to follow closer behind the vehicle"""
        try:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # Position camera closer behind the vehicle
            yaw_rad = math.radians(vehicle_rotation.yaw)
            backward_x = -math.cos(yaw_rad) * 15  # 15m behind (closer)
            backward_y = -math.sin(yaw_rad) * 15  # 15m behind (closer)
            
            camera_location = carla.Location(
                vehicle_location.x + backward_x,
                vehicle_location.y + backward_y,
                vehicle_location.z + 8  # 8m above (closer)
            )
            
            spectator_transform = carla.Transform(
                camera_location,
                carla.Rotation(pitch=-15, yaw=vehicle_rotation.yaw, roll=0)
            )
            self.spectator.set_transform(spectator_transform)
        except:
            pass
    
    def wait_for_speed(self, vehicle: carla.Vehicle, target_speed: float, timeout: float = 30.0):
        """Wait for vehicle to reach target speed"""
        start_time = time.time()
        
        print(f"Accelerating to {target_speed:.1f} m/s...")
        
        while time.time() - start_time < timeout:
            current_speed = self.get_vehicle_speed(vehicle)
            
            if current_speed >= target_speed * 0.95:
                print(f"Target speed reached: {current_speed:.2f} m/s")
                return True
            
            # Keep vehicle going straight
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
            
            # Update camera view
            self.update_spectator_view(vehicle)
            
            time.sleep(0.05)
            
            # Print progress every 3 seconds
            if int(time.time() - start_time) % 3 == 0:
                print(f"  Current speed: {current_speed:.2f} m/s")
        
        current_speed = self.get_vehicle_speed(vehicle)
        print(f"Acceleration finished. Final speed: {current_speed:.2f} m/s")
        return current_speed >= 5.0
    
    def apply_core_vehicle_factors(self, vehicle: carla.Vehicle, config: Dict[str, Any], run_id: int):
        """Apply only the 5 core factors that affect braking"""
        raw_params = config['raw_parameters']
        
        # Get current physics control
        physics_control = vehicle.get_physics_control()
        
        print("Applying core vehicle factors:")
        
        # CORE FACTOR 1: Mass
        applied_mass = max(1500, min(4000, raw_params.get('mass', 2000.0)))
        physics_control.mass = applied_mass
        
        # CORE FACTOR 2: Drag coefficient 
        applied_drag_coefficient = max(0.2, min(0.8, raw_params.get('drag_coefficient', 0.3)))
        physics_control.drag_coefficient = applied_drag_coefficient
        
        # CORE FACTOR 3: Tire friction (ALL 4 wheels SAME - fix)
        applied_tire_friction = max(0.4, min(1.6, raw_params.get('wheels[0].tire_friction', 1.0)))
        
        # CORE FACTOR 4: Brake torque (ALL 4 wheels based on first wheel)
        applied_brake_torque = max(1000, min(3500, raw_params.get('wheels[0].max_brake_torque', 1500.0)))
        
        # Apply wheel modifications - ALL 4 WHEELS IDENTICAL
        wheels = []
        for i in range(4):
            wheel = physics_control.wheels[i]  # Keep other default settings
            
            # Apply our core factors - ALL WHEELS SAME
            wheel.tire_friction = applied_tire_friction
            
            # Brake torque: slight front/rear difference but based on same value
            if i < 2:  # Front wheels
                wheel.max_brake_torque = applied_brake_torque
            else:  # Rear wheels  
                wheel.max_brake_torque = applied_brake_torque * 0.8
            
            wheels.append(wheel)
        
        physics_control.wheels = wheels
        
        # Apply physics control
        vehicle.apply_physics_control(physics_control)
        
        # Store the ACTUALLY APPLIED core factors
        self.applied_factors[run_id] = {
            'applied_mass': applied_mass,
            'applied_drag_coefficient': applied_drag_coefficient,
            'applied_tire_friction': applied_tire_friction,
            'applied_brake_torque_front': applied_brake_torque,
            'applied_brake_torque_rear': applied_brake_torque * 0.8,
        }
        
        print(f"  Mass: {applied_mass:.1f}kg")
        print(f"  Drag coefficient: {applied_drag_coefficient:.3f}")
        print(f"  All wheels tire friction: {applied_tire_friction:.3f}")
        print(f"  Brake torque: {applied_brake_torque:.0f}Nm (front), {applied_brake_torque*0.8:.0f}Nm (rear)")
    
    def spawn_road_friction(self, config: Dict[str, Any], location: carla.Location, run_id: int) -> carla.Actor:
        """Spawn road friction trigger - CORE FACTOR 5"""
        raw_params = config['raw_parameters']
        
        # CORE FACTOR 5: Road friction
        applied_road_friction = max(0.4, min(1.2, raw_params.get('static.trigger.friction.friction', 1.0)))
        
        friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
        friction_bp.set_attribute('friction', str(applied_road_friction))
        friction_bp.set_attribute('extent_x', '200.0')
        friction_bp.set_attribute('extent_y', '20.0')
        friction_bp.set_attribute('extent_z', '2.0')
        
        transform = carla.Transform(location)
        friction_actor = self.world.try_spawn_actor(friction_bp, transform)
        
        if friction_actor:
            print(f"  Road friction: {applied_road_friction:.3f}")
            
            if run_id not in self.applied_factors:
                self.applied_factors[run_id] = {}
            self.applied_factors[run_id]['applied_road_friction'] = applied_road_friction
        
        return friction_actor
    
    def spawn_test_vehicle(self, run_id: int) -> carla.Vehicle:
        """Spawn test vehicle with default settings"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")
        
        time.sleep(0.5)
        print(f"  Vehicle spawned successfully")
        return vehicle
    
    def run_brake_test(self, config: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Run a single brake distance test with core factors only"""
        print(f"\n--- Running brake test {run_id} ---")
        
        try:
            # Clean up previous test
            self.destroy_all_actors()
            
            # Spawn vehicle first
            vehicle = self.spawn_test_vehicle(run_id)
            
            # Apply ONLY core factors (no weather changes)
            self.apply_core_vehicle_factors(vehicle, config, run_id)
            
            # Reset vehicle position to ensure straight line
            spawn_point = self.world.get_map().get_spawn_points()[0]
            vehicle.set_transform(spawn_point)
            vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            
            # Double reset for clean start
            vehicle.set_transform(spawn_point)
            time.sleep(0.8)
            
            # Setup closer camera
            self.setup_spectator_view(vehicle)
            
            # Accelerate to target speed
            speed_reached = self.wait_for_speed(vehicle, self.TARGET_SPEED)
            if not speed_reached:
                current_speed = self.get_vehicle_speed(vehicle)
                if current_speed < 5.0:
                    raise RuntimeError(f"Vehicle speed too low: {current_speed:.2f} m/s")
            
            # Drive straight to stabilize
            print("Driving straight to stabilize...")
            for i in range(30):
                vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0))
                self.update_spectator_view(vehicle)
                time.sleep(0.05)
            
            # Spawn road friction ahead
            current_location = vehicle.get_location()
            forward_vec = vehicle.get_transform().get_forward_vector()
            friction_location = carla.Location(
                current_location.x + forward_vec.x * 50,
                current_location.y + forward_vec.y * 50,
                current_location.z
            )
            friction_actor = self.spawn_road_friction(config, friction_location, run_id)
            
            # Record brake start position
            brake_start_location = vehicle.get_location()
            brake_start_speed = self.get_vehicle_speed(vehicle)
            brake_start_time = time.time()
            
            print(f"Starting brake test at speed: {brake_start_speed:.2f} m/s")
            print("BRAKING NOW!")
            
            # Apply brakes with detailed logging
            brake_speeds = []
            brake_times = []
            brake_locations = []
            
            while self.get_vehicle_speed(vehicle) > self.BRAKE_THRESHOLD:
                current_speed = self.get_vehicle_speed(vehicle)
                current_time = time.time() - brake_start_time
                current_location = vehicle.get_location()
                
                brake_speeds.append(current_speed)
                brake_times.append(current_time)
                brake_locations.append(current_location)
                
                # Apply full brake with straight steering
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                
                # Update camera
                self.update_spectator_view(vehicle)
                
                time.sleep(0.05)
                
                if current_time > self.EXPERIMENT_TIMEOUT:
                    raise RuntimeError("Brake test timeout")
            
            # Calculate results
            stop_location = vehicle.get_location()
            brake_distance = brake_start_location.distance(stop_location)
            brake_time = brake_times[-1] if brake_times else 0.0
            
            print(f" Brake test completed: distance={brake_distance:.2f}m, time={brake_time:.2f}s")
            
            # Calculate tau for ReactSO mission
            safety_buffer = 0.5
            required_tau = brake_distance / max(brake_start_speed, 0.1) + safety_buffer
            
            # Get applied factors
            applied_factors = self.applied_factors.get(run_id, {})
            
            result = {
                'run_id': run_id,
                'parameter_set_id': config['run_metadata']['parameter_set_id'],
                'success': True,
                'brake_start_speed': brake_start_speed,
                'brake_distance': brake_distance,
                'brake_time': brake_time,
                'required_tau': required_tau,
                'target_speed': self.TARGET_SPEED,
                'safety_buffer': safety_buffer,
                
                # CORE FACTORS ONLY
                **applied_factors,
                
                # Additional analysis data
                'max_deceleration': max(brake_speeds) - min(brake_speeds) if brake_speeds else 0.0,
                'average_deceleration': (brake_start_speed - self.BRAKE_THRESHOLD) / brake_time if brake_time > 0 else 0.0,
                
                # Detailed brake profile for advanced analysis
                'brake_profile': {
                    'times': brake_times,
                    'speeds': brake_speeds,
                    'locations': [(loc.x, loc.y, loc.z) for loc in brake_locations]
                }
            }
            
            # Clean up
            vehicle.destroy()
            if friction_actor and friction_actor.is_alive:
                friction_actor.destroy()
            
            return result
            
        except Exception as e:
            print(f"Error in brake test {run_id}: {e}")
            self.destroy_all_actors()
            
            return {
                'run_id': run_id,
                'parameter_set_id': config.get('run_metadata', {}).get('parameter_set_id', f'param_{run_id:04d}'),
                'success': False,
                'error': str(e),
                'brake_distance': None,
                'required_tau': None
            }
    
    def run_experiment_batch(self, start_idx: int = 0, end_idx: int = None, 
                           results_file: str = "core_factors_brake_results.csv"):
        """Run a batch of core factor brake distance experiments"""
        if end_idx is None:
            end_idx = len(self.parameter_sets)
        
        print(f"Running core factors experiments {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
        print("=" * 80)
        print("CORE FACTORS: tire_friction, brake_torque, mass, drag_coefficient, road_friction")
        print("=" * 80)
        
        results_file_path = self.output_dir / results_file
        
        # Create CSV header for core factors analysis
        header = [
            'run_id', 'parameter_set_id', 'success', 
            'brake_start_speed', 'brake_distance', 'brake_time', 'required_tau', 
            'target_speed', 'safety_buffer',
            # CORE FACTORS ONLY
            'applied_mass', 'applied_drag_coefficient', 'applied_tire_friction', 
            'applied_brake_torque_front', 'applied_brake_torque_rear', 'applied_road_friction',
            # Analysis metrics
            'max_deceleration', 'average_deceleration',
            # Original parameter values for reference
            'original_mass', 'original_drag_coefficient', 'original_tire_friction_wheel0',
            'original_brake_torque_wheel0', 'original_road_friction',
            'error'
        ]
        
        with open(results_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        successful_runs = 0
        failed_runs = 0
        
        for i in range(start_idx, end_idx):
            config = self.parameter_sets[i]
            
            try:
                result = self.run_brake_test(config, i)
                
                if result['success']:
                    successful_runs += 1
                else:
                    failed_runs += 1
                
                # Extract original parameter values for comparison
                raw_params = config['raw_parameters']
                original_mass = raw_params.get('mass', 2000.0)
                original_drag_coefficient = raw_params.get('drag_coefficient', 0.3)
                original_tire_friction_wheel0 = raw_params.get('wheels[0].tire_friction', 1.0)
                original_brake_torque_wheel0 = raw_params.get('wheels[0].max_brake_torque', 1500.0)
                original_road_friction = raw_params.get('static.trigger.friction.friction', 1.0)
                
                # Write result to CSV
                with open(results_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result['run_id'], result['parameter_set_id'], result['success'],
                        result.get('brake_start_speed', ''), result.get('brake_distance', ''),
                        result.get('brake_time', ''), result.get('required_tau', ''),
                        result.get('target_speed', ''), result.get('safety_buffer', ''),
                        # Core factors applied
                        result.get('applied_mass', ''), result.get('applied_drag_coefficient', ''),
                        result.get('applied_tire_friction', ''), result.get('applied_brake_torque_front', ''),
                        result.get('applied_brake_torque_rear', ''), result.get('applied_road_friction', ''),
                        # Analysis metrics
                        result.get('max_deceleration', ''), result.get('average_deceleration', ''),
                        # Original values
                        original_mass, original_drag_coefficient, original_tire_friction_wheel0,
                        original_brake_torque_wheel0, original_road_friction,
                        result.get('error', '')
                    ])
                
                # Save detailed result for advanced analysis
                detailed_file = self.output_dir / f"detailed_core_result_{i:04d}.json"
                with open(detailed_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                if result['success']:
                    print(f" Run {i}: brake_distance={result['brake_distance']:.2f}m, "
                          f"tau={result['required_tau']:.2f}s, "
                          f"tire_friction={result.get('applied_tire_friction', 'N/A'):.2f}")
                else:
                    print(f" Run {i}: FAILED - {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                failed_runs += 1
                print(f" Run {i}: EXCEPTION - {e}")
                
                # Log the failure
                with open(results_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [i, f'param_{i:04d}', False] + [''] * (len(header) - 4) + [str(e)]
                    writer.writerow(row)
        
        print(f"\n" + "=" * 80)
        print(f"Core factors experiment batch completed!")
        print(f"Results saved to: {results_file_path}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Success rate: {successful_runs/(successful_runs+failed_runs)*100:.1f}%")
        print("=" * 80)
        
        # Generate summary statistics
        self.generate_summary_stats(results_file_path)
    
    def generate_summary_stats(self, results_file_path: Path):
        """Generate summary statistics for the core factors experiment"""
        try:
            df = pd.read_csv(results_file_path)
            successful_df = df[df['success'] == True]
            
            if len(successful_df) == 0:
                print("No successful runs to analyze.")
                return
            
            summary_file = self.output_dir / "core_factors_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write("CARLA Core Factors Brake Distance Experiment Summary\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total successful experiments: {len(successful_df)}\n")
                f.write(f"Brake distance range: {successful_df['brake_distance'].min():.2f}m - {successful_df['brake_distance'].max():.2f}m\n")
                f.write(f"Required tau range: {successful_df['required_tau'].min():.2f}s - {successful_df['required_tau'].max():.2f}s\n\n")
                
                f.write("Core Factors Ranges:\n")
                f.write("-" * 30 + "\n")
                for factor in ['applied_mass', 'applied_drag_coefficient', 'applied_tire_friction', 
                              'applied_brake_torque_front', 'applied_road_friction']:
                    if factor in successful_df.columns:
                        f.write(f"{factor}: {successful_df[factor].min():.3f} - {successful_df[factor].max():.3f}\n")
                
                f.write(f"\nCorrelation with brake distance:\n")
                f.write("-" * 35 + "\n")
                correlations = successful_df[['brake_distance', 'applied_mass', 'applied_drag_coefficient', 
                                           'applied_tire_friction', 'applied_brake_torque_front', 
                                           'applied_road_friction']].corr()['brake_distance'].sort_values(key=abs, ascending=False)
                
                for factor, corr in correlations.items():
                    if factor != 'brake_distance':
                        f.write(f"{factor}: {corr:.3f}\n")
            
            print(f"Summary statistics saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error generating summary stats: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run CARLA core factors brake distance experiments')
    parser.add_argument('--parameter-sets-dir', default='parameter_generation/parameter_output/parameter_sets', 
                       help='Directory containing generated parameter sets')
    parser.add_argument('--output-dir', default='core_experiment_results',
                       help='Directory to save experimental results')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting parameter set index')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='Ending parameter set index (exclusive)')
    parser.add_argument('--carla-host', default='localhost',
                       help='CARLA server host')
    parser.add_argument('--carla-port', type=int, default=2000,
                       help='CARLA server port')
    
    args = parser.parse_args()
    
    try:
        experiment = CarlaCoreBrakeExperiment(
            parameter_sets_dir=args.parameter_sets_dir,
            output_dir=args.output_dir
        )
        
        experiment.connect_to_carla(args.carla_host, args.carla_port)
        
        experiment.run_experiment_batch(
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Core factors experiment completed.")


if __name__ == "__main__":
    main()