#!/usr/bin/env python3
"""
CARLA Map Explorer
Analyzes all available CARLA maps to identify highways, intersections, and roundabouts
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import carla
import numpy as np


class MapAnalyzer:
    def __init__(self, host='127.0.0.1', port=2000, output_dir='map_analysis'):
        """
        Initialize the map analyzer
        
        Args:
            host: CARLA server host
            port: CARLA server port
            output_dir: Directory to save analysis results
        """
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / 'screenshots').mkdir(exist_ok=True)
        
        self.client = None
        self.world = None
        
    def connect_to_carla(self):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            print(f"Connected to CARLA server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to CARLA server: {e}")
            return False
    
    def get_available_maps(self) -> List[str]:
        """Get list of all available maps"""
        try:
            maps = self.client.get_available_maps()
            # Filter out empty or invalid maps and clean the names
            valid_maps = []
            for m in maps:
                if m and not m.endswith('.xodr'):
                    # Keep the full path as CARLA needs it, but we'll clean it for display
                    valid_maps.append(m)
            print(f"Found {len(valid_maps)} available maps")
            return valid_maps
        except Exception as e:
            print(f"Error getting available maps: {e}")
            return []
    
    def analyze_road_topology(self, world_map) -> Dict[str, Any]:
        """
        Analyze road topology to identify different road types
        
        Args:
            world_map: CARLA map object
            
        Returns:
            Dictionary containing road analysis results
        """
        topology = world_map.get_topology()
        waypoints = world_map.generate_waypoints(distance=2.0)
        
        # Initialize counters
        analysis = {
            'total_waypoints': len(waypoints),
            'total_road_segments': len(topology),
            'highway_segments': 0,
            'intersection_segments': 0,
            'roundabout_segments': 0,
            'junction_count': 0,
            'lane_types': {},
            'road_types': {},
            'max_speed_zones': {},
            'spawn_points_count': 0,
            'waypoint_details': []
        }
        
        # Analyze waypoints
        junctions_found = set()
        lane_types_count = {}
        road_types_count = {}
        speed_zones = {}
        
        for waypoint in waypoints:
            # Lane type analysis
            lane_type = str(waypoint.lane_type)
            lane_types_count[lane_type] = lane_types_count.get(lane_type, 0) + 1
            
            # Road type analysis (based on lane markings and characteristics)
            road_id = waypoint.road_id
            
            # Check for highway characteristics
            is_highway = (
                waypoint.lane_type == carla.LaneType.Driving and
                len(waypoint.next(10.0)) > 0 and  # Has continuation
                not waypoint.is_junction
            )
            
            # Check for intersections
            is_intersection = waypoint.is_junction
            if is_intersection and waypoint.junction_id is not None:
                junctions_found.add(waypoint.junction_id)
            
            # Speed limit analysis
            speed_limit = waypoint.transform.location
            # This is a simplification - CARLA doesn't directly provide speed limits
            # We'll categorize based on road characteristics
            
            # Detect potential roundabouts (circular roads)
            is_roundabout = False
            if waypoint.is_junction:
                # Check if junction has circular characteristics
                next_waypoints = waypoint.next(5.0)
                if next_waypoints:
                    # Simple heuristic: if we can go in multiple directions, might be roundabout
                    if len(next_waypoints) > 1:
                        is_roundabout = True
            
            # Update counters
            if is_highway and not is_intersection:
                analysis['highway_segments'] += 1
            if is_intersection:
                analysis['intersection_segments'] += 1
            if is_roundabout:
                analysis['roundabout_segments'] += 1
            
            # Store detailed waypoint info (sample)
            if len(analysis['waypoint_details']) < 100:  # Limit to first 100 for storage
                analysis['waypoint_details'].append({
                    'road_id': waypoint.road_id,
                    'lane_id': waypoint.lane_id,
                    'lane_type': lane_type,
                    'is_junction': is_intersection,
                    'junction_id': waypoint.junction_id if waypoint.junction_id else None,
                    'is_highway_candidate': is_highway,
                    'is_roundabout_candidate': is_roundabout,
                    'location': {
                        'x': waypoint.transform.location.x,
                        'y': waypoint.transform.location.y,
                        'z': waypoint.transform.location.z
                    }
                })
        
        # Finalize analysis
        analysis['junction_count'] = len(junctions_found)
        analysis['lane_types'] = lane_types_count
        analysis['unique_junctions'] = list(junctions_found)
        
        # Calculate percentages
        if analysis['total_waypoints'] > 0:
            analysis['highway_percentage'] = (analysis['highway_segments'] / analysis['total_waypoints']) * 100
            analysis['intersection_percentage'] = (analysis['intersection_segments'] / analysis['total_waypoints']) * 100
            analysis['roundabout_percentage'] = (analysis['roundabout_segments'] / analysis['total_waypoints']) * 100
        
        return analysis
    
    def get_spawn_points_info(self, world_map) -> Dict[str, Any]:
        """Get information about spawn points"""
        spawn_points = world_map.get_spawn_points()
        
        spawn_info = {
            'total_spawn_points': len(spawn_points),
            'spawn_locations': []
        }
        
        for i, spawn_point in enumerate(spawn_points):
            location = spawn_point.location
            
            # Try to get nearest waypoint to understand road context
            nearest_waypoint = world_map.get_waypoint(location)
            
            spawn_data = {
                'index': i,
                'location': {
                    'x': location.x,
                    'y': location.y,
                    'z': location.z
                },
                'rotation': {
                    'pitch': spawn_point.rotation.pitch,
                    'yaw': spawn_point.rotation.yaw,
                    'roll': spawn_point.rotation.roll
                }
            }
            
            if nearest_waypoint:
                spawn_data['road_context'] = {
                    'road_id': nearest_waypoint.road_id,
                    'lane_id': nearest_waypoint.lane_id,
                    'lane_type': str(nearest_waypoint.lane_type),
                    'is_junction': nearest_waypoint.is_junction,
                    'junction_id': nearest_waypoint.junction_id if nearest_waypoint.junction_id else None
                }
            
            spawn_info['spawn_locations'].append(spawn_data)
        
        return spawn_info
    
    def categorize_map(self, analysis: Dict[str, Any], spawn_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize map based on analysis results
        
        Returns:
            Dictionary with map categorization
        """
        categorization = {
            'primary_type': 'unknown',
            'secondary_types': [],
            'suitability_scores': {
                'highway': 0,
                'intersection': 0,
                'roundabout': 0
            },
            'recommended_spawn_points': {
                'highway': [],
                'intersection': [],
                'roundabout': []
            }
        }
        
        # Calculate suitability scores (0-100)
        highway_score = min(100, analysis.get('highway_percentage', 0) * 2)
        intersection_score = min(100, analysis.get('intersection_percentage', 0) * 5)  # Weight intersections more
        roundabout_score = min(100, analysis.get('roundabout_percentage', 0) * 10)  # Weight roundabouts heavily
        
        categorization['suitability_scores']['highway'] = highway_score
        categorization['suitability_scores']['intersection'] = intersection_score
        categorization['suitability_scores']['roundabout'] = roundabout_score
        
        # Determine primary type based on highest score
        scores = categorization['suitability_scores']
        primary_type = max(scores, key=scores.get)
        categorization['primary_type'] = primary_type
        
        # Determine secondary types (scores > 20)
        secondary_types = [t for t, score in scores.items() if score > 20 and t != primary_type]
        categorization['secondary_types'] = secondary_types
        
        # Recommend spawn points based on road context
        for spawn in spawn_info['spawn_locations']:
            spawn_idx = spawn['index']
            road_context = spawn.get('road_context', {})
            
            # Highway spawn points (not at junctions, driving lanes)
            if (not road_context.get('is_junction', True) and 
                road_context.get('lane_type') == 'LaneType.Driving'):
                categorization['recommended_spawn_points']['highway'].append(spawn_idx)
            
            # Intersection spawn points (at junctions)
            if road_context.get('is_junction', False):
                categorization['recommended_spawn_points']['intersection'].append(spawn_idx)
                # Could also be roundabout
                categorization['recommended_spawn_points']['roundabout'].append(spawn_idx)
        
        return categorization
    
    def take_overview_screenshot(self, map_name: str):
        """
        Take an overview screenshot of the map
        Note: This is a placeholder - actual screenshot implementation would require
        setting up a camera and capturing images
        """
        # Clean map name for filename (remove path and invalid characters)
        clean_map_name = map_name.replace('/Game/Carla/Maps/', '').replace('/', '_')
        
        # For now, we'll create a placeholder file
        screenshot_path = self.output_dir / 'screenshots' / f'{clean_map_name}_overview.txt'
        with open(screenshot_path, 'w') as f:
            f.write(f"Screenshot placeholder for {map_name}\n")
            f.write("To implement actual screenshots, set up a camera sensor and capture images\n")
        
        print(f"Screenshot placeholder saved: {screenshot_path}")
    
    def analyze_single_map(self, map_name: str) -> Dict[str, Any]:
        """Analyze a single map"""
        # Clean map name for display
        display_name = map_name.replace('/Game/Carla/Maps/', '')
        print(f"\nAnalyzing map: {display_name}")
        
        try:
            # Load the map
            self.world = self.client.load_world(map_name)
            time.sleep(2)  # Wait for map to load
            
            world_map = self.world.get_map()
            
            # Perform analysis
            print("  - Analyzing road topology...")
            topology_analysis = self.analyze_road_topology(world_map)
            
            print("  - Analyzing spawn points...")
            spawn_info = self.get_spawn_points_info(world_map)
            
            print("  - Categorizing map...")
            categorization = self.categorize_map(topology_analysis, spawn_info)
            
            print("  - Taking overview screenshot...")
            self.take_overview_screenshot(map_name)
            
            # Compile results
            map_analysis = {
                'map_name': map_name,
                'display_name': display_name,
                'analysis_timestamp': time.time(),
                'topology_analysis': topology_analysis,
                'spawn_info': spawn_info,
                'categorization': categorization,
                'summary': {
                    'total_waypoints': topology_analysis['total_waypoints'],
                    'total_spawn_points': spawn_info['total_spawn_points'],
                    'junction_count': topology_analysis['junction_count'],
                    'primary_type': categorization['primary_type'],
                    'suitability_scores': categorization['suitability_scores']
                }
            }
            
            print(f"  - Complete! Primary type: {categorization['primary_type']}")
            print(f"    Scores - Highway: {categorization['suitability_scores']['highway']:.1f}, "
                  f"Intersection: {categorization['suitability_scores']['intersection']:.1f}, "
                  f"Roundabout: {categorization['suitability_scores']['roundabout']:.1f}")
            
            return map_analysis
            
        except Exception as e:
            print(f"  - Error analyzing {display_name}: {e}")
            return {
                'map_name': map_name,
                'display_name': display_name,
                'error': str(e),
                'analysis_timestamp': time.time()
            }
    
    def analyze_all_maps(self) -> Dict[str, Any]:
        """Analyze all available maps"""
        if not self.connect_to_carla():
            return {}
        
        maps = self.get_available_maps()
        if not maps:
            print("No maps found!")
            return {}
        
        print(f"Starting analysis of {len(maps)} maps...")
        
        all_results = {}
        
        for map_name in maps:
            result = self.analyze_single_map(map_name)
            all_results[map_name] = result
        
        return all_results
    
    def generate_recommendations(self, all_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate map recommendations for each scenario type"""
        recommendations = {
            'highway_maps': [],
            'intersection_maps': [],
            'roundabout_maps': []
        }
        
        # Sort maps by suitability scores for each type
        for map_name, result in all_results.items():
            if 'error' in result:
                continue
            
            scores = result.get('categorization', {}).get('suitability_scores', {})
            
            # Add to category if score is above threshold
            if scores.get('highway', 0) > 30:
                recommendations['highway_maps'].append((map_name, scores['highway']))
            if scores.get('intersection', 0) > 30:
                recommendations['intersection_maps'].append((map_name, scores['intersection']))
            if scores.get('roundabout', 0) > 20:  # Lower threshold for roundabouts
                recommendations['roundabout_maps'].append((map_name, scores['roundabout']))
        
        # Sort by score and extract map names
        for category in recommendations:
            recommendations[category].sort(key=lambda x: x[1], reverse=True)
            recommendations[category] = [map_name for map_name, score in recommendations[category]]
        
        return recommendations
    
    def save_results(self, all_results: Dict[str, Any]):
        """Save analysis results to files"""
        # Save detailed analysis
        detailed_file = self.output_dir / 'map_topology_report.json'
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Detailed analysis saved to: {detailed_file}")
        
        # Generate and save recommendations
        recommendations = self.generate_recommendations(all_results)
        recommendations_file = self.output_dir / 'recommended_maps.yaml'
        
        # Create YAML-like content manually (to avoid dependency)
        yaml_content = "# CARLA Map Recommendations\n"
        yaml_content += "# Generated by map_explorer.py\n\n"
        
        for category, maps in recommendations.items():
            yaml_content += f"{category}:\n"
            for map_name in maps:
                yaml_content += f"  - {map_name}\n"
            yaml_content += "\n"
        
        with open(recommendations_file, 'w') as f:
            f.write(yaml_content)
        print(f"Map recommendations saved to: {recommendations_file}")
        
        # Create summary report
        summary_file = self.output_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("CARLA Map Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total maps analyzed: {len(all_results)}\n\n")
            
            # Count by primary type
            type_counts = {}
            for result in all_results.values():
                if 'error' not in result:
                    primary_type = result.get('categorization', {}).get('primary_type', 'unknown')
                    type_counts[primary_type] = type_counts.get(primary_type, 0) + 1
            
            f.write("Maps by primary type:\n")
            for map_type, count in type_counts.items():
                f.write(f"  {map_type}: {count}\n")
            f.write("\n")
            
            f.write("Recommended maps:\n")
            for category, maps in recommendations.items():
                f.write(f"  {category}: {', '.join(maps[:3])}...\n")  # Show first 3
            
        print(f"Summary report saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze CARLA maps for highway, intersection, and roundabout scenarios')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--output-dir', default='map_analysis', help='Output directory for analysis results')
    parser.add_argument('--maps', nargs='+', help='Specific maps to analyze (default: all available)')
    
    args = parser.parse_args()
    
    analyzer = MapAnalyzer(host=args.host, port=args.port, output_dir=args.output_dir)
    
    if args.maps:
        # Analyze specific maps
        print(f"Analyzing specified maps: {args.maps}")
        if not analyzer.connect_to_carla():
            return
        
        all_results = {}
        for map_name in args.maps:
            result = analyzer.analyze_single_map(map_name)
            all_results[map_name] = result
    else:
        # Analyze all maps
        all_results = analyzer.analyze_all_maps()
    
    if all_results:
        analyzer.save_results(all_results)
        print(f"\nAnalysis complete! Results saved in: {analyzer.output_dir}")
        print("Check 'recommended_maps.yaml' for curated map recommendations.")
    else:
        print("No analysis results generated. Check CARLA server connection.")


if __name__ == "__main__":
    main()