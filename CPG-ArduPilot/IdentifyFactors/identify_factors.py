#!/usr/bin/env python3
"""
Identify internal and external factors from ArduPilot documentation and binary.
This script parses HTML documentation, performs static analysis on the binary using LLVM tools,
and applies heuristic filtering to generate a list of relevant factors.
"""

import re
import os
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import subprocess
from typing import List, Set, Dict, Tuple

class FactorIdentifier:
    def __init__(self, doc_path: str, binary_path: str = None):
        """
        Initialize the factor identifier.
        
        Args:
            doc_path: Path to the HTML documentation file
            binary_path: Path to ArduPilot binary for static analysis
        """
        self.doc_path = doc_path
        self.binary_path = binary_path or "<RVSPEC_ROOT>/UAV/ardupilot/build/sitl/bin/arducopter"
        
        # Create results directory if it doesn't exist
        self.results_dir = "./results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Domain-specific keywords for identifying relevant factors
        self.keywords = {
            # Environmental factors
            'wind', 'temperature', 'temp', 'pressure', 'magnetic', 'mag',
            'turbulence', 'turb', 'drift', 'glitch', 'noise', 'random',
            
            # Sensor-related
            'gps', 'accelerometer', 'acc', 'accel', 'gyroscope', 'gyro', 'gyr',
            'barometer', 'baro', 'bar', 'magnetometer', 'imu', 'compass',
            
            # Physical properties
            'altitude', 'alt', 'speed', 'velocity', 'position', 'pos',
            'direction', 'dir', 'orientation', 'angle',
            
            # Disturbances and errors
            'error', 'err', 'offset', 'bias', 'variation', 'deviation',
            'disturbance', 'interference', 'fail', 'failure'
        }
        
        # Synonyms mapping for merging
        self.synonyms = {
            'acc': 'accelerometer',
            'accel': 'accelerometer',
            'gyr': 'gyroscope',
            'gyro': 'gyroscope',
            'bar': 'barometer',
            'baro': 'barometer',
            'mag': 'magnetometer',
            'temp': 'temperature',
            'alt': 'altitude',
            'pos': 'position',
            'dir': 'direction',
            'turb': 'turbulence',
            'err': 'error'
        }
        
        # Patterns to exclude (non-physical parameters)
        self.exclude_patterns = [
            r'.*_LOG_.*',      # Logging parameters
            r'.*_DEBUG_.*',    # Debug parameters
            r'.*_ENABLE$',     # Enable flags
            r'.*_TYPE$',       # Type selectors
            r'.*_PIN$',        # Pin assignments
            r'.*_CHAN.*',      # Channel configurations
            r'.*_SCHED_.*',    # Scheduler parameters
            r'.*_PARAM_.*',    # Parameter system
            r'.*_FORMAT_.*',   # Format specifiers
        ]

    def parse_documentation(self) -> Set[str]:
        """
        Parse HTML documentation to extract SIM_* parameters.
        
        Returns:
            Set of candidate factor names
        """
        candidates = set()
        
        try:
            with open(self.doc_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Find all SIM_* parameters
            sim_params = soup.find_all('span', id=re.compile(r'^sim-.*'))
            
            for param_span in sim_params:
                # Extract parameter name from id
                param_id = param_span.get('id', '')
                if param_id:
                    # Convert id format (sim-wind-spd) to parameter name (SIM_WIND_SPD)
                    param_name = 'SIM_' + param_id.replace('sim-', '').replace('-', '_').upper()
                    
                    # Check if parameter contains relevant keywords
                    param_lower = param_name.lower()
                    if any(keyword in param_lower for keyword in self.keywords):
                        candidates.add(param_name)
            
            # Also look for parameter definitions in tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        text = cell.get_text()
                        # Look for SIM_* parameters
                        sim_matches = re.findall(r'SIM_[A-Z0-9_]+', text)
                        for match in sim_matches:
                            match_lower = match.lower()
                            if any(keyword in match_lower for keyword in self.keywords):
                                candidates.add(match)
            
            # Write documentation results
            doc_results_path = os.path.join(self.results_dir, 'doc-results.txt')
            with open(doc_results_path, 'w') as f:
                f.write("# Candidates from documentation parsing\n")
                for candidate in sorted(candidates):
                    f.write(f"{candidate}\n")
            
            print(f"Found {len(candidates)} candidates from documentation")
            print(f"Saved to: {doc_results_path}")
            
        except Exception as e:
            print(f"Error parsing documentation: {e}")
        
        return candidates

    def static_analysis(self) -> Set[str]:
        """
        Perform static analysis on ArduPilot binary using LLVM tools.
        
        Returns:
            Set of symbols from binary
        """
        symbols = set()
        raw_symbols = []
        
        try:
            # Use llvm-nm to extract symbol table from binary
            print(f"Running llvm-nm on {self.binary_path}...")
            
            # Check if llvm-nm is available, otherwise try nm
            nm_command = None
            if subprocess.run(['which', 'llvm-nm'], capture_output=True).returncode == 0:
                nm_command = 'llvm-nm'
            elif subprocess.run(['which', 'nm'], capture_output=True).returncode == 0:
                nm_command = 'nm'
            else:
                print("Warning: Neither llvm-nm nor nm found. Trying to continue with nm...")
                nm_command = 'nm'
            
            # Run nm command to get symbol table
            result = subprocess.run(
                [nm_command, '--demangle', self.binary_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Warning: {nm_command} returned non-zero exit code")
            
            # Parse the output
            lines = result.stdout.split('\n')
            
            # Process each symbol
            for line in lines:
                if not line.strip():
                    continue
                
                # nm output format: address type symbol_name
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    symbol_name = parts[2]
                    raw_symbols.append(symbol_name)
                    
                    # Look for relevant symbols containing our keywords
                    symbol_lower = symbol_name.lower()
                    
                    # Extract potential factor names from symbols
                    # Look for SITL-related symbols
                    if 'sitl' in symbol_lower or 'sim' in symbol_lower:
                        # Check if contains relevant keywords
                        if any(keyword in symbol_lower for keyword in self.keywords):
                            # Try to extract parameter names
                            # Look for patterns like SIM_WIND_SPD, etc.
                            sim_matches = re.findall(r'SIM_[A-Z0-9_]+', symbol_name)
                            for match in sim_matches:
                                symbols.add(match)
                            
                            # Also look for field names that might be factors
                            if 'wind' in symbol_lower or 'gps' in symbol_lower or 'noise' in symbol_lower:
                                # Extract the relevant part
                                field_matches = re.findall(r'(\w+)(?:_noise|_drift|_error|_glitch)', symbol_name, re.IGNORECASE)
                                for match in field_matches:
                                    if match.upper().startswith('SIM_'):
                                        symbols.add(match.upper())
                                    else:
                                        symbols.add(f"SIM_{match.upper()}")
            
            # Additionally, use objdump to get more detailed information
            if subprocess.run(['which', 'objdump'], capture_output=True).returncode == 0:
                print("Running objdump for additional analysis...")
                objdump_result = subprocess.run(
                    ['objdump', '-t', self.binary_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if objdump_result.returncode == 0:
                    for line in objdump_result.stdout.split('\n'):
                        # Look for SITL-related symbols
                        if 'SITL' in line or 'SIM_' in line:
                            # Extract SIM_ parameters
                            sim_matches = re.findall(r'SIM_[A-Z0-9_]+', line)
                            for match in sim_matches:
                                match_lower = match.lower()
                                if any(keyword in match_lower for keyword in self.keywords):
                                    symbols.add(match)
            
            # Write all raw symbols for debugging
            symbols_raw_path = os.path.join(self.results_dir, 'symbols_raw.txt')
            with open(symbols_raw_path, 'w') as f:
                f.write("# Raw symbols from binary analysis\n")
                for symbol in raw_symbols[:1000]:  # Limit output for readability
                    f.write(f"{symbol}\n")
                if len(raw_symbols) > 1000:
                    f.write(f"\n# ... and {len(raw_symbols) - 1000} more symbols\n")
            
            # Write filtered symbols to symbols.txt
            symbols_path = os.path.join(self.results_dir, 'symbols.txt')
            with open(symbols_path, 'w') as f:
                f.write("# Filtered symbols from static analysis\n")
                f.write(f"# Total raw symbols: {len(raw_symbols)}\n")
                f.write(f"# Filtered symbols: {len(symbols)}\n\n")
                for symbol in sorted(symbols):
                    f.write(f"{symbol}\n")
            
            print(f"Found {len(raw_symbols)} total symbols, filtered to {len(symbols)} relevant symbols")
            print(f"Saved to: {symbols_path}")
            
        except subprocess.TimeoutExpired:
            print("Error: Static analysis timed out")
        except FileNotFoundError:
            print(f"Error: Binary not found at {self.binary_path}")
        except Exception as e:
            print(f"Error in static analysis: {e}")
        
        return symbols

    def apply_heuristics(self, candidates: Set[str]) -> Set[str]:
        """
        Apply heuristic rules to filter and merge candidates.
        
        Args:
            candidates: Set of candidate factor names
            
        Returns:
            Filtered set of factor names
        """
        filtered = set()
        
        for candidate in candidates:
            # Skip if matches exclude patterns
            if any(re.match(pattern, candidate) for pattern in self.exclude_patterns):
                continue
            
            # Skip non-SIM parameters for now (focus on environmental factors)
            if not candidate.startswith('SIM_'):
                continue
            
            # Apply additional filters
            # Skip parameters that are purely configuration (not physical)
            skip_keywords = ['ENABLE', 'TYPE', 'PIN', 'CHAN', 'SCHED', 'PARAM', 'FORMAT']
            if any(skip in candidate for skip in skip_keywords):
                continue
            
            filtered.add(candidate)
        
        print(f"After heuristic filtering: {len(filtered)} factors")
        return filtered

    def generate_output(self, factors: Set[str]) -> None:
        """
        Generate the final CSV output with only the Parameter column.
        
        Args:
            factors: Final set of factor names
        """
        # Sort factors for consistent output
        sorted_factors = sorted(factors)
        
        # Write to CSV
        csv_path = os.path.join(self.results_dir, 'ArduPilot_SITL_Factors.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter'])  # Header
            for factor in sorted_factors:
                writer.writerow([factor])
        
        print(f"Written {len(sorted_factors)} factors to {csv_path}")
        
        # Also write a simple text file for reference
        txt_path = os.path.join(self.results_dir, 'identified_factors.txt')
        with open(txt_path, 'w') as f:
            f.write("# Final identified internal and external factors\n")
            f.write(f"# Total: {len(sorted_factors)} factors\n\n")
            for factor in sorted_factors:
                f.write(f"{factor}\n")
        
        print(f"Also saved text version to {txt_path}")

    def run(self) -> None:
        """
        Run the complete factor identification pipeline.
        """
        print("="*60)
        print("Starting Factor Identification Pipeline")
        print(f"All results will be saved to: {self.results_dir}/")
        print("="*60)
        
        # Step 1: Parse documentation
        print("\n[Step 1] Parsing documentation...")
        doc_candidates = self.parse_documentation()
        
        # Step 2: Static analysis on binary
        print("\n[Step 2] Performing static analysis on binary...")
        symbols = self.static_analysis()
        
        # Step 3: Combine candidates
        print("\n[Step 3] Combining candidates from doc-results.txt and symbols.txt...")
        all_candidates = doc_candidates.union(symbols)
        print(f"Total candidates before filtering: {len(all_candidates)}")
        
        # Step 4: Apply heuristics with keywords
        print("\n[Step 4] Applying heuristic filtering with keywords...")
        filtered_factors = self.apply_heuristics(all_candidates)
        
        # Step 5: Generate output
        print("\n[Step 5] Generating output files...")
        self.generate_output(filtered_factors)
        
        print("\n" + "="*60)
        print("Factor Identification Complete!")
        print(f"Results saved in: {self.results_dir}/")
        print(f"Main output: {self.results_dir}/ArduPilot_SITL_Factors.csv")
        print("="*60)


def main():
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description='Identify internal and external factors'
    )
    
    parser.add_argument(
        '--ardupilot-path',
        type=str,
        default='<RVSPEC_ROOT>/UAV/ardupilot',
        help='Path to ArduPilot root directory (default: <RVSPEC_ROOT>/UAV/ardupilot)'
    )
    
    parser.add_argument(
        '--doc-path',
        type=str,
        default='<RVSPEC_ROOT>/CPG-ArduPilot/IdentifyFactors/docs/Complete Parameter List — Copter documentation.html',
        help='Path to HTML documentation file'
    )
    
    args = parser.parse_args()
    
    # Construct binary path from ArduPilot root
    # Default to arducopter binary
    binary_path = os.path.join(args.ardupilot_path, 'build', 'sitl', 'bin', 'arducopter')
    doc_path = os.path.join(args.doc_path, 'Complete Parameter List — Copter documentation.html')
    
    # Check if documentation exists
    if not os.path.exists(doc_path):
        print(f"Error: Documentation file not found at {doc_path}")
        return
    
    # Check if binary exists
    if not os.path.exists(binary_path):
        print(f"Warning: Binary file not found at {binary_path}")
        print("Static analysis will be limited.")
    
    # Run the factor identifier
    identifier = FactorIdentifier(doc_path, binary_path)
    identifier.run()


if __name__ == "__main__":
    main()