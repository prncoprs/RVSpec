#!/usr/bin/env python3
"""
Cyber-Physical Interplay Graph (CPG) Construction
==================================================
This script constructs a CPG from trained Random Forest models,
showing the relationships between environmental factors and physical states.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pyvis.network import Network
import networkx as nx
import json
from typing import Dict, List, Tuple

# Configuration
ANALYSIS_DIR = "<RVSPEC_ROOT>/CPG-ArduPilot/DataAnalysis/analysis_output_sitl"

# Flight mode mapping
PHASE_TO_FLIGHT_MODE = {
    1: "TAKEOFF",
    2: "ALT_HOLD",
    3: "FLIP",
    5: "CIRCLE",
    6: "LOITER",
    7: "DRIFT",
    9: "BRAKE",
    10: "GUIDED",
    11: "RTL",
    12: "LAND"
}

# Environmental factors in order
ENV_FACTORS = [
    "SIM_ACC1_RND",
    "SIM_ACC2_RND",
    "SIM_BAR2_DRIFT",
    "SIM_BAR2_GLITCH",
    "SIM_BAR2_RND",
    "SIM_BARO_DRIFT",
    "SIM_BARO_GLITCH",
    "SIM_BARO_RND",
    "SIM_BARO_WCF_BAK",
    "SIM_BARO_WCF_DN",
    "SIM_BARO_WCF_FWD",
    "SIM_BARO_WCF_LFT",
    "SIM_BARO_WCF_RGT",
    "SIM_BARO_WCF_UP",
    "SIM_GPS_DRIFTALT",
    "SIM_GPS_GLITCH_X",
    "SIM_GPS_GLITCH_Y",
    "SIM_GPS_GLITCH_Z",
    "SIM_GPS_NOISE",
    "SIM_GPS2_DRFTALT",
    "SIM_GPS2_GLTCH_X",
    "SIM_GPS2_GLTCH_Y",
    "SIM_GPS2_GLTCH_Z",
    "SIM_GPS2_NOISE",
    "SIM_GYR1_RND",
    "SIM_GYR2_RND",
    "SIM_MAG_RND",
    "SIM_TEMP_BFACTOR",
    "SIM_TEMP_BRD_OFF",
    "SIM_TEMP_START",
    "SIM_TEMP_TCONST",
    "SIM_WIND_DIR",
    "SIM_WIND_DIR_Z",
    "SIM_WIND_SPD",
    "SIM_WIND_T",
    "SIM_WIND_T_ALT",
    "SIM_WIND_T_COEF",
    "SIM_WIND_TURB"
]


class CPGConstructor:
    def __init__(self, analysis_dir: str):
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = self.analysis_dir / "cpg_outputs"
        self.neo4j_dir = self.output_dir / "neo4j_imports"
        self.output_dir.mkdir(exist_ok=True)
        self.neo4j_dir.mkdir(exist_ok=True)
        
        self.models_data = {}
        self.importance_matrix = {}
        
    def model_fitting(self):
        """Load Random Forest models and extract feature importances."""
        print("\n" + "="*60)
        print("[MODEL FITTING]")
        print("-"*60)
        print(f"Loading Random Forest models from: {self.analysis_dir}")
        print("\nProcessing flight modes:")
        
        total_models = 0
        
        for phase, flight_mode in PHASE_TO_FLIGHT_MODE.items():
            model_dir = self.analysis_dir / f"models_phase{phase}_rf"
            
            if not model_dir.exists():
                print(f" {flight_mode:<10} (Phase {phase}): Directory not found")
                continue
            
            # Find all .joblib files in the directory
            model_files = list(model_dir.glob("*.joblib"))
            
            if not model_files:
                print(f" {flight_mode:<10} (Phase {phase}): No models found")
                continue
            
            phase_models = {}
            
            for model_file in model_files:
                # Extract physical state name from filename
                # Format: ATT_DesPitch_mean_Phase1.joblib
                filename = model_file.stem  # Remove .joblib
                state_name = filename.replace(f"_Phase{phase}", "")
                
                # Load model
                try:
                    model_data = joblib.load(model_file)
                    model = model_data['model']
                    
                    # Extract feature importances
                    importances = model.feature_importances_
                    
                    # Normalize importances to sum to 1
                    importances_normalized = importances / importances.sum()
                    
                    # Store with full node name
                    node_name = f"{flight_mode}_{state_name}"
                    phase_models[node_name] = {
                        'state': state_name,
                        'flight_mode': flight_mode,
                        'phase': phase,
                        'importances': importances_normalized,
                        'model_path': str(model_file)
                    }
                    
                except Exception as e:
                    print(f"  Error loading {model_file.name}: {e}")
                    continue
            
            if phase_models:
                self.models_data.update(phase_models)
                print(f" {flight_mode:<10} (Phase {phase}): {len(phase_models):3} models loaded")
                total_models += len(phase_models)
        
        print(f"\nTotal: {total_models} models processed")
        print("Feature importance extraction complete")
        
        # Create importance matrix
        self._create_importance_matrix()
        
    def _create_importance_matrix(self):
        """Create a matrix of importances for all factor-state pairs."""
        rows = []
        
        for node_name, model_info in self.models_data.items():
            importances = model_info['importances']
            
            for i, factor in enumerate(ENV_FACTORS):
                rows.append({
                    'Factor': factor,
                    'PhysicalState': node_name,
                    'FlightMode': model_info['flight_mode'],
                    'State': model_info['state'],
                    'Importance': importances[i]
                })
        
        self.importance_matrix = pd.DataFrame(rows)
        
        # Save importance matrix
        matrix_path = self.output_dir / "importance_all_phases.csv"
        self.importance_matrix.to_csv(matrix_path, index=False)
        
    def graph_construction(self):
        """Construct the Cyber-Physical Interplay Graph."""
        print("\n" + "="*60)
        print("[CYBER-PHYSICAL INTERPLAY GRAPH CONSTRUCTION]")
        print("-"*60)
        print("Building combined Cyber-Physical Interplay Graph...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add factor nodes (environmental factors)
        for factor in ENV_FACTORS:
            G.add_node(factor, 
                      node_type='factor',
                      color='#FFF2CC',  # Yellow
                      label=factor)
        
        # Add physical state nodes and edges
        for node_name, model_info in self.models_data.items():
            # Add physical state node
            G.add_node(node_name,
                      node_type='physical_state',
                      color='#DAE8FC',  # Blue
                      label=f"{model_info['state']}\n[{model_info['flight_mode']}]",
                      flight_mode=model_info['flight_mode'],
                      state=model_info['state'])
            
            # Add edges from factors to this physical state
            for i, factor in enumerate(ENV_FACTORS):
                importance = model_info['importances'][i]
                
                # Only add edges with non-zero importance
                if importance > 0:
                    G.add_edge(factor, node_name,
                              weight=importance,
                              label=f"{importance:.3f}")
        
        print(f"\nGraph Statistics:")
        print(f"- Factor nodes: {len(ENV_FACTORS)}")
        print(f"- Physical state nodes: {len(self.models_data)}")
        print(f"- Total weighted edges: {G.number_of_edges()}")
        print(f"- All edge importance values included")
        
        # Save graph
        self.G = G
        self._visualize_pyvis()
        self._export_neo4j()
        
    def _visualize_pyvis(self):
        """Create interactive visualization using Pyvis."""
        net = Network(height="900px", width="100%", 
                     bgcolor="#ffffff", 
                     font_color="black",
                     directed=True)
        
        # Configure physics for better layout
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 12
            }
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "font": {
              "size": 10,
              "align": "middle"
            }
          },
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "nodeSpacing": 200,
              "levelSeparation": 400
            }
          },
          "physics": {
            "enabled": false
          }
        }
        """)
        
        # Add nodes
        for node in self.G.nodes(data=True):
            node_id = node[0]
            node_attrs = node[1]
            
            # Set node position hint for hierarchical layout
            if node_attrs['node_type'] == 'factor':
                level = 0
            else:
                level = 1
            
            net.add_node(node_id, 
                        label=node_attrs['label'],
                        color=node_attrs['color'],
                        level=level,
                        shape="box")
        
        # Add edges with importance labels
        for edge in self.G.edges(data=True):
            source = edge[0]
            target = edge[1]
            edge_attrs = edge[2]
            
            # Scale edge width based on importance
            width = max(1, edge_attrs['weight'] * 10)
            
            net.add_edge(source, target,
                        label=edge_attrs['label'],
                        width=width,
                        color={'color': '#666666'})
        
        # Save HTML
        html_path = self.output_dir / "cpg_combined.html"
        net.save_graph(str(html_path))
        print(f"\n Interactive visualization: {html_path}")
        
    def _export_neo4j(self):
        """Export graph data for Neo4j import."""
        # Create nodes CSV
        nodes_data = []
        
        # Add factor nodes
        for factor in ENV_FACTORS:
            nodes_data.append({
                'nodeId': factor,
                'label': factor,
                'type': 'Factor',
                'color': '#FFF2CC',
                'flight_mode': ''
            })
        
        # Add physical state nodes
        for node_name, model_info in self.models_data.items():
            nodes_data.append({
                'nodeId': node_name,
                'label': model_info['state'],
                'type': 'PhysicalState',
                'color': '#DAE8FC',
                'flight_mode': model_info['flight_mode']
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_path = self.neo4j_dir / "all_nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)
        
        # Create edges CSV
        edges_data = []
        for edge in self.G.edges(data=True):
            edges_data.append({
                'source': edge[0],
                'target': edge[1],
                'weight': edge[2]['weight'],
                'importance_label': edge[2]['label']
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_path = self.neo4j_dir / "all_edges.csv"
        edges_df.to_csv(edges_path, index=False)
        
        # Create Cypher import script
        cypher_script = """// Cyber-Physical Interplay Graph Import Script
// =============================================

// Clear existing data (optional - comment out if you want to preserve existing data)
// MATCH (n) DETACH DELETE n;

// Create constraints for unique IDs
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Factor) REQUIRE f.nodeId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (ps:PhysicalState) REQUIRE ps.nodeId IS UNIQUE;

// Load Factor nodes
LOAD CSV WITH HEADERS FROM 'file:///all_nodes.csv' AS row
WITH row WHERE row.type = 'Factor'
CREATE (f:Factor {
    nodeId: row.nodeId,
    name: row.label,
    color: row.color
});

// Load PhysicalState nodes
LOAD CSV WITH HEADERS FROM 'file:///all_nodes.csv' AS row
WITH row WHERE row.type = 'PhysicalState'
CREATE (ps:PhysicalState {
    nodeId: row.nodeId,
    name: row.label,
    flight_mode: row.flight_mode,
    color: row.color
});

// Load relationships
LOAD CSV WITH HEADERS FROM 'file:///all_edges.csv' AS row
MATCH (f:Factor {nodeId: row.source})
MATCH (ps:PhysicalState {nodeId: row.target})
CREATE (f)-[:INFLUENCES {
    weight: toFloat(row.weight),
    importance: row.importance_label
}]->(ps);

// Create indexes for better query performance
CREATE INDEX IF NOT EXISTS FOR (f:Factor) ON (f.name);
CREATE INDEX IF NOT EXISTS FOR (ps:PhysicalState) ON (ps.name);
CREATE INDEX IF NOT EXISTS FOR (ps:PhysicalState) ON (ps.flight_mode);

// Verify import
MATCH (f:Factor)
WITH count(f) as factorCount
MATCH (ps:PhysicalState)
WITH factorCount, count(ps) as stateCount
MATCH ()-[r:INFLUENCES]->()
RETURN factorCount as Factors, stateCount as PhysicalStates, count(r) as Relationships;
"""
        
        cypher_path = self.neo4j_dir / "import_all.cypher"
        with open(cypher_path, 'w') as f:
            f.write(cypher_script)
        
        print(f" Neo4j import files: {self.neo4j_dir}")
        
        # Print Neo4j instructions
        print("\nNeo4j Import Instructions:")
        print("1. Copy files to Neo4j import directory:")
        print(f"   sudo cp {self.neo4j_dir}/*.csv /var/lib/neo4j/import/")
        print("\n2. Run in Neo4j Browser (http://localhost:7474):")
        print(f"   - Copy and execute the commands from: {cypher_path}")
        print("   - Or run each section of the Cypher script separately")
        
    def run(self):
        """Execute the complete CPG construction pipeline."""
        print("="*60)
        print("    CYBER-PHYSICAL INTERPLAY GRAPH CONSTRUCTION")
        print("="*60)
        
        self.model_fitting()
        self.graph_construction()
        
        print("\n" + "="*60)
        print("                    COMPLETE")
        print("="*60)


def main():
    """Main execution function."""
    import sys
    
    # Allow custom analysis directory
    if len(sys.argv) > 1:
        analysis_dir = sys.argv[1]
    else:
        analysis_dir = ANALYSIS_DIR
    
    # Check if directory exists
    if not Path(analysis_dir).exists():
        print(f"Error: Analysis directory not found: {analysis_dir}")
        sys.exit(1)
    
    # Create and run CPG constructor
    cpg = CPGConstructor(analysis_dir)
    cpg.run()


if __name__ == "__main__":
    main()