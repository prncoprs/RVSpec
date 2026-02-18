#!/usr/bin/env python3
"""
Filtered CPG Visualization with Pyvis
======================================
Creates an interactive HTML visualization of the CPG filtered for:
- Flight mode: LAND
- Physical states containing 'SIM2'
"""

import pandas as pd
from pathlib import Path
from pyvis.network import Network
import numpy as np

def create_filtered_cpg_visualization(cpg_dir, output_file="cpg_land_sim2.html"):
    """
    Create a filtered interactive CPG visualization.
    
    Args:
        cpg_dir: Path to cpg_outputs directory
        output_file: Name of output HTML file
    """
    cpg_path = Path(cpg_dir)
    neo4j_dir = cpg_path / "neo4j_imports"
    
    # Load the data
    print("Loading CPG data...")
    nodes_df = pd.read_csv(neo4j_dir / "all_nodes.csv")
    edges_df = pd.read_csv(neo4j_dir / "all_edges.csv")
    
    print(f"Total nodes: {len(nodes_df)}")
    print(f"Total edges: {len(edges_df)}")
    
    # Filter for LAND flight mode and SIM physical states
    print("\nFiltering for LAND flight mode and SIM physical states...")
    
    # Get physical states that match our criteria
    filtered_ps = nodes_df[
        (nodes_df['type'] == 'PhysicalState') & 
        (nodes_df['flight_mode'] == 'LAND') &
        (nodes_df['nodeId'].str.contains('SIM'))
    ]['nodeId'].tolist()
    
    print(f"Found {len(filtered_ps)} physical states matching criteria")
    
    if len(filtered_ps) == 0:
        print("No physical states found matching criteria!")
        print("\nChecking available flight modes:")
        print(nodes_df[nodes_df['type'] == 'PhysicalState']['flight_mode'].unique())
        print("\nSample physical state node IDs:")
        print(nodes_df[nodes_df['type'] == 'PhysicalState']['nodeId'].head(10))
        return
    
    # Filter edges to only include those connecting to our filtered physical states
    filtered_edges = edges_df[edges_df['target'].isin(filtered_ps)]
    
    # Get the factors that connect to these physical states
    connected_factors = filtered_edges['source'].unique()
    
    # Filter nodes to include only connected factors and filtered physical states
    filtered_nodes = nodes_df[
        (nodes_df['nodeId'].isin(connected_factors)) |
        (nodes_df['nodeId'].isin(filtered_ps))
    ]
    
    print(f"Filtered to {len(filtered_nodes)} nodes and {len(filtered_edges)} edges")
    
    # Create the network
    print("\nCreating Pyvis network...")
    net = Network(
        height="900px", 
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False,
        cdn_resources='in_line'
    )
    
    # Configure physics for better layout
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14,
          "face": "arial"
        },
        "shape": "box",
        "margin": 10
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.8
          }
        },
        "color": {
          "color": "#848484",
          "highlight": "#FF0000"
        },
        "font": {
          "size": 10,
          "align": "middle",
          "background": "white"
        },
        "smooth": {
          "type": "cubicBezier",
          "roundness": 0.5
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 200,
          "treeSpacing": 250,
          "levelSeparation": 600,
          "blockShifting": true,
          "edgeMinimization": false,
          "parentCentralization": false
        }
      },
      "physics": {
        "enabled": false,
        "hierarchicalRepulsion": {
          "nodeDistance": 300
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)
    
    # Define abbreviation to full name mapping
    abbrev_mapping = {
        'PN': 'PositionN',
        'PE': 'PositionE', 
        'PD': 'PositionD',
        'VN': 'VelocityN',
        'VE': 'VelocityE',
        'VD': 'VelocityD',
        'As': 'Airspeed',
        'ASpdU': 'ASpdU',  # Keep as is - Achieved Simulation Speedup
        'Pitch': 'Pitch',
        'Roll': 'Roll',
        'Yaw': 'Yaw'
    }
    
    def clean_label(label, is_physical_state=False):
        """Remove SIM_ or SIM2_ prefix from label and expand abbreviations for physical states"""
        # Remove prefix for display
        display_label = label
        if display_label.startswith('SIM_'):
            display_label = display_label[4:]
        elif display_label.startswith('SIM2_'):
            display_label = display_label[5:]
        
        # For physical states, expand abbreviations
        if is_physical_state:
            # Split by underscore to separate base name from suffix
            parts = display_label.split('_')
            if len(parts) >= 2:
                base_name = parts[0]
                suffix = '_'.join(parts[1:])  # Could be _mean or _std
                
                # Expand abbreviation if found
                if base_name in abbrev_mapping:
                    display_label = f"{abbrev_mapping[base_name]}_{suffix}"
        
        return display_label
    
    # Sort factors to create a staggered two-column layout
    factor_nodes = filtered_nodes[filtered_nodes['type'] == 'Factor'].copy()
    factor_nodes = factor_nodes.sort_values('nodeId')
    ps_nodes = filtered_nodes[filtered_nodes['type'] == 'PhysicalState'].copy()
    ps_nodes = ps_nodes.sort_values('nodeId')  # Sort physical states too
    
    # Add nodes with custom positioning
    print("Adding nodes to network...")
    
    # Add factors in a staggered two-column layout
    factor_count = 0
    for _, node in factor_nodes.iterrows():
        node_id = node['nodeId']
        original_label = node['label']
        display_label = clean_label(original_label, is_physical_state=False)
        color = node['color']
        
        # Create hover title with original label
        title = f"Factor: {original_label}"
        
        # Stagger factors in two columns
        if factor_count % 2 == 0:
            level = 1  # First column
        else:
            level = 2  # Second column (slightly to the right)
        
        net.add_node(
            node_id,
            label=display_label,  # Use cleaned label for display
            color=color,
            title=title,
            level=level,
            borderWidth=2,
            borderWidthSelected=4
        )
        factor_count += 1
    
    # Add physical state nodes, all in one column on the right
    for _, node in ps_nodes.iterrows():
        node_id = node['nodeId']
        original_label = node['label']
        display_label = clean_label(original_label, is_physical_state=True)
        color = node['color']
        flight_mode = node['flight_mode']
        
        # All physical states in the same level (far right)
        level = 4  # Use level 4 for more spacing from factors
        
        title = f"Physical State: {original_label}\nFlight Mode: {flight_mode}"
        
        net.add_node(
            node_id,
            label=display_label,  # Use cleaned and expanded label for display
            color=color,
            title=title,
            level=level,
            borderWidth=2,
            borderWidthSelected=4
        )
    
    # Add edges with importance labels
    print("Adding edges to network...")
    for _, edge in filtered_edges.iterrows():
        source = edge['source']
        target = edge['target']
        weight = float(edge['weight'])
        
        # Scale edge width based on importance
        width = max(1, weight * 20)  # Scale factor of 20 for visibility
        
        # Create edge label and title
        edge_label = f"{weight:.3f}"
        edge_title = f"Importance: {weight:.3f}"
        
        # Color based on importance level
        if weight > 0.1:
            edge_color = "#FF6B6B"  # Red for high importance
        elif weight > 0.05:
            edge_color = "#FFA500"  # Orange for medium importance
        else:
            edge_color = "#848484"  # Gray for low importance
        
        net.add_edge(
            source,
            target,
            label=edge_label,
            title=edge_title,
            width=width,
            color=edge_color
        )
    
    # Add some statistics to the HTML
    stats_html = f"""
    <div style="background-color: #f0f0f0; padding: 15px; margin: 10px; border-radius: 5px;">
        <h3>Cyber-Physical Interplay Graph - Filtered View</h3>
        <p><strong>Flight Mode:</strong> LAND</p>
        <p><strong>Physical States:</strong> SIM-related states</p>
        <p><strong>Nodes:</strong> {len(filtered_nodes)} ({len(connected_factors)} Factors, {len(filtered_ps)} Physical States)</p>
        <p><strong>Edges:</strong> {len(filtered_edges)}</p>
        <p><strong>Layout:</strong></p>
        <ul>
            <li>Factors: Two columns on the left (levels 1-2)</li>
            <li>Physical States: Single column on the right (level 4)</li>
            <li>Level separation: 600px for clear visibility</li>
        </ul>
        <p><strong>Color Legend:</strong></p>
        <ul>
            <li style="color: #FFF2CC; background-color: #FFF2CC; padding: 2px;"></li> Environmental Factors (Yellow)
            <li style="color: #DAE8FC; background-color: #DAE8FC; padding: 2px;"></li> Physical States (Blue)
        </ul>
        <p><strong>Edge Colors:</strong></p>
        <ul>
            <li style="color: #FF6B6B;"></li> High Importance (>0.1)
            <li style="color: #FFA500;"></li> Medium Importance (0.05-0.1)
            <li style="color: #848484;"></li> Low Importance (<0.05)
        </ul>
        <p><strong>Instructions:</strong> Drag nodes to rearrange. Scroll to zoom. Click and drag background to pan. Hover over nodes for full names.</p>
    </div>
    """
    
    # Generate the HTML
    html_content = net.generate_html()
    
    # Insert statistics at the beginning of the body
    html_content = html_content.replace('<body>', f'<body>\n{stats_html}')
    
    # Save to file
    output_path = Path(cpg_dir) / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n Interactive CPG visualization saved to: {output_path}")
    print(f"Open this file in your browser to view the graph.")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Factors with connections: {len(connected_factors)}")
    print(f"Physical states (LAND + SIM2): {len(filtered_ps)}")
    print(f"Total relationships: {len(filtered_edges)}")
    
    # Find most important factors
    importance_by_factor = filtered_edges.groupby('source')['weight'].agg(['mean', 'max', 'count'])
    importance_by_factor = importance_by_factor.sort_values('mean', ascending=False)
    
    print("\nTop 5 most influential factors (by average importance):")
    for factor, row in importance_by_factor.head(5).iterrows():
        print(f"  {factor}: avg={row['mean']:.3f}, max={row['max']:.3f}, connections={row['count']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cpg_dir = sys.argv[1]
    else:
        cpg_dir = "<RVSPEC_ROOT>/CPG-ArduPilot/DataAnalysis/analysis_output_sitl/cpg_outputs"
    
    create_filtered_cpg_visualization(cpg_dir)