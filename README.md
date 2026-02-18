# RVSPEC: Cyber-Physical Interplay Graphs for Formal Specification of Robotic Vehicle Control Software

RVSPEC is an automatic specification generation framework for robotic vehicle (RV) control software. It constructs **Cyber-Physical Interplay Graphs (CPGs)** that capture how internal factors (e.g., servo lag, sensor fusion) and external factors (e.g., wind, temperature) influence an RV's physical states. RVSPEC then leverages CPGs to guide LLM-based agents in generating **cyber-physical interplay-aware formal specifications** in Metric Temporal Logic (MTL).

## Key Results

- **78.4%** specification accuracy (vs. 51.5% baseline)
- **79.9%** reduction in false positives (4,790 → 964) while preserving bug-finding capability
- Detected all **156/156** previously known logic bugs
- Evaluated on **4 platforms**: ArduPilot, PX4, openpilot, and cFS

## Repository Structure

```
RVSpec/
├── CPG-ArduPilot/          # CPG construction for ArduPilot (aerial vehicles)
│   ├── DataAnalysis/       #   Data preprocessing, CPG construction, model fitting
│   ├── DataAnalysis_BRAKE/ #   Brake mode analysis
│   ├── DataAnalysis_LAND/  #   Landing mode analysis and plotting
│   ├── IdentifyFactors/    #   Internal & external factor identification
│   ├── sim_task.py         #   Simulation task execution
│   ├── sim_schedule.py     #   Simulation scheduling
│   ├── generate_param_files.py
│   ├── baseline.py         #   Baseline comparison (PGFuzz MTL)
│   └── fuzzing.py          #   Fuzzing with RVSPEC MTL formulas
├── CPG-PX4/                # CPG construction for PX4 (aerial vehicles)
├── CPG-openpilot/          # CPG construction for openpilot (autonomous vehicles)
│   ├── false_positives/    #   False positive evaluation
│   ├── parameter_generation/
│   ├── carla_exp.py        #   CARLA simulator experiments
│   └── simulation_orchestrator.py
├── CPG-cFS/                # CPG construction for cFS (spacecraft)
│   ├── DataAnalysis/       #   Telemetry data analysis
│   ├── albedo/             #   Earth albedo effect analysis on CSS
│   ├── parameter_generation/
│   └── sim-cFS.py          #   NOS3 simulation experiments
├── SpecSynthesis/          # LLM-based specification generation
│   ├── specsynthesis.py    #   Main pipeline (ArduPilot/PX4)
│   ├── specsynthesis-openpilot.py
│   ├── specsynthesis-cFS.py
│   ├── extract_logic_policies*.py   # Logic policy extraction agents
│   ├── query_logic_policies*.py     # Logic policy querying agents
│   ├── query_cpg_mtl_formulas*.py   # CPG-enhanced MTL synthesis agents
│   ├── ablation/           #   Ablation study scripts
│   └── evaluation/         #   Evaluation utilities
├── environment.yml         # Conda environment
├── LICENSE
└── README.md
```

## Prerequisites

- **OS**: Ubuntu 22.04 (tested)
- **Python**: 3.10+
- **Simulators** (depending on target platform):
  - [ArduPilot SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html) + [Gazebo](https://gazebosim.org/) for aerial vehicles
  - [CARLA](https://carla.org/) v0.9.13 for autonomous vehicles (openpilot)
  - [NOS3](https://github.com/nasa/nos3) v1.7.3 + [COSMOS](https://openc3.com/) v5.11.3 for spacecraft (cFS)
- **API Keys**: OpenAI and/or Anthropic API access for specification generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prncoprs/RVSpec.git
cd RVSpec
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate RVSpec
```

3. Configure paths and API keys:

RVSPEC uses placeholder paths in the source code marked as `<RVSPEC_ROOT>` and `<DATA_DIR>`. You need to replace these with your actual paths:

```bash
# Replace <RVSPEC_ROOT> with your actual RVSpec directory
find . -name "*.py" -exec sed -i "s|<RVSPEC_ROOT>|/your/path/to/RVSpec|g" {} +

# Replace <DATA_DIR> with your data directory
find . -name "*.py" -exec sed -i "s|<DATA_DIR>|/your/path/to/data|g" {} +

# Replace <HOME_DIR> with your home directory
find . -name "*.py" -exec sed -i "s|<HOME_DIR>|$HOME|g" {} +
```

4. Set API keys (for specification generation):
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Or replace `YOUR_API_KEY_HERE` in the relevant files under `SpecSynthesis/`.

## Usage

RVSPEC consists of three main stages:

### 1. Internal & External Factor Identification

Identify factors that influence an RV's physical states from documentation and source code:

```bash
cd CPG-ArduPilot/IdentifyFactors
python identify_factors.py --ardupilot-path /path/to/ardupilot
```

### 2. CPG Construction

Generate test input vectors, run simulations, and construct the CPG:

```bash
# Generate parameter combinations (Latin Hypercube Sampling)
cd CPG-ArduPilot
python generate_param_files.py

# Run simulation experiments
python sim_schedule.py

# Preprocess data and fit uncertainty-aware models
cd DataAnalysis
python data_preprocessor.py
python construct_cpg.py
```

### 3. Specification Generation

Generate CPG-enhanced MTL formulas using LLM agents:

```bash
cd SpecSynthesis

# Step 1: Extract documentation and generate logic policies
python query_logic_policies1_with_claude.py

# Step 2: Extract logic policies 
python extract_logic_policies1_with_claude.py

# Step 3: Query and generate CPG-enhanced MTL formulas
python query_cpg_mtl_formulas1_with_claude.py
```

For other platforms, use the corresponding scripts.

## Supported Platforms

| Platform | Vehicle Type | Simulator | CPG Directory |
|----------|-------------|-----------|---------------|
| [ArduPilot](https://ardupilot.org/) v4.5.7 | Aerial (drone) | SITL + Gazebo v11.15.1 | `CPG-ArduPilot/` |
| [PX4](https://px4.io/) v1.15.4 | Aerial (drone) | SITL + Gazebo v11.15.1 | `CPG-PX4/` |
| [openpilot](https://comma.ai/openpilot) v0.9.4 | Autonomous vehicle | CARLA v0.9.13 | `CPG-openpilot/` |
| [cFS](https://github.com/nasa/cFS) equuleus-rc1 | Spacecraft | NOS3 v1.7.3 | `CPG-cFS/` |

## Citation

If you use RVSPEC in your research, please cite:

```bibtex
@inproceedings{rvspec2026,
  title     = {{RVSPEC}: Cyber-Physical Interplay Graphs for Formal Specification of Robotic Vehicle Control Software},
  author    = {[Chaoqi Zhang, Minhyun Cho, Inseok Hwang, Hyungsub Kim]},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.