# Transient Simulation with PowerWorld via ESA

This repository provides a modular framework for performing transient stability simulations using PowerWorld and ESA (SimAuto).

## Features
- Random fault generation and application
- Power flow setup
- Transient simulation result retrieval
- Scenario-specific data and metadata export

## Structure
- `main.py`: Entry point for running the simulations.
- `utils/`: Core modules for network interaction, fault handling, result processing.
- `config/`: Centralized configuration for file paths.
- `scripts/`: Data post-processing tools (e.g., conversion to `.pkl`).

## Requirements
- PowerWorld Simulator with SimAuto
- Python 3.8+
- Required packages (see `requirements.txt`)

## Setup
```bash
pip install -r requirements.txt
```

## Run Simulation
```bash
python main.py
```

## Convert Results to PKL
```bash
python scripts/convert_to_pkl.py
```
