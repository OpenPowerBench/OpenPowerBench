import os
import numpy as np
import pandas as pd
from config.paths import out_path

def save_results(scenario_idx, results_df):
    scenario_dir = os.path.join(out_path, f"scenario_{scenario_idx + 1}")
    os.makedirs(scenario_dir, exist_ok=True)
    results_df.astype(np.float32).to_csv(os.path.join(scenario_dir, "bus_data.csv"), index=False)

def save_fault_info(scenario_idx, fault_type, fault_start, fault_end, fault_location):
    scenario_dir = os.path.join(out_path, f"scenario_{scenario_idx + 1}")
    os.makedirs(scenario_dir, exist_ok=True)
    with open(os.path.join(scenario_dir, "fault_info.csv"), 'w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(["Scenario", "bus1", "bus2", "type", "start", "end"])
        bus1 = fault_location[0] if isinstance(fault_location, list) else fault_location
        bus2 = fault_location[1] if isinstance(fault_location, list) and len(fault_location) > 1 else "-1"
        writer.writerow([scenario_idx + 1, bus1, bus2, fault_type, fault_start, fault_end])
