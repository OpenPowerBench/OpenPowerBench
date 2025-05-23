import random
import numpy as np
from utils.aux_writer import write_aux
from config.paths import ctg_filepath
from esa import SAW
from config.paths import case_T

saw = SAW(case_T)

def rand_R():
    ranges = [[0.002,0.01],[0.01, 0.1],[0.1,1],[1,5]]
    R = round(np.random.uniform(*random.choice(ranges)), 4)
    X = R * np.random.uniform(0.2, 3)
    return R, X

def set_random_fault():
    fault_type = random.choice(["generator_trip", "line_trip", "line_fault", "bus_trip", "bus_fault"])
    if fault_type in ["line_trip", "line_fault"]:
        data = saw.GetParametersMultipleElement('Branch', ['BusNum', 'BusNum:1', 'LineCircuit'], '')
    elif fault_type == "generator_trip":
        data = saw.GetParametersMultipleElement("Gen", ['BusNum', 'GenID'], '')
    else:
        data = saw.GetParametersMultipleElement('Bus', ['BusNum'], '')
    entry = data.sample(1).values.tolist()[0]
    R, X = rand_R() if 'fault' in fault_type else (0, 0)
    return fault_type, entry, R, X

def apply_fault(fault_type, location, start, end, time, step, R, X, idx):
    name = f"My Transient Contingency{idx}"
    if fault_type == "generator_trip":
        obj = f"Gen '{location[0]}' '{location[1]}'"
        a0, a1 = "OPEN", "CLOSE Angle_Deg 0 Exciter_Setpoint_pu 0 Governor_Setpoint_MW 0"
    elif fault_type == "line_trip":
        obj = f"Branch '{location[0]}' '{location[1]}' '{location[2]}'"
        a0, a1 = "OPEN BOTH", "CLOSE BOTH"
    elif fault_type == "line_fault":
        obj = f"Branch '{location[0]}' '{location[1]}' '{location[2]}'"
        a0, a1 = f"FAULT 50 3PB IMP {R} {X}", "CLEARFAULT"
    elif fault_type == "bus_trip":
        obj = f"Bus '{location}' '1'"
        a0, a1 = "OPEN", "CLEARFAULT"
    else:
        obj = f"Bus '{location}' '1'"
        a0, a1 = f"FAULT 3PB IMP {R} {X}", "CLEARFAULT"

    data = {
        0: f'"{name}" {start} "{obj}" "{a0}" "CHECK" "NO" 3.0 0.0',
        1: f'"{name}" {end} "{obj}" "{a1}" "CHECK" "NO" 3.0 0.0'
    }
    write_aux(data, time, step, idx)
