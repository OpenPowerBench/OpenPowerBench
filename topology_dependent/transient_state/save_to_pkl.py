import os
import pickle
import pandas as pd
import numpy as np
from config.paths import base_path

def save2pkl():
    folder = os.path.join(base_path, "output_fixfaulttime")
    data_list, fault_info = [], []

    for i in range(len(os.listdir(folder))):
        fpath = os.path.join(folder, f"scenario_{i+1}", "bus_data.csv")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        df = pd.read_csv(fpath).drop(columns=["time"])
        data_list.append(df.values.T.astype(np.float32))

        info_path = os.path.join(folder, f"scenario_{i+1}", "fault_info.csv")
        if os.path.exists(info_path):
            info = pd.read_csv(info_path)
            fault_info.append({
                'bus1': info['bus1'][0],
                'bus2': info['bus2'][0],
                'type': info['type'][0]
            })

    data_array = np.stack(data_list, axis=0)

    data_file = os.path.join(base_path, f"dataset{len(data_list)}_2s120hz.pckl")
    fault_file = os.path.join(base_path, f"faultinfo{len(data_list)}_2s120hz.pckl")

    with open(data_file, 'wb') as f: pickle.dump(data_array, f)
    with open(fault_file, 'wb') as f: pickle.dump(fault_info, f)

    print(f"Saved dynamic dataset: {data_file}")
    print(f"Saved fault info: {fault_file}")

if __name__ == "__main__":
    save2pkl()
