
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_and_prepare_data(ts_file, label_file, label_target='type'):
    with open(ts_file, 'rb') as f:
        ts_data = pickle.load(f)

    n_bus = ts_data.shape[1]

    with open(label_file, 'rb') as f:
        label_dicts_raw = pickle.load(f)

    label_dicts = []
    for entry in label_dicts_raw:
        clean_entry = {k: v.iloc[0] if hasattr(v, 'iloc') else v for k, v in entry.items()}
        label_dicts.append(clean_entry)

    ts_data = np.transpose(ts_data, (0, 2, 1))

    type_set = sorted(list(set(d['type'] for d in label_dicts)))
    type_map = {k: i for i, k in enumerate(type_set)}

    buses = list(range(1, n_bus+1))
    ground_pairs = [f"{bus}_-1" for bus in buses]

    location_pairs = set()
    for d in label_dicts:
        location_pairs.add(f"{d['bus1']}_{d['bus2']}")

    full_location_set = sorted(location_pairs.union(ground_pairs))
    location_map = {name: idx for idx, name in enumerate(full_location_set)}

    if label_target == 'type':
        labels = np.array([type_map[d['type']] for d in label_dicts])
    elif label_target == 'location':
        labels = []
        for d in label_dicts:
            key1 = f"{d['bus1']}_{d['bus2']}"
            key2 = f"{d['bus2']}_{d['bus1']}"
            labels.append(location_map.get(key1, location_map.get(key2)))
    else:
        raise ValueError("Unknown label target")

    return ts_data, labels, len(type_map) if label_target == 'type' else len(location_map), ts_data.shape[2]
