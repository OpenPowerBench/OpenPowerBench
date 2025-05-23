import csv

def load_profiles(data_path, row_indices):
    profiles, profile_bus = [], []
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for idx, row in enumerate(reader):
            if idx in row_indices:
                profiles.append([float(val) for val in row])
        profile_bus = [int(col.split(".")[1]) for col in header]
    return profiles, profile_bus
