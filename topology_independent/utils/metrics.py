import json
import os

def log_metrics(metrics, config):
    filename = config["log_path"]
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    model_name = config["model_name"]
    log_entry = {"model": model_name}
    log_entry.update(metrics)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(filename, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[Log] Metrics saved to {filename}")

def log_metrics_timellm(metrics, args):
    filename = args.log_path
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    model_name = args.model_name
    log_entry = {"model": model_name}
    log_entry.update(metrics)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(filename, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[Log] Metrics saved to {filename}")
