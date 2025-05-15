from pathlib import Path
from types import SimpleNamespace
import yaml
import argparse

def load_yaml_to_namespace(path: str | Path) -> argparse.Namespace:
    """Read a YAML file and return an argparse‐style namespace."""
    data = yaml.safe_load(Path(path).read_text())
    return argparse.Namespace(**data)