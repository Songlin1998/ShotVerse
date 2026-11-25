#!/usr/bin/env python3
"""Parse YAML config and output as shell variables."""
import os
import re
import shlex
import sys

import yaml

def flatten(d, prefix=''):
    items = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, key))
        else:
            if isinstance(v, bool):
                v = str(v).lower()
            items[key] = str(v) if v is not None else ""
    return items

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline_config.yaml"
    if not os.path.isfile(config_path):
        sys.exit(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in flatten(config).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            sys.exit(f"Invalid config key: {key}")
        print(f"CFG_{key.upper()}={shlex.quote(value)}")

if __name__ == "__main__":
    main()
