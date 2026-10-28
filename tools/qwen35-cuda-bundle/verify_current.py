#!/usr/bin/env python3
"""Authenticate current Qwen CUDA bytes/contracts and report staging lineage."""
import argparse
import json
from pathlib import Path
import stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(stage.verify_current(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
