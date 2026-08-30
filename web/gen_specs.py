"""Dump specs/models and specs/hardware into web/specs.json for the static page.

Single source of truth stays the yaml files; this just re-serializes them so
the browser doesn't need a yaml parser.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.memory import _load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "specs"
OUT_PATH = Path(__file__).resolve().parent / "specs.json"


def collect(subdir: str) -> list[dict]:
    specs = []
    for path in sorted((SPECS_DIR / subdir).glob("*.yaml")):
        data = _load_yaml(path)
        data["_id"] = path.stem
        specs.append(data)
    return specs


def main():
    payload = {
        "models": collect("models"),
        "hardware": collect("hardware"),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH} ({len(payload['models'])} models, {len(payload['hardware'])} hardware)")


if __name__ == "__main__":
    main()
