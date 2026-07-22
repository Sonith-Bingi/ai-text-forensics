#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
python demo/app.py
