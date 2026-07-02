#!/bin/sh
set -e
export REDROB_CPU_ONLY=1
export REDROB_IN_CONTAINER=1
export PYTHONPATH=/app

# Default: rank full pool1k (1000) → top-100 CSV in /output
exec python /app/docker/scripts/run_demo.py "$@"
