#!/bin/sh
set -eu
if [ -z "${WORKBENCH_PASSWORD:-}" ]; then
  echo "Set WORKBENCH_PASSWORD in the pod environment to start the workbench." >&2
  exit 1
fi
exec python -m uvicorn workbench.server:app --host 0.0.0.0 --port 8000
