#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null

if python -m pip install -r requirements.txt; then
  echo "[info] requirements installed in venv"
  python -m py_compile app.py train_model.py tests/test_offline.py
  python tests/test_offline.py
else
  echo "[warn] Could not install runtime dependencies (likely network/proxy limits)."
  echo "[warn] Running offline checks instead."
  python -m py_compile app.py train_model.py tests/test_offline.py
  python tests/test_offline.py
fi
