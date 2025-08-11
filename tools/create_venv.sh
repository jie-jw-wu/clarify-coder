#!/bin/bash

# Set this to your Python (leave as "python3" to use system default)
PYTHON_EXE="python"

echo "Creating virtual environment..."
"$PYTHON_EXE" -m venv ../venv

# Activate the virtual environment
source ../venv/bin/activate

echo "Installing requirements..."
pip install -r ../requirements.txt

echo "[DONE] Setup complete."
