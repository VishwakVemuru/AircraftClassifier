#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "  Aircraft Classifier - First-time setup"
echo "============================================"
echo

if ! command -v python3 &> /dev/null; then
    echo "Python 3 was not found on this computer."
    echo "Please install it from https://www.python.org/downloads/ and try again."
    read -p "Press enter to exit..."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating a private Python environment for this app..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! python3 -c "import torch, torchvision, PIL" &> /dev/null; then
    echo "Installing torch, torchvision, pillow (this can take a few minutes)..."
    pip install --quiet torch torchvision pillow
fi

echo
echo "Starting Aircraft Classifier..."
python3 predict_gui.py
