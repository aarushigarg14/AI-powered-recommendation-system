#!/bin/bash
# Force Python version
echo "Using Python 3.11"

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
