#!/bin/bash

echo "Fixing numpy/scikit-learn compatibility issue..."

# Remove existing environment
rm -rf venv

# Create new environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install with compatible versions
echo "Installing compatible package versions..."
pip install --upgrade pip
pip install numpy==1.23.5
pip install scipy==1.10.1
pip install scikit-learn==1.2.2
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install jupyter==1.0.0

echo "Testing installation..."
python -c "
import numpy as np
import pandas as pd
import sklearn
print('✓ NumPy version:', np.__version__)
print('✓ Pandas version:', pd.__version__)
print('✓ Scikit-learn version:', sklearn.__version__)
print('All packages installed successfully!')
"

echo "Environment fixed! Run: source venv/bin/activate"