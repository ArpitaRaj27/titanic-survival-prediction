# test_fix.py
import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"NumPy: {np.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

import sklearn
print(f"Scikit-learn: {sklearn.__version__}")

# Test the problematic import
from numpy.core.numeric import ComplexWarning
print("✓ ComplexWarning import successful!")

# Test scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
print("✓ All scikit-learn imports successful!")

print("\n✅ All tests passed! Your environment is ready.")
