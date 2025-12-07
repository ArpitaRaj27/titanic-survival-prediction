# simple_test.py - Test if packages are working
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import numpy as np
    print("✓ NumPy installed successfully")
    print(f"  NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas installed successfully")
    print(f"  Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import sklearn
    print("✓ Scikit-learn installed successfully")
    print(f"  Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")