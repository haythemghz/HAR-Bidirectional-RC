import scipy.io
import os
import numpy as np

# Path to a sample file
file_path = r'c:\Users\Dell\Desktop\HAR_with_RC\Source Code and Dataset\CZU-MHAD-skeleton_mat\cx_a10_t1.mat'

try:
    mat_data = scipy.io.loadmat(file_path)
    print(f"Keys in .mat file: {mat_data.keys()}")
    
    for key in mat_data:
        if not key.startswith('__'):
            val = mat_data[key]
            print(f"Key: {key}")
            print(f"Type: {type(val)}")
            if isinstance(val, np.ndarray):
                print(f"Shape: {val.shape}")
                print(f"Sample data (first row): {val[0] if len(val) > 0 else 'Empty'}")
            else:
                print(f"Value: {val}")

except Exception as e:
    print(f"Error loading .mat file: {e}")
