#!/usr/bin/env python3
"""
This script fixes issues in the data.py file.
It ensures that the data is returned in the correct shape.
"""

import os
import sys
import re

# Path to the data.py file
DATA_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/data.py"

def fix_data_processing():
    """Fix issues in the data.py file."""
    if not os.path.exists(DATA_PY_PATH):
        print(f"Error: File not found: {DATA_PY_PATH}")
        return False

    # Read the file
    with open(DATA_PY_PATH, "r") as f:
        content = f.read()

    # Find the get_data method in the FeatureHandler class
    get_data_pattern = r'def get_data\(\s*self,\s*mode:.*?return data, labels, weights'
    get_data_match = re.search(get_data_pattern, content, re.DOTALL)
    
    if not get_data_match:
        print("Could not find the get_data method.")
        return False
    
    get_data_block = get_data_match.group(0)
    
    # Check if we need to add the reshape code
    if "reshape" not in get_data_block or "features_length" not in get_data_block:
        # Add code to reshape the data before returning it
        old_return = "        return data, labels, weights"
        
        new_return = """        # Ensure data has the correct 3D shape if needed
        if mode != "training_ambient" and mode != "validation_ambient" and mode != "testing_ambient":
            # Convert to numpy array if it's a list
            if isinstance(data, list):
                import numpy as np
                data = np.array(data)
            
            # Check if data needs reshaping
            if len(data.shape) == 2 and data.shape[1] == features_length * 40:
                # Data is flattened, reshape it to the expected 3D shape
                data = data.reshape(-1, features_length, 40)
                print(f"DEBUG: Reshaped data from 2D to 3D. New shape: {data.shape}")
        
        return data, labels, weights"""
        
        modified_content = content.replace(old_return, new_return)
        
        # Check if any changes were made
        if content == modified_content:
            print("No changes were made. The patterns might not match exactly.")
            return False
        
        # Write the modified content back to the file
        with open(DATA_PY_PATH, "w") as f:
            f.write(modified_content)
        
        print(f"Successfully updated {DATA_PY_PATH}")
        print("Added code to ensure data is returned in the correct shape.")
        return True
    else:
        print("The reshape code already exists in the get_data method.")
        return False

if __name__ == "__main__":
    fix_data_processing()
