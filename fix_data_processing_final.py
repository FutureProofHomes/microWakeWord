#!/usr/bin/env python3
"""
This script fixes issues in the data.py file.
It ensures that the data is returned in the correct shape.
"""

import os
import sys

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

    # Find the return statements in the get_data method
    changes_made = False
    
    # First return statement
    old_return1 = "            return data, np.array(labels), np.array(weights)"
    new_return1 = """            # Ensure data has the correct 3D shape if needed
            if mode != "training_ambient" and mode != "validation_ambient" and mode != "testing_ambient":
                # Check if data needs reshaping
                if isinstance(data[0], np.ndarray) and len(data[0].shape) == 2:
                    # Convert to numpy array
                    data = np.array(data)
                    if len(data.shape) == 3 and data.shape[2] == features_length * 40:
                        # Data is flattened, reshape it to the expected 3D shape
                        data = data.reshape(-1, features_length, 40)
                        print(f"DEBUG: Reshaped ambient data from 2D to 3D. New shape: {data.shape}")
            
            return data, np.array(labels), np.array(weights)"""
    
    if old_return1 in content:
        content = content.replace(old_return1, new_return1)
        changes_made = True
    
    # Second return statement
    old_return2 = "        return data[indices], labels[indices], weights[indices]"
    new_return2 = """        # Ensure data has the correct 3D shape if needed
        if mode != "training_ambient" and mode != "validation_ambient" and mode != "testing_ambient":
            # Convert to numpy array if it's a list
            if isinstance(data, list):
                data = np.array(data)
            
            # Check if data needs reshaping
            if len(data.shape) == 2 and data.shape[1] == features_length * 40:
                # Data is flattened, reshape it to the expected 3D shape
                data = data.reshape(-1, features_length, 40)
                print(f"DEBUG: Reshaped data from 2D to 3D. New shape: {data.shape}")
        
        return data[indices], labels[indices], weights[indices]"""
    
    if old_return2 in content:
        content = content.replace(old_return2, new_return2)
        changes_made = True
    
    # Check if any changes were made
    if not changes_made:
        print("No changes were made. The patterns might not match exactly.")
        return False
    
    # Add import for numpy if not already present
    if "import numpy as np" not in content:
        content = "import numpy as np\n" + content
    
    # Write the modified content back to the file
    with open(DATA_PY_PATH, "w") as f:
        f.write(content)
    
    print(f"Successfully updated {DATA_PY_PATH}")
    print("Added code to ensure data is returned in the correct shape.")
    return True

if __name__ == "__main__":
    fix_data_processing()
