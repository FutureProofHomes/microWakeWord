#!/usr/bin/env python3
"""
This script fixes the batch size issue in the generate_samples method.
It modifies the easy_train.py file to use the batch size from the class instead of hardcoded values.
"""

import os
import sys
import re

# Path to the easy_train.py file
EASY_TRAIN_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/easy_train.py"

def fix_sample_batch_size():
    """Fix the batch size issue in the generate_samples method."""
    if not os.path.exists(EASY_TRAIN_PY_PATH):
        print(f"Error: File not found: {EASY_TRAIN_PY_PATH}")
        return False

    # Read the file
    with open(EASY_TRAIN_PY_PATH, "r") as f:
        content = f.read()

    # Replace the hardcoded batch size in the generate_samples method
    pattern = r'--batch-size", "100",'
    replacement = '--batch-size", str(self.batch_size),'
    
    # Apply the replacement
    modified_content = re.sub(pattern, replacement, content)
    
    # Check if any changes were made
    if content == modified_content:
        print("No changes were needed or pattern not found.")
        return False
    
    # Write the modified content back to the file
    with open(EASY_TRAIN_PY_PATH, "w") as f:
        f.write(modified_content)
    
    print(f"Successfully updated {EASY_TRAIN_PY_PATH}")
    print("The batch size in generate_samples method has been fixed.")
    return True

if __name__ == "__main__":
    fix_sample_batch_size()
