#!/usr/bin/env python3
"""
This script fixes the batch size issue in the validate_nonstreaming function.
It modifies the train.py file to use the batch size from the config instead of hardcoded values.
"""

import os
import sys
import re

# Path to the train.py file
TRAIN_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/train.py"

def fix_batch_size():
    """Fix the batch size issue in the validate_nonstreaming function."""
    if not os.path.exists(TRAIN_PY_PATH):
        print(f"Error: File not found: {TRAIN_PY_PATH}")
        return False

    # Read the file
    with open(TRAIN_PY_PATH, "r") as f:
        content = f.read()

    # Replace the hardcoded batch size in the validate_nonstreaming function
    pattern1 = r"result = model\.evaluate\(\s*testing_fingerprints,\s*testing_ground_truth,\s*batch_size=1024,"
    replacement1 = "result = model.evaluate(\n        testing_fingerprints,\n        testing_ground_truth,\n        batch_size=config[\"batch_size\"],"
    
    # Replace the hardcoded batch size in the ambient evaluation
    pattern2 = r"ambient_predictions = model\.evaluate\(\s*ambient_testing_fingerprints,\s*ambient_testing_ground_truth,\s*batch_size=1024,"
    replacement2 = "ambient_predictions = model.evaluate(\n                ambient_testing_fingerprints,\n                ambient_testing_ground_truth,\n                batch_size=config[\"batch_size\"],"
    
    # Apply the replacements
    modified_content = re.sub(pattern1, replacement1, content)
    modified_content = re.sub(pattern2, replacement2, modified_content)
    
    # Check if any changes were made
    if content == modified_content:
        print("No changes were needed or patterns not found.")
        return False
    
    # Write the modified content back to the file
    with open(TRAIN_PY_PATH, "w") as f:
        f.write(modified_content)
    
    print(f"Successfully updated {TRAIN_PY_PATH}")
    print("The batch size in validate_nonstreaming function has been fixed.")
    return True

if __name__ == "__main__":
    fix_batch_size()
