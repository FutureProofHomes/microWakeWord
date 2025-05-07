#!/usr/bin/env python3
"""
This script fixes the data shape issue in the validate_nonstreaming function.
It modifies the train.py file to ensure the data has the correct shape before evaluation.
"""

import os
import sys
import re

# Path to the train.py file
TRAIN_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/train.py"

def fix_data_shape():
    """Fix the data shape issue in the validate_nonstreaming function."""
    if not os.path.exists(TRAIN_PY_PATH):
        print(f"Error: File not found: {TRAIN_PY_PATH}")
        return False

    # Read the file
    with open(TRAIN_PY_PATH, "r") as f:
        content = f.read()

    # Find the validate_nonstreaming function
    validate_pattern = r'def validate_nonstreaming\(config, data_processor, model, test_set\):(.*?)return metrics'
    validate_match = re.search(validate_pattern, content, re.DOTALL)
    
    if not validate_match:
        print("Could not find the validate_nonstreaming function.")
        return False
    
    validate_function = validate_match.group(1)
    
    # Check if we need to add the reshape code
    if "reshape" not in validate_function or "expected_shape" not in validate_function:
        # Add code to reshape the data before evaluation
        old_code = """    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)

    model.reset_metrics()"""
        
        new_code = """    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)
    
    # Ensure data has the correct shape for the model
    expected_shape = tuple(config["training_input_shape"])
    if len(testing_fingerprints.shape) == 2 and testing_fingerprints.shape[1] == expected_shape[0] * expected_shape[1]:
        # Data is flattened, reshape it to the expected 3D shape
        testing_fingerprints = testing_fingerprints.reshape(-1, expected_shape[0], expected_shape[1])
    
    model.reset_metrics()"""
        
        modified_content = content.replace(old_code, new_code)
        
        # Also fix the ambient evaluation part
        old_ambient_code = """        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)"""
        
        new_ambient_code = """        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)
        
        # Ensure ambient data has the correct shape for the model
        if len(ambient_testing_fingerprints.shape) == 2 and ambient_testing_fingerprints.shape[1] == expected_shape[0] * expected_shape[1]:
            # Data is flattened, reshape it to the expected 3D shape
            ambient_testing_fingerprints = ambient_testing_fingerprints.reshape(-1, expected_shape[0], expected_shape[1])"""
        
        modified_content = modified_content.replace(old_ambient_code, new_ambient_code)
        
        # Check if any changes were made
        if content == modified_content:
            print("No changes were made. The patterns might not match exactly.")
            return False
        
        # Write the modified content back to the file
        with open(TRAIN_PY_PATH, "w") as f:
            f.write(modified_content)
        
        print(f"Successfully updated {TRAIN_PY_PATH}")
        print("The data shape in validate_nonstreaming function has been fixed.")
        return True
    else:
        print("The reshape code already exists in the validate_nonstreaming function.")
        return False

if __name__ == "__main__":
    fix_data_shape()
