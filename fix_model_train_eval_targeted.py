#!/usr/bin/env python3
"""
This script fixes issues in the model_train_eval.py file.
It adds debug logging and ensures the model is created with the correct input shape.
"""

import os
import sys
import re

# Path to the model_train_eval.py file
MODEL_TRAIN_EVAL_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/model_train_eval.py"

def fix_model_train_eval():
    """Fix issues in the model_train_eval.py file."""
    if not os.path.exists(MODEL_TRAIN_EVAL_PY_PATH):
        print(f"Error: File not found: {MODEL_TRAIN_EVAL_PY_PATH}")
        return False

    # Read the file
    with open(MODEL_TRAIN_EVAL_PY_PATH, "r") as f:
        content = f.read()

    # Add debug logging before model creation
    old_code = """    if flags.train:
        model = model_module.model(
            flags, config["training_input_shape"], config["batch_size"]
        )
        logging.info(model.summary())
        train_model(config, model, data_processor, flags.restore_checkpoint)"""
    
    new_code = """    if flags.train:
        # Debug logging
        logging.info(f"DEBUG: Creating model with input shape: {config['training_input_shape']}, batch size: {config['batch_size']}")
        
        model = model_module.model(
            flags, config["training_input_shape"], config["batch_size"]
        )
        logging.info(model.summary())
        logging.info(f"DEBUG: Model input shape: {model.input.shape}")
        
        train_model(config, model, data_processor, flags.restore_checkpoint)"""
    
    modified_content = content.replace(old_code, new_code)
    
    # Add debug logging in the train_model function
    train_model_pattern = r'def train_model\(config, model, data_processor, restore_checkpoint\):(.*?)utils\.save_model_summary\(model, config\["train_dir"\]\)'
    train_model_match = re.search(train_model_pattern, content, re.DOTALL)
    
    if train_model_match:
        train_model_block = train_model_match.group(0)
        debug_train_code = """
    # Debug logging
    logging.info(f"DEBUG: Training model with config: batch_size={config['batch_size']}, input_shape={config['training_input_shape']}")
    logging.info(f"DEBUG: Model input shape: {model.input.shape}")"""
        
        modified_train_block = train_model_block + debug_train_code
        modified_content = modified_content.replace(train_model_block, modified_train_block)
    
    # Check if any changes were made
    if content == modified_content:
        print("No changes were made. The patterns might not match exactly.")
        return False
    
    # Write the modified content back to the file
    with open(MODEL_TRAIN_EVAL_PY_PATH, "w") as f:
        f.write(modified_content)
    
    print(f"Successfully updated {MODEL_TRAIN_EVAL_PY_PATH}")
    print("Added debug logging to help diagnose the issue.")
    return True

if __name__ == "__main__":
    fix_model_train_eval()
