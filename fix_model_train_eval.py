#!/usr/bin/env python3
"""
This script fixes issues in the model_train_eval.py file.
It ensures that the model is created with the correct input shape.
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

    # Add debug logging to help diagnose the issue
    main_pattern = r'if __name__ == "__main__":(.*?)$'
    main_match = re.search(main_pattern, content, re.DOTALL)
    
    if not main_match:
        print("Could not find the main block.")
        return False
    
    main_block = main_match.group(1)
    
    # Add debug logging before model creation
    model_creation_pattern = r'model = model_module\.model\(\s*flags, config\["training_input_shape"\], config\["batch_size"\]\s*\)'
    if model_creation_pattern in content:
        debug_code = """
    # Debug logging
    print(f"DEBUG: Creating model with input shape: {config['training_input_shape']}, batch size: {config['batch_size']}")
    """
        modified_content = content.replace(
            "    if flags.train:",
            "    if flags.train:" + debug_code
        )
        
        # Add more debug logging in the train_model function
        train_model_pattern = r'def train_model\(config, model, data_processor, restore_checkpoint\):(.*?)train\.train\(model, config, data_processor\)'
        train_model_match = re.search(train_model_pattern, content, re.DOTALL)
        
        if train_model_match:
            train_model_block = train_model_match.group(1)
            debug_train_code = """
    # Debug logging
    print(f"DEBUG: Training model with config: batch_size={config['batch_size']}, input_shape={config['training_input_shape']}")
    print(f"DEBUG: Model input shape: {model.input_shape}")
    """
            modified_content = modified_content.replace(
                "    utils.save_model_summary(model, config[\"train_dir\"])",
                "    utils.save_model_summary(model, config[\"train_dir\"])" + debug_train_code
            )
            
            # Write the modified content back to the file
            with open(MODEL_TRAIN_EVAL_PY_PATH, "w") as f:
                f.write(modified_content)
            
            print(f"Successfully updated {MODEL_TRAIN_EVAL_PY_PATH}")
            print("Added debug logging to help diagnose the issue.")
            return True
        else:
            print("Could not find the train_model function.")
            return False
    else:
        print("Could not find the model creation code.")
        return False

if __name__ == "__main__":
    fix_model_train_eval()
