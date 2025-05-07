#!/usr/bin/env python3
"""
This script fixes the model creation in model_train_eval.py.
It ensures that the model is created with the correct input shape.
"""

import os
import sys

# Path to the model_train_eval.py file
MODEL_TRAIN_EVAL_PY_PATH = "/Users/michaelpersson/microWakeWord/notebooks/microWakeWord/microwakeword/model_train_eval.py"

def fix_model_creation():
    """Fix the model creation in model_train_eval.py."""
    if not os.path.exists(MODEL_TRAIN_EVAL_PY_PATH):
        print(f"Error: File not found: {MODEL_TRAIN_EVAL_PY_PATH}")
        return False

    # Read the file
    with open(MODEL_TRAIN_EVAL_PY_PATH, "r") as f:
        content = f.read()

    # Find the model creation code
    old_code = """        model = model_module.model(
            flags, config["training_input_shape"], config["batch_size"]
        )
        logging.info(model.summary())
        logging.info(f"DEBUG: Model input shape: {model.input.shape}")"""
    
    if old_code in content:
        # Add a wrapper around the model to handle different input shapes
        new_code = """        # Create the model with the correct input shape
        model = model_module.model(
            flags, config["training_input_shape"], config["batch_size"]
        )
        logging.info(model.summary())
        logging.info(f"DEBUG: Model input shape: {model.input.shape}")
        
        # Add a wrapper layer to handle different input shapes
        import tensorflow as tf
        
        # Get the original input shape
        input_shape = config["training_input_shape"]
        batch_size = config["batch_size"]
        
        # Create a wrapper model that reshapes the input if needed
        input_layer = tf.keras.layers.Input(shape=(None,), batch_size=batch_size)
        reshape_layer = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (-1, input_shape[0], input_shape[1]))
            if tf.rank(x) < 3 else x
        )(input_layer)
        output_layer = model(reshape_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Use the wrapped model for training
        logging.info("Using wrapped model to handle different input shapes")
        logging.info(f"DEBUG: Wrapped model input shape: {model.input.shape}")"""
        
        # Replace the old code with the new code
        modified_content = content.replace(old_code, new_code)
        
        # Write the modified content back to the file
        with open(MODEL_TRAIN_EVAL_PY_PATH, "w") as f:
            f.write(modified_content)
        
        print(f"Successfully updated {MODEL_TRAIN_EVAL_PY_PATH}")
        print("Added a wrapper around the model to handle different input shapes.")
        return True
    else:
        print("Could not find the model creation code.")
        return False

if __name__ == "__main__":
    fix_model_creation()
