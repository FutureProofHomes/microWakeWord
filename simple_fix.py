#!/usr/bin/env python3

# This script directly modifies the easy_training_notebook.ipynb file to use a simpler approach
# for running the model_train_eval module, similar to the basic_training_notebook.ipynb

import os
import re

# Read the original notebook
with open('notebooks/easy_training_notebook.ipynb', 'r') as f:
    content = f.read()

# Replace the import statement
content = content.replace(
    'from microwakeword.model_train_eval import main as train_main',
    'import subprocess'
)

# Replace the run_training function with a simpler version that uses subprocess
pattern = r'def run_training\(\):\s*try:\s*with output_text:\s*print\("Starting training..."\)\s*train_main\(\)(.*?)progress_bar.bar_style = \'danger\''
replacement = '''def run_training():
        try:
            with output_text:
                print("Starting training...")

                # Build the command
                cmd = [
                    "python", "-m", "microwakeword.model_train_eval",
                    f"--training_config=training_parameters.yaml",
                    "--train", "1",
                    "--restore_checkpoint", "1",
                    "--test_tf_nonstreaming", "0",
                    "--test_tflite_nonstreaming", "0",
                    "--test_tflite_nonstreaming_quantized", "0",
                    "--test_tflite_streaming", "0",
                    "--test_tflite_streaming_quantized", "1",
                    "--use_weights", "best_weights",
                    "mixednet",
                    f"--pointwise_filters", pointwise_filters,
                    f"--repeat_in_block", "1,1,1,1",
                    f"--mixconv_kernel_sizes", kernel_sizes,
                    f"--residual_connection", "0,0,0,0",
                    f"--first_conv_filters", str(first_conv_filters),
                    f"--first_conv_kernel_size", "5",
                    f"--stride", "3"
                ]

                # Run the command and capture output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Print output in real-time
                for line in process.stdout:
                    print(line.strip())

                # Wait for process to complete
                process.wait()

                if process.returncode == 0:
                    print("Training completed successfully!")
                    progress_bar.value = 100
                    progress_bar.bar_style = 'success'
                else:
                    print(f"Training failed with return code {process.returncode}")
                    progress_bar.bar_style = 'danger'
        except Exception as e:
            with output_text:
                print(f"Error during training: {str(e)}")
                import traceback
                print(traceback.format_exc())
                progress_bar.bar_style = \\'danger\\''''

# Use a simpler approach to replace the function
start_marker = 'def run_training():'
end_marker = "progress_bar.bar_style = 'danger'"

start_pos = content.find(start_marker)
if start_pos != -1:
    end_pos = content.find(end_marker, start_pos)
    if end_pos != -1:
        end_pos += len(end_marker)
        content = content[:start_pos] + replacement + content[end_pos:]

# Write the modified content to a new file
with open('notebooks/easy_training_notebook_simple_direct.ipynb', 'w') as f:
    f.write(content)

print("Notebook fixed successfully with a simpler approach!")
