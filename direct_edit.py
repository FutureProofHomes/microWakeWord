#!/usr/bin/env python3

# This script directly edits the notebook file to replace the problematic code

with open('notebooks/easy_training_notebook.ipynb', 'r') as f:
    lines = f.readlines()

# Find the line with the import statement
for i, line in enumerate(lines):
    if 'from microwakeword.model_train_eval import main as train_main' in line:
        lines[i] = line.replace('from microwakeword.model_train_eval import main as train_main', 'import subprocess')
    
    # Replace the train_main() call with subprocess
    if 'train_main()' in line:
        lines[i] = line.replace('train_main()', '''# Build the command
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
                else:
                    print(f"Training failed with return code {process.returncode}")''')

# Write the modified content back to a new file
with open('notebooks/easy_training_notebook_direct.ipynb', 'w') as f:
    f.writelines(lines)

print("Notebook fixed with direct editing approach!")
