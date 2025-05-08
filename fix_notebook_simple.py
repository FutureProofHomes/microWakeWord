#!/usr/bin/env python3
import json
import sys

# Read the notebook file
with open('notebooks/easy_training_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the import statement and run_training function
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'from microwakeword.model_train_eval import main as train_main' in line:
                # Replace the entire cell with a simpler approach that uses subprocess
                start_idx = 0
                while start_idx < len(source) and 'import threading' not in source[start_idx]:
                    start_idx += 1
                
                if start_idx < len(source):
                    # Find the end of the cell or at least past the training code
                    end_idx = start_idx
                    while end_idx < len(source) and 'Add a note about training' not in source[end_idx]:
                        end_idx += 1
                    
                    # Create the new code that uses subprocess instead of importing
                    new_code = [
                        '    # Run training using subprocess\n',
                        '    import subprocess\n',
                        '    import threading\n',
                        '    import time\n',
                        '    \n',
                        '    # Create directories if they don\'t exist\n',
                        '    os.makedirs("trained_models/wakeword", exist_ok=True)\n',
                        '    \n',
                        '    def run_training():\n',
                        '        try:\n',
                        '            with output_text:\n',
                        '                print("Starting training...")\n',
                        '                \n',
                        '                # Build the command\n',
                        '                cmd = [\n',
                        '                    "python", "-m", "microwakeword.model_train_eval",\n',
                        '                    f"--training_config=training_parameters.yaml",\n',
                        '                    "--train", "1",\n',
                        '                    "--restore_checkpoint", "1",\n',
                        '                    "--test_tf_nonstreaming", "0",\n',
                        '                    "--test_tflite_nonstreaming", "0",\n',
                        '                    "--test_tflite_nonstreaming_quantized", "0",\n',
                        '                    "--test_tflite_streaming", "0",\n',
                        '                    "--test_tflite_streaming_quantized", "1",\n',
                        '                    "--use_weights", "best_weights",\n',
                        '                    "mixednet",\n',
                        '                    f"--pointwise_filters", pointwise_filters,\n',
                        '                    f"--repeat_in_block", "1,1,1,1",\n',
                        '                    f"--mixconv_kernel_sizes", kernel_sizes,\n',
                        '                    f"--residual_connection", "0,0,0,0",\n',
                        '                    f"--first_conv_filters", str(first_conv_filters),\n',
                        '                    f"--first_conv_kernel_size", "5",\n',
                        '                    f"--stride", "3"\n',
                        '                ]\n',
                        '                \n',
                        '                # Run the command and capture output\n',
                        '                process = subprocess.Popen(\n',
                        '                    cmd,\n',
                        '                    stdout=subprocess.PIPE,\n',
                        '                    stderr=subprocess.STDOUT,\n',
                        '                    text=True,\n',
                        '                    bufsize=1\n',
                        '                )\n',
                        '                \n',
                        '                # Print output in real-time\n',
                        '                for line in process.stdout:\n',
                        '                    print(line.strip())\n',
                        '                \n',
                        '                # Wait for process to complete\n',
                        '                process.wait()\n',
                        '                \n',
                        '                if process.returncode == 0:\n',
                        '                    print("Training completed successfully!")\n',
                        '                    progress_bar.value = 100\n',
                        '                    progress_bar.bar_style = \'success\'\n',
                        '                else:\n',
                        '                    print(f"Training failed with return code {process.returncode}")\n',
                        '                    progress_bar.bar_style = \'danger\'\n',
                        '        except Exception as e:\n',
                        '            with output_text:\n',
                        '                print(f"Error during training: {str(e)}")\n',
                        '                import traceback\n',
                        '                print(traceback.format_exc())\n',
                        '                progress_bar.bar_style = \'danger\'\n',
                        '    \n',
                        '    # Start training in a separate thread\n',
                        '    training_thread = threading.Thread(target=run_training)\n',
                        '    training_thread.start()\n',
                        '    \n',
                        '    # Simulate progress updates (since we can\'t directly hook into the training process)\n',
                        '    total_steps = training_steps_slider.value\n',
                        '    for i in range(1, 101):\n',
                        '        time.sleep(0.1)  # Just to show some initial progress\n',
                        '        if progress_bar.value < i:\n',
                        '            progress_bar.value = i\n',
                        '        if i >= 10:\n',
                        '            break\n'
                    ]
                    
                    # Replace the code
                    source[start_idx:end_idx] = new_code
                
                # Also replace the import statement
                source[i] = '    # Import necessary modules for UI\n'

# Write the modified notebook
with open('notebooks/easy_training_notebook_simple.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully with a simpler approach!")
