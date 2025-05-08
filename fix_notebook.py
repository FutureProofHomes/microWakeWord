#!/usr/bin/env python3
import json
import sys

# Read the notebook file
with open('notebooks/easy_training_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the import statement
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'from microwakeword.model_train_eval import main as train_main' in line:
                # Replace the import statement
                source[i] = line.replace(
                    'from microwakeword.model_train_eval import main as train_main',
                    'import microwakeword.model_train_eval as model_train_eval\n    import microwakeword.inception as inception\n    import microwakeword.mixednet as mixednet\n    import microwakeword.data as input_data\n    from absl import logging'
                )
            
            # Replace the train_main() call with direct function calls
            if 'train_main()' in line:
                # Find the run_training function
                start_idx = i
                while start_idx > 0 and 'def run_training()' not in source[start_idx]:
                    start_idx -= 1
                
                # Find the end of the function
                end_idx = i
                while end_idx < len(source) and 'progress_bar.bar_style = \'danger\'' not in source[end_idx]:
                    end_idx += 1
                
                # Replace the function body
                new_function = [
                    '    def run_training():\n',
                    '        try:\n',
                    '            with output_text:\n',
                    '                print("Starting training...")\n',
                    '                \n',
                    '                # Parse the arguments\n',
                    '                parser = model_train_eval.argparse.ArgumentParser()\n',
                    '                parser.add_argument("--training_config", type=str, default="training_parameters.yaml")\n',
                    '                parser.add_argument("--train", type=int, default=1)\n',
                    '                parser.add_argument("--restore_checkpoint", type=int, default=1)\n',
                    '                parser.add_argument("--test_tf_nonstreaming", type=int, default=0)\n',
                    '                parser.add_argument("--test_tflite_nonstreaming", type=int, default=0)\n',
                    '                parser.add_argument("--test_tflite_nonstreaming_quantized", type=int, default=0)\n',
                    '                parser.add_argument("--test_tflite_streaming", type=int, default=0)\n',
                    '                parser.add_argument("--test_tflite_streaming_quantized", type=int, default=1)\n',
                    '                parser.add_argument("--use_weights", type=str, default="best_weights")\n',
                    '                \n',
                    '                # Add model-specific arguments\n',
                    '                subparsers = parser.add_subparsers(dest="model_name")\n',
                    '                parser_mixednet = subparsers.add_parser("mixednet")\n',
                    '                mixednet.model_parameters(parser_mixednet)\n',
                    '                \n',
                    '                # Parse the arguments\n',
                    '                flags = parser.parse_args([])\n',
                    '                flags.model_name = "mixednet"\n',
                    '                flags.pointwise_filters = pointwise_filters\n',
                    '                flags.repeat_in_block = "1,1,1,1"\n',
                    '                flags.mixconv_kernel_sizes = kernel_sizes.strip("\'")\n',
                    '                flags.residual_connection = "0,0,0,0"\n',
                    '                flags.first_conv_filters = first_conv_filters\n',
                    '                flags.first_conv_kernel_size = 5\n',
                    '                flags.stride = 3\n',
                    '                \n',
                    '                # Set up logging\n',
                    '                logging.set_verbosity(logging.INFO)\n',
                    '                \n',
                    '                # Load the configuration\n',
                    '                model_module = mixednet\n',
                    '                config = model_train_eval.load_config(flags, model_module)\n',
                    '                \n',
                    '                # Create the data processor\n',
                    '                data_processor = input_data.FeatureHandler(config)\n',
                    '                \n',
                    '                # Create and train the model\n',
                    '                model = model_module.model(flags, config["training_input_shape"], config["batch_size"])\n',
                    '                print(model.summary())\n',
                    '                model_train_eval.train_model(config, model, data_processor, flags.restore_checkpoint)\n',
                    '                \n',
                    '                # Evaluate the model if requested\n',
                    '                if (flags.test_tf_nonstreaming or flags.test_tflite_nonstreaming or \n',
                    '                    flags.test_tflite_streaming or flags.test_tflite_streaming_quantized):\n',
                    '                    model = model_module.model(flags, shape=config["training_input_shape"], batch_size=1)\n',
                    '                    model.load_weights(os.path.join(config["train_dir"], flags.use_weights) + ".weights.h5")\n',
                    '                    print(model.summary())\n',
                    '                    model_train_eval.evaluate_model(\n',
                    '                        config,\n',
                    '                        model,\n',
                    '                        data_processor,\n',
                    '                        flags.test_tf_nonstreaming,\n',
                    '                        flags.test_tflite_nonstreaming,\n',
                    '                        flags.test_tflite_nonstreaming_quantized,\n',
                    '                        flags.test_tflite_streaming,\n',
                    '                        flags.test_tflite_streaming_quantized\n',
                    '                    )\n',
                    '                \n',
                    '                print("Training completed!")\n',
                    '                progress_bar.value = 100\n',
                    '                progress_bar.bar_style = \'success\'\n',
                    '        except Exception as e:\n',
                    '            with output_text:\n',
                    '                print(f"Error during training: {str(e)}")\n',
                    '                import traceback\n',
                    '                print(traceback.format_exc())\n',
                    '                progress_bar.bar_style = \'danger\'\n'
                ]
                
                # Replace the function
                source[start_idx:end_idx+1] = new_function

# Write the modified notebook
with open('notebooks/easy_training_notebook_fixed.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!")
