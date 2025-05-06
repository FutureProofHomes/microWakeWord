#!/usr/bin/env python3
# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Command-line interface for training microWakeWord models.
This script provides a simple way to train wake word models from the terminal.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_wake_word")

def main():
    """Command-line interface for training wake word models."""
    parser = argparse.ArgumentParser(
        description="Train a custom wake word model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "wake_word", 
        type=str, 
        help="The wake word to train (e.g., 'hey_computer')"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="trained_models",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["short", "medium", "long"],
        default="medium",
        help="Model preset based on wake word length"
    )
    
    parser.add_argument(
        "--augmentation", 
        type=str, 
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Level of audio augmentation to apply"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of synthetic samples to generate"
    )
    
    parser.add_argument(
        "--create-manifest", 
        action="store_true",
        help="Create an ESPHome model manifest file"
    )
    
    parser.add_argument(
        "--detection-threshold", 
        type=float, 
        default=0.7,
        help="Detection threshold for the model manifest (0.0-1.0)"
    )
    
    parser.add_argument(
        "--negative-weight", 
        type=float, 
        default=None,
        help="Override the negative class weight (higher = fewer false positives)"
    )
    
    parser.add_argument(
        "--training-steps", 
        type=int, 
        default=None,
        help="Override the number of training steps"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow imports when just showing help
    try:
        from microwakeword.easy_train import WakeWordTrainer
    except ImportError:
        logger.error("Could not import microwakeword. Please install it with: pip install -e .")
        sys.exit(1)
    
    # Create advanced config if needed
    advanced_config = {}
    if args.negative_weight is not None:
        advanced_config["negative_class_weight"] = [args.negative_weight]
    
    if args.training_steps is not None:
        advanced_config["training_steps"] = [args.training_steps]
    
    # Create trainer
    trainer = WakeWordTrainer(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        preset=args.preset,
        augmentation_level=args.augmentation,
        samples_count=args.samples,
        advanced_config=advanced_config
    )
    
    # Run training
    model_path = trainer.run_full_pipeline()
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
        # Create manifest if requested
        if args.create_manifest:
            manifest = {
                "name": args.wake_word,
                "version": 2,
                "type": "micro_speech",
                "description": f"Custom wake word model for '{args.wake_word}'",
                "specs": {
                    "average_window_length": 10,
                    "detection_threshold": args.detection_threshold,
                    "suppression_ms": 1000,
                    "minimum_count": 3,
                    "sample_rate": 16000,
                    "vocabulary": ["_silence_", "_unknown_", args.wake_word]
                }
            }
            
            manifest_file = os.path.join(os.path.dirname(model_path), "manifest.json")
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Model manifest created at: {manifest_file}")
            
        logger.info("\nTo use this model with ESPHome:")
        logger.info("1. Copy the .tflite file to your ESPHome configuration directory")
        logger.info("2. Create a manifest.json file (or use the one created with --create-manifest)")
        logger.info("3. Add the following to your ESPHome configuration:")
        logger.info(f"""
    # Wake word configuration
    micro_wake_word:
      model_file: "{os.path.basename(model_path)}"
      model_name: "{args.wake_word}"
      probability_cutoff: {args.detection_threshold}
      
    binary_sensor:
      - platform: micro_wake_word
        name: "Wake Word Detected"
        id: wake_word
        model_id: {args.wake_word}
        
    # Optional - add a text-to-speech response
    esphome:
      on_boot:
        priority: -100
        then:
          - delay: 5s
          - logger.log: "Wake word detection ready"
          
    on_wake_word:
      - logger.log: "Wake word detected!"
      # Add your actions here
        """)
    else:
        logger.error("Training failed or model not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
