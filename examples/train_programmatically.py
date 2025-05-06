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
Example script demonstrating how to use the microWakeWord Python API programmatically.
This shows how to train multiple wake word models with different configurations.
"""

import os
import json
import logging
from microwakeword.easy_train import WakeWordTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_example")

def create_manifest(wake_word, model_dir, detection_threshold=0.7):
    """Create an ESPHome model manifest file."""
    manifest = {
        "name": wake_word,
        "version": 2,
        "type": "micro_speech",
        "description": f"Custom wake word model for '{wake_word}'",
        "specs": {
            "average_window_length": 10,
            "detection_threshold": detection_threshold,
            "suppression_ms": 1000,
            "minimum_count": 3,
            "sample_rate": 16000,
            "vocabulary": ["_silence_", "_unknown_", wake_word]
        }
    }
    
    manifest_file = os.path.join(model_dir, "manifest.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Model manifest created at: {manifest_file}")
    return manifest_file

def train_wake_word(
    wake_word,
    output_dir="trained_models",
    preset="medium",
    augmentation_level="medium",
    samples_count=1000,
    advanced_config=None,
    create_manifest_file=True,
    detection_threshold=0.7
):
    """Train a wake word model with the specified parameters."""
    logger.info(f"Training wake word model: {wake_word}")
    
    # Create trainer
    trainer = WakeWordTrainer(
        wake_word=wake_word,
        output_dir=output_dir,
        preset=preset,
        augmentation_level=augmentation_level,
        samples_count=samples_count,
        advanced_config=advanced_config or {}
    )
    
    # Run training
    model_path = trainer.run_full_pipeline()
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
        # Create manifest if requested
        if create_manifest_file:
            manifest_file = create_manifest(
                wake_word=wake_word,
                model_dir=os.path.dirname(model_path),
                detection_threshold=detection_threshold
            )
            
        return {
            "wake_word": wake_word,
            "model_path": model_path,
            "manifest_path": manifest_file if create_manifest_file else None,
            "success": True
        }
    else:
        logger.error(f"Training failed for wake word: {wake_word}")
        return {
            "wake_word": wake_word,
            "success": False
        }

def main():
    """Train multiple wake word models with different configurations."""
    # Define wake words to train
    wake_words = [
        {
            "name": "hey_computer",
            "preset": "medium",
            "augmentation": "medium",
            "samples": 1000,
            "advanced_config": {
                "negative_class_weight": [20]
            }
        },
        {
            "name": "jarvis",
            "preset": "short",
            "augmentation": "heavy",
            "samples": 1500,
            "advanced_config": {
                "negative_class_weight": [25],
                "training_steps": [15000]
            }
        }
    ]
    
    # Train each wake word
    results = []
    for wake_word_config in wake_words:
        result = train_wake_word(
            wake_word=wake_word_config["name"],
            preset=wake_word_config["preset"],
            augmentation_level=wake_word_config["augmentation"],
            samples_count=wake_word_config["samples"],
            advanced_config=wake_word_config.get("advanced_config")
        )
        results.append(result)
    
    # Print summary
    logger.info("\n=== Training Summary ===")
    for result in results:
        if result["success"]:
            logger.info(f"✅ {result['wake_word']}: Model saved to {result['model_path']}")
        else:
            logger.info(f"❌ {result['wake_word']}: Training failed")
    
    # Generate ESPHome configuration example
    if any(result["success"] for result in results):
        logger.info("\n=== ESPHome Configuration Example ===")
        logger.info("Add the following to your ESPHome configuration:")
        
        esphome_config = """
# Wake word configuration
micro_wake_word:"""
        
        for result in results:
            if result["success"]:
                wake_word = result["wake_word"]
                model_filename = os.path.basename(result["model_path"])
                esphome_config += f"""
  - model_file: "{model_filename}"
    model_name: "{wake_word}"
    probability_cutoff: 0.7"""
        
        esphome_config += """

binary_sensor:"""
        
        for result in results:
            if result["success"]:
                wake_word = result["wake_word"]
                esphome_config += f"""
  - platform: micro_wake_word
    name: "{wake_word.replace('_', ' ').title()} Detected"
    id: {wake_word}_detected
    model_id: {wake_word}"""
        
        esphome_config += """

# Example actions
on_wake_word:
  - if:
      condition:
        binary_sensor.is_on: hey_computer_detected
      then:
        - logger.log: "Hey Computer detected!"
        # Add your actions here
"""
        
        logger.info(esphome_config)

if __name__ == "__main__":
    main()
