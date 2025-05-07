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
Easy training interface for microWakeWord models.
This module provides a simplified interface for training wake word models
with sensible defaults and guided parameter selection.
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("microwakeword.easy_train")


class WakeWordTrainer:
    """A simplified interface for training wake word models."""

    # Predefined model configurations for different wake word types
    MODEL_PRESETS = {
        "short": {  # For short wake words (1-2 syllables)
            "model_type": "mixednet",
            "model_args": {
                "pointwise_filters": "48,48,48,48",
                "repeat_in_block": "1,1,1,1",
                "mixconv_kernel_sizes": "[5],[7,11],[9,15],[17]",
                "residual_connection": "0,0,0,0",
                "first_conv_filters": 32,
                "first_conv_kernel_size": 5,
                "stride": 3,
            },
            "training_steps": 15000,
            "negative_class_weight": 15,
        },
        "medium": {  # For medium wake words (3-4 syllables)
            "model_type": "mixednet",
            "model_args": {
                "pointwise_filters": "64,64,64,64",
                "repeat_in_block": "1,1,1,1",
                "mixconv_kernel_sizes": "[5],[7,11],[9,15],[23]",
                "residual_connection": "0,0,0,0",
                "first_conv_filters": 32,
                "first_conv_kernel_size": 5,
                "stride": 3,
            },
            "training_steps": 20000,
            "negative_class_weight": 20,
        },
        "long": {  # For longer wake words (5+ syllables)
            "model_type": "mixednet",
            "model_args": {
                "pointwise_filters": "64,64,64,64",
                "repeat_in_block": "1,1,1,1",
                "mixconv_kernel_sizes": "[5],[7,11],[9,15],[29]",
                "residual_connection": "0,0,0,0",
                "first_conv_filters": 32,
                "first_conv_kernel_size": 5,
                "stride": 3,
            },
            "training_steps": 25000,
            "negative_class_weight": 25,
        },
    }

    # Predefined augmentation settings for different robustness levels
    AUGMENTATION_PRESETS = {
        "light": {
            "augmentation_probabilities": {
                "SevenBandParametricEQ": 0.1,
                "TanhDistortion": 0.1,
                "PitchShift": 0.1,
                "BandStopFilter": 0.1,
                "AddColorNoise": 0.1,
                "AddBackgroundNoise": 0.5,
                "Gain": 1.0,
                "RIR": 0.3,
            },
            "background_min_snr_db": 0,
            "background_max_snr_db": 15,
        },
        "medium": {
            "augmentation_probabilities": {
                "SevenBandParametricEQ": 0.2,
                "TanhDistortion": 0.1,
                "PitchShift": 0.2,
                "BandStopFilter": 0.1,
                "AddColorNoise": 0.2,
                "AddBackgroundNoise": 0.75,
                "Gain": 1.0,
                "RIR": 0.5,
            },
            "background_min_snr_db": -3,
            "background_max_snr_db": 12,
        },
        "heavy": {
            "augmentation_probabilities": {
                "SevenBandParametricEQ": 0.3,
                "TanhDistortion": 0.2,
                "PitchShift": 0.3,
                "BandStopFilter": 0.2,
                "AddColorNoise": 0.3,
                "AddBackgroundNoise": 0.9,
                "Gain": 1.0,
                "RIR": 0.7,
            },
            "background_min_snr_db": -5,
            "background_max_snr_db": 10,
        },
    }

    def __init__(
        self,
        wake_word: str,
        output_dir: str = "trained_models",
        preset: str = "medium",
        augmentation_level: str = "medium",
        samples_count: int = 1000,
        advanced_config: Optional[Dict] = None,
    ):
        """
        Initialize the wake word trainer with the given parameters.

        Args:
            wake_word: The wake word to train for (e.g., "hey_computer")
            output_dir: Directory to save the trained model
            preset: Model preset to use ("short", "medium", or "long")
            augmentation_level: Level of audio augmentation ("light", "medium", or "heavy")
            samples_count: Number of synthetic samples to generate
            advanced_config: Optional dictionary with advanced configuration parameters
        """
        self.wake_word = wake_word
        self.output_dir = os.path.join(output_dir, wake_word.replace(" ", "_"))
        self.preset = preset
        self.augmentation_level = augmentation_level
        self.samples_count = samples_count
        self.advanced_config = advanced_config or {}

        # Validate inputs
        if preset not in self.MODEL_PRESETS:
            raise ValueError(
                f"Invalid preset: {preset}. Choose from {list(self.MODEL_PRESETS.keys())}"
            )
        if augmentation_level not in self.AUGMENTATION_PRESETS:
            raise ValueError(
                f"Invalid augmentation level: {augmentation_level}. Choose from {list(self.AUGMENTATION_PRESETS.keys())}"
            )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.features_dir = os.path.join(self.output_dir, "features")
        self.config_path = os.path.join(self.output_dir, "training_parameters.yaml")

        # Initialize with default configuration
        self._init_config()

    def _init_config(self):
        """Initialize the training configuration with sensible defaults."""
        self.config = {
            "window_step_ms": 10,
            "train_dir": os.path.join(self.output_dir, "model"),
            "features": [],  # Will be populated during the process
            "training_steps": [self.MODEL_PRESETS[self.preset]["training_steps"]],
            "positive_class_weight": [1],
            "negative_class_weight": [self.MODEL_PRESETS[self.preset]["negative_class_weight"]],
            "learning_rates": [0.001],
            "batch_size": 128,
            "time_mask_max_size": [5],
            "time_mask_count": [2],
            "freq_mask_max_size": [5],
            "freq_mask_count": [2],
            "eval_step_interval": 500,
            "clip_duration_ms": 1500,
            "target_minimization": 0.9,
            "minimization_metric": None,
            "maximization_metric": "average_viable_recall",
            # Explicitly set spectrogram dimensions to ensure compatibility
            "spectrogram_length": 204,
            "feature_count": 40,
        }

        # Override with any advanced configuration
        for key, value in self.advanced_config.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def generate_samples(self):
        """Generate synthetic wake word samples using Piper."""
        logger.info(f"Generating {self.samples_count} samples for wake word: {self.wake_word}")

        # Create samples directory
        os.makedirs(self.samples_dir, exist_ok=True)

        # Check if piper-sample-generator exists
        piper_dir = "piper-sample-generator"
        if not os.path.exists(piper_dir):
            logger.info("Downloading piper-sample-generator...")
            subprocess.run(["git", "clone", "https://github.com/rhasspy/piper-sample-generator"], check=True)

            # Download model if needed
            model_path = os.path.join(piper_dir, "models", "en_US-libritts_r-medium.pt")
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                subprocess.run(
                    ["wget", "-O", model_path, "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"],
                    check=True
                )

        # Generate samples
        cmd = [
            "python3",
            os.path.join(piper_dir, "generate_samples.py"),
            self.wake_word,
            "--max-samples", str(self.samples_count),
            "--batch-size", "100",
            "--output-dir", self.samples_dir
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Verify samples were generated
        sample_count = len([f for f in os.listdir(self.samples_dir) if f.endswith(".wav")])
        logger.info(f"Generated {sample_count} samples in {self.samples_dir}")

        return sample_count

    def prepare_training_config(self):
        """Prepare the training configuration file."""
        logger.info("Preparing training configuration...")

        # Generate spectrograms from samples
        logger.info("Preparing to generate spectrograms from samples...")

        # Ensure the features directory exists
        os.makedirs(self.features_dir, exist_ok=True)

        # Add generated samples as positive features
        self.config["features"].append({
            "features_dir": os.path.join(self.features_dir, "generated_augmented"),
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        })

        # Add negative datasets
        negative_datasets = ["speech", "dinner_party", "no_speech"]
        for dataset in negative_datasets:
            self.config["features"].append({
                "features_dir": f"negative_datasets/{dataset}",
                "sampling_weight": 10.0 if dataset in ["speech", "dinner_party"] else 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            })

        # Add evaluation dataset
        self.config["features"].append({
            "features_dir": "negative_datasets/dinner_party_eval",
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        })

        # Calculate training input shape based on spectrogram dimensions
        # This ensures the model expects the correct input shape
        self.config["training_input_shape"] = [self.config["spectrogram_length"], self.config["feature_count"]]

        # Write configuration to file
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        logger.info(f"Training configuration saved to {self.config_path}")
        return self.config_path

    def train_model(self):
        """Train the wake word model."""
        logger.info("Starting model training...")

        # Get model preset
        model_preset = self.MODEL_PRESETS[self.preset]
        model_type = model_preset["model_type"]
        model_args = model_preset["model_args"]

        # Build command
        cmd = [
            "python", "-m", "microwakeword.model_train_eval",
            f"--training_config={self.config_path}",
            "--train", "1",
            "--restore_checkpoint", "1",
            "--test_tf_nonstreaming", "0",
            "--test_tflite_nonstreaming", "0",
            "--test_tflite_nonstreaming_quantized", "0",
            "--test_tflite_streaming", "0",
            "--test_tflite_streaming_quantized", "1",
            "--use_weights", "best_weights",
            model_type,
        ]

        # Add model-specific arguments
        for arg_name, arg_value in model_args.items():
            cmd.append(f"--{arg_name}")
            cmd.append(str(arg_value))

        # Run training
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        logger.info(f"Model training completed. Model saved in {self.config['train_dir']}")
        return os.path.join(self.config["train_dir"], "streaming_quantized.tflite")

    def generate_spectrograms(self):
        """Generate spectrograms from audio samples."""
        logger.info("Generating spectrograms from audio samples...")

        # Create features directory
        os.makedirs(os.path.join(self.features_dir, "generated_augmented"), exist_ok=True)

        # Check if we have audio samples
        if not os.path.exists(self.samples_dir) or len([f for f in os.listdir(self.samples_dir) if f.endswith(".wav")]) == 0:
            logger.error("No audio samples found. Please generate samples first.")
            return False

        try:
            # Import here to avoid circular imports
            import sys
            import numpy as np
            from microwakeword.audio.clips import Clips
            from microwakeword.audio.augmentation import Augmentation
            from microwakeword.audio.spectrograms import SpectrogramGeneration

            # Configure augmentation based on preset
            augmentation_config = self.AUGMENTATION_PRESETS[self.augmentation_level]

            # Create clips handler
            clips = Clips(
                input_directory=self.samples_dir,
                file_pattern="*.wav",
                max_clip_duration_s=None,
                remove_silence=False,
                random_split_seed=10,
                split_count=0.1,
            )

            # Create augmentation handler
            augmenter = Augmentation(
                augmentation_duration_s=3.2,
                augmentation_probabilities=augmentation_config["augmentation_probabilities"],
                background_min_snr_db=augmentation_config["background_min_snr_db"],
                background_max_snr_db=augmentation_config["background_max_snr_db"],
                impulse_paths=["mit_rirs"] if os.path.exists("mit_rirs") else [],
                background_paths=["fma_16k", "audioset_16k"] if os.path.exists("fma_16k") else [],
                min_jitter_s=0.195,
                max_jitter_s=0.205,
            )

            # Create spectrogram generator
            spectrograms = SpectrogramGeneration(
                clips=clips,
                augmenter=augmenter,
                slide_frames=10,
                step_ms=10,
            )

            # Generate spectrograms
            logger.info("Generating spectrograms... This may take a while.")
            spectrograms.generate_ragged_mmap(
                output_dir=os.path.join(self.features_dir, "generated_augmented", "training"),
                repetition=2,
            )

            logger.info("Spectrograms generated successfully.")
            return True

        except Exception as e:
            logger.error(f"Error generating spectrograms: {str(e)}")
            return False

    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        logger.info(f"Starting full training pipeline for wake word: {self.wake_word}")

        # Step 1: Generate samples
        sample_count = self.generate_samples()
        if sample_count == 0:
            logger.error("Failed to generate samples. Aborting.")
            return None

        # Step 1.5: Generate spectrograms
        if not self.generate_spectrograms():
            logger.warning("Failed to generate spectrograms. Continuing with training, but it may fail.")

        # Step 2: Prepare training configuration
        self.prepare_training_config()

        # Step 3: Train model
        model_path = self.train_model()

        logger.info(f"Training pipeline completed successfully!")
        logger.info(f"Trained model saved to: {model_path}")

        return model_path


def main():
    """Command-line interface for the easy training tool."""
    parser = argparse.ArgumentParser(description="Easy microWakeWord model training")

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

    args = parser.parse_args()

    trainer = WakeWordTrainer(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        preset=args.preset,
        augmentation_level=args.augmentation,
        samples_count=args.samples
    )

    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
