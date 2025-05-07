# microWakeWord Installation and Usage Guide

This guide provides step-by-step instructions for installing and using microWakeWord on different operating systems. The recommended way to use microWakeWord is through the Jupyter notebook interface, which provides an interactive and guided experience.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
- [Using microWakeWord](#using-microwakeword)
  - [Jupyter Notebook (Recommended)](#jupyter-notebook-recommended)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Troubleshooting](#troubleshooting)

## System Requirements

- **Python**: Version 3.10 (required, not compatible with 3.11+ yet)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: At least 5GB free space
- **GPU**: Optional but recommended for faster training
  - NVIDIA GPU with CUDA support
  - Apple Silicon (M1/M2/M3) for macOS

## Installation

### macOS

#### Prerequisites

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python 3.10:
   ```bash
   brew install python@3.10
   ```

3. Install required system dependencies:
   ```bash
   brew install ffmpeg portaudio
   ```

#### Installing microWakeWord

1. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv microwakeword-env
   source microwakeword-env/bin/activate
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/BigPappy098/microWakeWord.git
   cd microWakeWord
   ```

3. Install dependencies (special handling for macOS):
   ```bash
   pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'
   pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
   pip install -e .
   ```

4. Install Jupyter and ipywidgets (if you plan to use notebooks):
   ```bash
   pip install jupyter ipywidgets
   ```

### Linux

#### Prerequisites

1. Install Python 3.10 and required system dependencies:

   **Ubuntu/Debian**:
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   sudo apt install ffmpeg libportaudio2 libsndfile1 cmake
   ```

   **Fedora**:
   ```bash
   sudo dnf install python3.10 python3.10-devel
   sudo dnf install ffmpeg portaudio-devel libsndfile-devel cmake
   ```

#### Installing microWakeWord

1. Create and activate a virtual environment:
   ```bash
   python3.10 -m venv microwakeword-env
   source microwakeword-env/bin/activate
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/BigPappy098/microWakeWord.git
   cd microWakeWord
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Install Jupyter and ipywidgets (if you plan to use notebooks):
   ```bash
   pip install jupyter ipywidgets
   ```

### Windows

#### Prerequisites

1. Install Python 3.10 from the [official Python website](https://www.python.org/downloads/release/python-31011/)
   - During installation, check "Add Python to PATH"

2. Install Visual Studio Build Tools:
   - Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++" workload

3. Install FFmpeg:
   - Download from [FFmpeg website](https://www.gyan.dev/ffmpeg/builds/)
   - Extract and add the `bin` folder to your PATH

#### Installing microWakeWord

1. Create and activate a virtual environment:
   ```cmd
   python -m venv microwakeword-env
   microwakeword-env\Scripts\activate
   ```

2. Clone the repository:
   ```cmd
   git clone https://github.com/BigPappy098/microWakeWord.git
   cd microWakeWord
   ```

3. Install dependencies:
   ```cmd
   pip install -e .
   ```

4. Install Jupyter and ipywidgets (if you plan to use notebooks):
   ```cmd
   pip install jupyter ipywidgets
   ```

## Using microWakeWord

### Jupyter Notebook (Recommended)

The recommended way to train wake word models is using the interactive Jupyter notebooks, which provide step-by-step guidance and explanations:

1. Start Jupyter:
   ```bash
   # Activate your virtual environment first
   source microwakeword-env/bin/activate  # Linux/macOS
   microwakeword-env\Scripts\activate     # Windows

   jupyter notebook
   ```

2. Navigate to the `notebooks` directory and open one of these notebooks:
   - **`easy_training_notebook.ipynb`** (recommended for most users)
     - Guided interface with clear explanations
     - Simplified parameter selection
     - Automatic handling of common issues
     - Best for beginners and those who want quick results

   - **`basic_training_notebook.ipynb`** (for advanced users)
     - Full control over all training parameters
     - Detailed explanations of each component
     - Allows customization of every aspect of the training process
     - Best for researchers and those who want to fine-tune models

3. Follow the step-by-step instructions in the notebook

The notebooks will guide you through:
- Installing dependencies
- Generating synthetic wake word samples
- Creating spectrograms
- Training the model
- Converting to a format suitable for deployment
- Testing and evaluating your model

### Command-Line Interface

For users who prefer a command-line approach, you can use the `train_wake_word.py` script:

```bash
# Activate your virtual environment first
source microwakeword-env/bin/activate  # Linux/macOS
microwakeword-env\Scripts\activate     # Windows

# Basic usage
./train_wake_word.py "hey_computer"

# With custom parameters
./train_wake_word.py "hey_computer" --preset medium --augmentation heavy --samples 2000 --create-manifest
```

#### Command-Line Options

- `wake_word`: The wake word to train (required)
- `--output-dir`: Directory to save the trained model (default: "trained_models")
- `--preset`: Model preset based on wake word length (choices: "short", "medium", "long", default: "medium")
- `--augmentation`: Level of audio augmentation (choices: "light", "medium", "heavy", default: "medium")
- `--samples`: Number of synthetic samples to generate (default: 1000)
- `--create-manifest`: Create an ESPHome model manifest file
- `--detection-threshold`: Detection threshold for the model manifest (default: 0.7)
- `--negative-weight`: Override the negative class weight (higher = fewer false positives)
- `--training-steps`: Override the number of training steps
- `--batch-size`: Set the batch size for training (default: 128)

### Python API

You can also use the Python API directly in your own scripts:

```python
from microwakeword.easy_train import WakeWordTrainer

# Create a trainer with default settings
trainer = WakeWordTrainer(
    wake_word="hey_computer",
    output_dir="trained_models",
    preset="medium",
    augmentation_level="medium",
    samples_count=1000
)

# Run the full training pipeline
model_path = trainer.run_full_pipeline()
print(f"Model saved to: {model_path}")
```

For advanced users who want more control:

```python
# Advanced configuration
advanced_config = {
    "training_steps": [30000],
    "negative_class_weight": [25],
    "time_mask_max_size": [8],
    "freq_mask_max_size": [8],
    "batch_size": 64  # Smaller batch size if memory is limited
}

trainer = WakeWordTrainer(
    wake_word="hey_computer",
    preset="long",
    augmentation_level="heavy",
    samples_count=3000,
    batch_size=256,  # Larger batch size for faster training
    advanced_config=advanced_config
)

model_path = trainer.run_full_pipeline()
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'microwakeword'

Make sure you've installed the package in development mode:
```bash
pip install -e .
```

#### Missing dependencies

If you encounter errors about missing dependencies, try installing them manually:
```bash
pip install audiomentations audio_metadata datasets mmap_ninja numpy pymicro-features pyyaml tensorflow webrtcvad
```

#### TensorFlow GPU issues

For GPU acceleration:

- **NVIDIA GPUs**: Install CUDA and cuDNN, then:
  ```bash
  pip install tensorflow[and-cuda]
  ```

- **Apple Silicon**: TensorFlow uses the Metal Performance Shaders (MPS) backend automatically

#### Permission denied when running train_wake_word.py

Make the script executable:
```bash
chmod +x train_wake_word.py
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/BigPappy098/microWakeWord/issues) for similar problems
2. Read the [Training Best Practices](training_best_practices.md) guide
3. Open a new issue with details about your system and the error message
