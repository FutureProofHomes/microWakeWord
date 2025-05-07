# microWakeWord Quick Start Guide

This guide will help you quickly get started with training your own custom wake word model using microWakeWord. Follow these simple steps to create a model that can be deployed on microcontrollers.

## Prerequisites

- Python 3.10 installed
- Git installed
- 5GB+ of free disk space
- Internet connection for downloading dependencies

## 5-Minute Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/BigPappy098/microWakeWord.git
cd microWakeWord
```

### Step 2: Set Up Environment

**macOS**:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'
pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
pip install -e .
```

**Linux**:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

**Windows**:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -e .
```

### Step 3: Train Your First Model

Use the command-line interface to train a model with default settings:

```bash
# Make the script executable (Linux/macOS only)
chmod +x train_wake_word.py

# Train a model for "hey computer"
./train_wake_word.py "hey_computer" --create-manifest
```

That's it! The script will:
1. Generate synthetic wake word samples
2. Download negative samples
3. Train a model with sensible defaults
4. Create a model manifest for ESPHome

Your trained model will be saved in `trained_models/hey_computer/model/`.

## Next Steps

### Customize Your Training

Adjust parameters to improve your model:

```bash
./train_wake_word.py "hey_computer" \
  --preset medium \
  --augmentation heavy \
  --samples 2000 \
  --detection-threshold 0.8 \
  --negative-weight 25 \
  --create-manifest
```

### Use the Interactive Notebook

For a more guided experience:

```bash
pip install jupyter
jupyter notebook
```

Then open `notebooks/easy_training_notebook.ipynb`.

### Deploy to ESPHome

1. Copy the `.tflite` file and `manifest.json` to your ESPHome configuration directory
2. Add to your ESPHome configuration:

```yaml
# Wake word configuration
micro_wake_word:
  model_file: "streaming_quantized.tflite"
  model_name: "hey_computer"
  probability_cutoff: 0.7

binary_sensor:
  - platform: micro_wake_word
    name: "Wake Word Detected"
    id: wake_word
    model_id: hey_computer

# Optional - add a text-to-speech response
on_wake_word:
  - logger.log: "Wake word detected!"
  # Add your actions here
```

## Common Wake Word Tips

- **Short wake words** (1-2 syllables): Use `--preset short`
- **Medium wake words** (3-4 syllables): Use `--preset medium`
- **Long wake words** (5+ syllables): Use `--preset long`

- **Quiet environments**: Use `--augmentation light`
- **Normal home environments**: Use `--augmentation medium`
- **Noisy environments**: Use `--augmentation heavy`

## Troubleshooting

- **False positives** (activates too often): Increase `--negative-weight` (try 25-30)
- **False negatives** (doesn't activate when it should): Decrease `--negative-weight` (try 15)
- **Model too large**: Use `--preset short` for a smaller model

## Further Reading

- [Installation Guide](installation_guide.md) - Detailed installation instructions
- [Training Best Practices](training_best_practices.md) - Tips for creating effective models
- [ESPHome Documentation](https://esphome.io/components/micro_wake_word) - How to use your model
