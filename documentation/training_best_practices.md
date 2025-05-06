# Best Practices for Training Wake Word Models

This guide provides recommendations for training effective wake word models with microWakeWord. Following these guidelines will help you create models that are both responsive to the intended wake word and resistant to false activations.

## Choosing an Effective Wake Word

The choice of wake word significantly impacts model performance:

### Recommended Characteristics

- **Multiple syllables**: 3-5 syllables work best (e.g., "hey computer", "alexa")
- **Distinctive phonemes**: Words with unique sound combinations are easier to detect
- **Balanced consonants and vowels**: A mix of both types of sounds improves recognition
- **Uncommon in everyday speech**: Avoid words that appear frequently in normal conversation

### Examples

| Good Wake Words | Why They Work |
|-----------------|---------------|
| "Hey Computer"  | Multiple syllables, distinctive sounds |
| "Jarvis"        | Uncommon in everyday speech |
| "Alexa"         | Clear vowels, distinctive pattern |
| "Picovoice"     | Unique sound combination |

### Phonetic Spelling

Sometimes using phonetic spelling can improve recognition:

- "Computer" → "kuhm-pyoo-ter"
- "Assistant" → "uh-sis-tuhnt"
- "Alexa" → "uh-lek-suh"

## Sample Generation

The quality and quantity of training samples significantly impact model performance:

### Recommended Settings

- **Sample count**: 
  - 1,000-2,000 for testing
  - 3,000-5,000 for production models
  - More samples generally lead to better performance

- **Augmentation level**:
  - Light: For quiet, controlled environments
  - Medium: For typical home environments (recommended default)
  - Heavy: For noisy environments (kitchens, public spaces)

### Advanced Sample Generation

For even better results, consider:

1. **Multiple pronunciations**: Generate samples with different pronunciations of your wake word
2. **Varied speech rates**: Generate samples at different speaking speeds
3. **Adversarial samples**: Generate phrases that sound similar but aren't the wake word

## Model Architecture Selection

microWakeWord supports different model architectures, each with strengths:

### MixedNet (Default)

- Best for most wake words
- Good balance of accuracy and model size
- Configurable depth and width

#### Recommended Configurations:

- **Short wake words** (1-2 syllables):
  ```
  --pointwise_filters "48,48,48,48"
  --mixconv_kernel_sizes "[5],[7,11],[9,15],[17]"
  ```

- **Medium wake words** (3-4 syllables):
  ```
  --pointwise_filters "64,64,64,64"
  --mixconv_kernel_sizes "[5],[7,11],[9,15],[23]"
  ```

- **Long wake words** (5+ syllables):
  ```
  --pointwise_filters "64,64,64,64"
  --mixconv_kernel_sizes "[5],[7,11],[9,15],[29]"
  ```

### Inception

- Alternative architecture for certain wake words
- May work better for wake words with distinctive patterns
- Generally requires more computation

## Training Parameters

Fine-tuning these parameters can significantly improve model performance:

### Class Weights

- **Positive class weight**: Controls importance of correctly identifying wake word
  - Default: 1.0
  - Increase to improve recall (reduce false negatives)

- **Negative class weight**: Controls importance of correctly rejecting non-wake words
  - Default: 20.0
  - Increase to reduce false positives
  - Decrease to reduce false negatives

### Training Steps

- **Default**: 10,000-20,000 steps
- **Recommended**: 
  - Short wake words: 15,000 steps
  - Medium wake words: 20,000 steps
  - Long wake words: 25,000+ steps

### Learning Rate

- **Default**: 0.001
- **Recommendation**: The default works well for most cases
- For fine-tuning: Try a lower learning rate (0.0005) for more stable but slower training

### SpecAugment Parameters

These parameters control data augmentation during training:

- **time_mask_max_size**: Maximum size of time mask (default: 5)
- **time_mask_count**: Number of time masks (default: 2)
- **freq_mask_max_size**: Maximum size of frequency mask (default: 5)
- **freq_mask_count**: Number of frequency masks (default: 2)

Increasing these values adds more variation to training data, which can improve robustness but may require more training steps.

## Deployment Parameters

After training, these parameters control how the model behaves in real-world use:

### ESPHome Manifest Settings

- **detection_threshold**: Confidence threshold for detection
  - Default: 0.7
  - Increase to reduce false positives (0.8-0.9)
  - Decrease to reduce false negatives (0.5-0.6)

- **average_window_length**: Number of inference windows to average
  - Default: 10
  - Increase for more stable but slower detection

- **minimum_count**: Minimum detections required in window
  - Default: 3
  - Increase to reduce false positives
  - Decrease to reduce false negatives

- **suppression_ms**: Time to suppress repeated detections
  - Default: 1000 ms (1 second)
  - Adjust based on your application needs

## Troubleshooting

### Model Activates Too Often (False Positives)

1. Increase negative class weight (e.g., from 20 to 30)
2. Increase detection threshold (e.g., from 0.7 to 0.8)
3. Increase minimum_count (e.g., from 3 to 5)
4. Generate more negative samples similar to your wake word
5. Try a different wake word with more distinctive sounds

### Model Doesn't Activate When It Should (False Negatives)

1. Decrease negative class weight (e.g., from 20 to 15)
2. Decrease detection threshold (e.g., from 0.7 to 0.6)
3. Decrease minimum_count (e.g., from 3 to 2)
4. Generate more positive samples with varied pronunciations
5. Try a phonetic spelling of your wake word

### Model Size Too Large

1. Reduce the number of filters (e.g., change `--pointwise_filters "64,64,64,64"` to `"48,48,48,48"`)
2. Reduce the number of blocks (e.g., change to `"48,48,48"`)
3. Use smaller kernel sizes

## Advanced Techniques

For even better results, consider these advanced techniques:

### Ensemble Models

Train multiple models with different configurations and use a voting system to combine their predictions.

### Transfer Learning

Start with a pre-trained model and fine-tune it for your specific wake word.

### Progressive Training

Train with increasing levels of difficulty:
1. Start with clean samples
2. Add light augmentation
3. Add heavy augmentation
4. Fine-tune with challenging samples

## Conclusion

Training an effective wake word model is an iterative process. Start with these recommendations, test your model in real-world conditions, and adjust parameters based on performance. Remember that the choice of wake word itself is often the most important factor in achieving good results.
