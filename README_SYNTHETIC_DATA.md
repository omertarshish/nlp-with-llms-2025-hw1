# Using Synthetic Data to Improve POS Tagging

This document explains how to use the synthetic data generation and model retraining features developed for the optional parts of HW1.

## Overview

The implementation consists of three main components:

1. **Error Explanation**: Uses an LLM to explain and categorize POS tagging errors
2. **Synthetic Data Generation**: Creates challenging sentences with multiple instances of error categories
3. **Model Retraining**: Trains a new LogisticRegression model with the combined original and synthetic data

## Requirements

Make sure you have all the requirements from the main README.md installed, including:
- Python > 3.11
- Required libraries (`conllu`, `sklearn`, `matplotlib`, `seaborn`, etc.)
- API keys for either Gemini or Grok

## Step 1: Generate Synthetic Data

To generate 200 synthetic sentences with multiple error instances in each:

```bash
# Make sure your API keys are set as environment variables
export GROK_API_KEY=your_key_here
# OR
export GOOGLE_API_KEY=your_key_here

# Generate 200 synthetic examples with 3 errors per sentence (default)
python error_explanation.py --generate

# Or customize the number of examples and errors per sentence
python error_explanation.py --generate --examples 100 --errors-per-example 2
```

This will create a JSON file in the current directory named `synthetic_examples_XXX_YYYYMMDD_HHMMSS.json` where:
- `XXX` is the number of examples generated
- `YYYYMMDD_HHMMSS` is a timestamp

Each example in the JSON file contains:
- The full sentence
- Complete POS tagging for every word
- Challenging words with their correct tags, likely error tags, and explanations

## Step 2: Retrain the Model with Synthetic Data

After generating synthetic data, you can retrain the Logistic Regression model:

```bash
python lr_with_synthetic_data.py
```

This script:
1. Loads the original UD dataset
2. Finds and loads the most recent synthetic data file
3. Combines the datasets and trains a new Logistic Regression model
4. Compares the performance of the new model with the original model
5. Generates visualizations of error patterns and improvements

## Output and Analysis

The script produces:
- A new model file: `lr_model_with_synthetic.pkl`
- A new vectorizer file: `vectorizer_with_synthetic.pkl`
- Visualizations:
  - `confusion_matrix_comparison.png`: Side-by-side comparison of confusion matrices
  - `fixed_error_patterns.png`: Bar chart of the most common error patterns fixed

The terminal output includes:
- Accuracy comparison between the original and new models
- Counts of improvements vs. regressions
- Tag-specific performance comparison
- Analysis of the most common error patterns that were fixed

## Implementation Details

### 1. Error Explanation (`error_explanation.py`)

Uses the LLM to:
- Analyze and explain why tagging errors occur
- Categorize errors into patterns (e.g., "Ambiguity (ADJ/NOUN)", "Symbol-Punctuation Ambiguity")
- Generate synthetic examples that deliberately include challenging cases

### 2. Synthetic Data Generation

- Generates 200 sentences by default (configurable)
- Each sentence contains 3 challenging words from different error categories (configurable)
- Provides complete POS tagging for all words in the sentence
- Includes explanations of why each challenging word is difficult to tag correctly

### 3. Model Retraining (`lr_with_synthetic_data.py`)

- Combines the original UD training data with the synthetic data
- Trains a new Logistic Regression model with the same hyperparameters
- Compares performance against the original model
- Produces detailed analysis of where and how the model improved

## Best Practices

- Generate at least 200 synthetic examples for meaningful improvements
- Include 2-3 challenging words per sentence from different error categories
- Use a variety of sentence structures and vocabulary
- Focus on error categories that the original model struggled with the most

## Limitations

- API rate limits may slow down the synthetic data generation process
- Very large synthetic datasets may require adjusting batch sizes
- The quality of synthetic data depends on the LLM's understanding of POS tagging