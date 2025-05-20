#!/usr/bin/env python
"""
Logistic Regression POS Tagger with Synthetic Data

This script:
1. Loads the original UD dataset
2. Loads synthetic data generated from error_explanation.py
3. Combines the datasets and trains a Logistic Regression model
4. Evaluates the model performance and compares with the original model
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import conllu

# Path constants
TRAIN_FILE = "../UD_English-EWT/en_ewt-ud-train.conllu"
DEV_FILE = "../UD_English-EWT/en_ewt-ud-dev.conllu"
TEST_FILE = "../UD_English-EWT/en_ewt-ud-test.conllu"
LR_MODEL_FILE = "lr_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
SYNTHETIC_DATA_DIR = "."  # Directory where synthetic data is stored

def load_ud_data(file_path):
    """Load and parse a UD CoNLL-U file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return conllu.parse(f.read())

def extract_features(word, i, sentence):
    """Extract features for a word in context."""
    features = {
        'word.lower': word.lower(),
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.prefix3': word[:3].lower() if len(word) >= 3 else '<PAD>',
        'word.suffix3': word[-3:].lower() if len(word) >= 3 else '<PAD>',
        'word.prefix2': word[:2].lower() if len(word) >= 2 else '<PAD>',
        'word.suffix2': word[-2:].lower() if len(word) >= 2 else '<PAD>',
        'word.prefix1': word[:1].lower() if len(word) >= 1 else '<PAD>',
        'word.suffix1': word[-1:].lower() if len(word) >= 1 else '<PAD>',
    }
    
    # Add surrounding word features if available
    if i > 0:
        prev_word = sentence[i-1]['form']
        features.update({
            'prev_word.lower': prev_word.lower(),
            'prev_word.istitle': prev_word.istitle(),
            'prev_word.isupper': prev_word.isupper(),
            'prev_word.isdigit': prev_word.isdigit(),
        })
    else:
        features['BOS'] = True
        
    if i < len(sentence) - 1:
        next_word = sentence[i+1]['form']
        features.update({
            'next_word.lower': next_word.lower(),
            'next_word.istitle': next_word.istitle(),
            'next_word.isupper': next_word.isupper(),
            'next_word.isdigit': next_word.isdigit(),
        })
    else:
        features['EOS'] = True
        
    return features

def prepare_ud_dataset(sentences):
    """Extract features and labels from sentences."""
    X, y = [], []
    for sentence in sentences:
        for i, token in enumerate(sentence):
            if token['form'] and token['upos']:  # Skip empty tokens or missing tags
                X.append(extract_features(token['form'], i, sentence))
                y.append(token['upos'])
    return X, y

def load_synthetic_data(file_path):
    """Load synthetic data generated from error_explanation.py."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        X, y = [], []
        examples = data.get('examples', [])
        
        for example in examples:
            # Extract tokens and tags from complete_pos_tagging
            tagging = example.get('complete_pos_tagging', [])
            sentence_tokens = [item[0] for item in tagging]
            
            # Extract features and tags for each token
            for i, (token, tag) in enumerate(tagging):
                if token and tag:  # Skip empty tokens or missing tags
                    # Create a simple conllu-like structure for feature extraction
                    sentence = [{'form': t} for t in sentence_tokens]
                    X.append(extract_features(token, i, sentence))
                    y.append(tag)
        
        return X, y
    
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        return [], []

def find_latest_synthetic_data():
    """Find the most recent synthetic data file."""
    files = [f for f in os.listdir(SYNTHETIC_DATA_DIR) 
             if f.startswith('synthetic_examples_') and f.endswith('.json')]
    
    if not files:
        return None
    
    # Sort by timestamp (assuming filename format includes timestamp)
    return os.path.join(SYNTHETIC_DATA_DIR, sorted(files)[-1])

def train_model_with_synthetic_data():
    """Train a Logistic Regression model with combined original and synthetic data."""
    # Load original UD data
    train_data = load_ud_data(TRAIN_FILE)
    dev_data = load_ud_data(DEV_FILE)
    test_data = load_ud_data(TEST_FILE)
    
    # Prepare original datasets
    X_train_orig, y_train_orig = prepare_ud_dataset(train_data)
    X_dev, y_dev = prepare_ud_dataset(dev_data)
    X_test, y_test = prepare_ud_dataset(test_data)
    
    # Load synthetic data
    synthetic_file = find_latest_synthetic_data()
    if not synthetic_file:
        print("No synthetic data found. Using only original training data.")
        X_train_synth, y_train_synth = [], []
    else:
        print(f"Using synthetic data from: {synthetic_file}")
        X_train_synth, y_train_synth = load_synthetic_data(synthetic_file)
    
    # Combine original and synthetic data
    X_train_combined = X_train_orig + X_train_synth
    y_train_combined = y_train_orig + y_train_synth
    
    print(f"Original training examples: {len(X_train_orig)}")
    print(f"Synthetic training examples: {len(X_train_synth)}")
    print(f"Combined training examples: {len(X_train_combined)}")
    
    # Vectorize features
    vectorizer = DictVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_combined)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    model = LogisticRegression(
        solver='saga',
        multi_class='multinomial',
        max_iter=10,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_vec, y_train_combined)
    
    # Save the model and vectorizer
    with open("lr_model_with_synthetic.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("vectorizer_with_synthetic.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Evaluate on dev set
    y_dev_pred = model.predict(X_dev_vec)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\nDev accuracy with synthetic data: {dev_accuracy:.4f}")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy with synthetic data: {test_accuracy:.4f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'X_test_vec': X_test_vec
    }

def compare_with_original_model(new_results):
    """Compare the new model (with synthetic data) to the original model."""
    # Check if original model exists
    if not os.path.exists(LR_MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print("\n===== Model Comparison =====")
        print("Original model files not found. Skipping comparison.")
        print("To enable comparison, please run the ud_pos_tagger_sklearn.ipynb notebook to generate:")
        print(f"- {LR_MODEL_FILE}")
        print(f"- {VECTORIZER_FILE}")

        # Just report new model performance
        new_accuracy = accuracy_score(new_results['y_test'], new_results['y_test_pred'])
        print(f"\nNew model (with synthetic data) accuracy on test: {new_accuracy:.4f}")

        # Get classification report for the new model
        new_report = classification_report(new_results['y_test'], new_results['y_test_pred'], output_dict=True)

        return {
            'new_accuracy': new_accuracy,
            'new_report': new_report
        }

    # If original model exists, perform comparison
    try:
        with open(LR_MODEL_FILE, 'rb') as f:
            original_model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            original_vectorizer = pickle.load(f)

        # Load test data
        test_data = load_ud_data(TEST_FILE)
        X_test, y_test = prepare_ud_dataset(test_data)

        # Transform with original vectorizer
        X_test_vec_orig = original_vectorizer.transform(X_test)

        # Get predictions from original model
        y_test_pred_orig = original_model.predict(X_test_vec_orig)

        # Calculate accuracy for original model
        orig_accuracy = accuracy_score(y_test, y_test_pred_orig)

        # Calculate accuracy for new model
        new_accuracy = accuracy_score(new_results['y_test'], new_results['y_test_pred'])

        print("\n===== Model Comparison =====")
        print(f"Original model accuracy on test: {orig_accuracy:.4f}")
        print(f"New model (with synthetic data) accuracy on test: {new_accuracy:.4f}")
        print(f"Improvement: {(new_accuracy - orig_accuracy) * 100:.2f}%")

        # Detailed error analysis
        print("\n===== Error Analysis =====")

        # Find where models differ
        diff_indices = [i for i, (o, n) in enumerate(zip(y_test_pred_orig, new_results['y_test_pred']))
                         if o != n]

        # Count improvements vs regressions
        improvements = sum(1 for i in diff_indices
                          if y_test_pred_orig[i] != y_test[i] and new_results['y_test_pred'][i] == y_test[i])
        regressions = sum(1 for i in diff_indices
                         if y_test_pred_orig[i] == y_test[i] and new_results['y_test_pred'][i] != y_test[i])

        print(f"Total differences between models: {len(diff_indices)}")
        print(f"Improvements (original wrong → new correct): {improvements}")
        print(f"Regressions (original correct → new wrong): {regressions}")

        # Tag-specific comparison
        print("\n===== Tag-Specific Performance =====")
        # Get classification reports
        orig_report = classification_report(y_test, y_test_pred_orig, output_dict=True)
        new_report = classification_report(new_results['y_test'], new_results['y_test_pred'], output_dict=True)

        # Compare F1 scores by tag
        tags = sorted(set(y_test))
        print("Tag\tOriginal F1\tNew F1\tDifference")
        print("-" * 40)
        for tag in tags:
            if tag in orig_report and tag in new_report:
                orig_f1 = orig_report[tag]['f1-score']
                new_f1 = new_report[tag]['f1-score']
                diff = new_f1 - orig_f1
                print(f"{tag}\t{orig_f1:.4f}\t{new_f1:.4f}\t{diff:+.4f}")

        # Create confusion matrices
        plot_confusion_matrices(y_test, y_test_pred_orig, new_results['y_test_pred'])

        return {
            'original_accuracy': orig_accuracy,
            'new_accuracy': new_accuracy,
            'improvements': improvements,
            'regressions': regressions,
            'orig_report': orig_report,
            'new_report': new_report
        }

    except Exception as e:
        print(f"Error comparing with original model: {e}")
        return {
            'new_accuracy': accuracy_score(new_results['y_test'], new_results['y_test_pred']),
            'new_report': classification_report(new_results['y_test'], new_results['y_test_pred'], output_dict=True)
        }

def plot_confusion_matrices(y_true, y_pred_orig=None, y_pred_new=None):
    """Plot confusion matrices for both models, focusing on the most confused tags."""
    # If we don't have the original model predictions
    if y_pred_orig is None:
        # Just plot the new model's confusion matrix
        tags = sorted(set(y_true))
        conf_matrix_new = confusion_matrix(y_true, y_pred_new, labels=tags)

        # Convert to percentages (normalize by true labels)
        row_sums_new = conf_matrix_new.sum(axis=1, keepdims=True)
        conf_matrix_new_pct = np.divide(conf_matrix_new, row_sums_new,
                                       out=np.zeros_like(conf_matrix_new, dtype=float),
                                       where=row_sums_new!=0) * 100

        # Find most confused tags (top 5)
        tag_errors = {}
        for i, tag in enumerate(tags):
            # Calculate confusion for this tag (excluding the diagonal)
            row = conf_matrix_new_pct[i].copy()
            row[i] = 0  # Exclude correct predictions
            tag_errors[tag] = np.sum(row)

        # Get top tags with the most confusion
        top_tags = sorted(tag_errors.items(), key=lambda x: x[1], reverse=True)[:5]
        top_indices = [tags.index(tag) for tag, _ in top_tags]

        # Plot confusion for top 5 most confused tags
        plt.figure(figsize=(15, 10))

        for i, idx in enumerate(top_indices):
            tag = tags[idx]

            plt.subplot(len(top_indices), 1, i+1)
            plt.bar(tags, conf_matrix_new_pct[idx])
            plt.title(f"Confusion for Tag: {tag}")
            plt.xlabel("Predicted Tag")
            plt.ylabel("Percentage of True Instances")
            plt.xticks(rotation=90)
            plt.ylim(0, 100)

            # Highlight the correct tag in green
            bars = plt.gca().patches
            bars[idx].set_facecolor('green')

        plt.tight_layout()
        plt.savefig("confusion_matrix_new_only.png")
        print("Saved confusion matrix to confusion_matrix_new_only.png")
        return

    # If we have both models, do the comparison
    tags = sorted(set(y_true))
    conf_matrix_orig = confusion_matrix(y_true, y_pred_orig, labels=tags)
    conf_matrix_new = confusion_matrix(y_true, y_pred_new, labels=tags)

    # Convert to percentages (normalize by true labels)
    row_sums_orig = conf_matrix_orig.sum(axis=1, keepdims=True)
    conf_matrix_orig_pct = (conf_matrix_orig / row_sums_orig) * 100

    row_sums_new = conf_matrix_new.sum(axis=1, keepdims=True)
    conf_matrix_new_pct = (conf_matrix_new / row_sums_new) * 100

    # Calculate difference matrix
    diff_matrix = conf_matrix_new_pct - conf_matrix_orig_pct

    # Find most significant differences (focusing on top 5 tags)
    tag_errors = {}
    for i, tag in enumerate(tags):
        # Sum absolute differences for each tag
        tag_errors[tag] = np.sum(np.abs(diff_matrix[i]))

    # Get top tags with the most significant changes
    top_tags = sorted(tag_errors.items(), key=lambda x: x[1], reverse=True)[:5]
    top_indices = [tags.index(tag) for tag, _ in top_tags]

    # Create side-by-side plots for the top tags
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(top_indices):
        tag = tags[idx]

        # Original model confusion for this tag
        plt.subplot(len(top_indices), 3, i*3 + 1)
        plt.bar(tags, conf_matrix_orig_pct[idx])
        plt.title(f"Original: {tag} → Predicted")
        plt.xticks(rotation=90)
        plt.ylim(0, 100)

        # New model confusion for this tag
        plt.subplot(len(top_indices), 3, i*3 + 2)
        plt.bar(tags, conf_matrix_new_pct[idx])
        plt.title(f"New: {tag} → Predicted")
        plt.xticks(rotation=90)
        plt.ylim(0, 100)

        # Difference
        plt.subplot(len(top_indices), 3, i*3 + 3)
        plt.bar(tags, diff_matrix[idx])
        plt.title(f"Difference: {tag}")
        plt.xticks(rotation=90)
        plt.axhline(y=0, color='k', linestyle='-')

    plt.tight_layout()
    plt.savefig("confusion_matrix_comparison.png")
    print("Saved confusion matrix comparison to confusion_matrix_comparison.png")

def error_pattern_analysis(comparison_results, new_results):
    """Analyze error patterns that were improved in the new model."""
    # Check if we have original model results for comparison
    if 'original_accuracy' not in comparison_results:
        print("\n===== Error Pattern Analysis =====")
        print("Original model not available. Skipping error pattern analysis.")

        # Instead, analyze the errors made by the new model
        test_data = load_ud_data(TEST_FILE)
        X_test, y_test = prepare_ud_dataset(test_data)

        # Get error distribution for the new model
        y_pred = new_results['y_test_pred']
        errors = [(true, pred) for true, pred in zip(y_test, y_pred) if true != pred]
        error_patterns = Counter(errors)

        print("\n===== Most Common Error Patterns in New Model =====")
        print("True → Predicted : Count")
        print("-" * 40)
        for (true, pred), count in error_patterns.most_common(10):
            print(f"{true} → {pred} : {count}")

        # Create a bar chart of the top 10 error patterns
        top_patterns = error_patterns.most_common(10)
        labels = [f"{true}->{pred}" for (true, pred), _ in top_patterns]
        counts = [count for _, count in top_patterns]

        plt.figure(figsize=(12, 6))
        plt.bar(labels, counts)
        plt.title("Top 10 Error Patterns in Model with Synthetic Data")
        plt.xlabel("Error Pattern (True → Predicted)")
        plt.ylabel("Number of Errors")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("error_patterns.png")
        print("Saved error pattern analysis to error_patterns.png")
        return

    # If we have original model results, perform comparative analysis
    # Identify cases where the original model was wrong but the new model is correct
    improvements = []
    tags_improved = []

    for i, (y_true, y_orig, y_new) in enumerate(zip(
            comparison_results['y_test'],
            comparison_results['y_test_pred_orig'],
            new_results['y_test_pred'])):

        if y_orig != y_true and y_new == y_true:
            improvements.append((y_true, y_orig))
            tags_improved.append((y_orig, y_true))  # (from_incorrect, to_correct)

    # Count the most common error patterns that were fixed
    error_patterns = Counter(tags_improved)

    print("\n===== Most Common Fixed Error Patterns =====")
    print("Original → Correct : Count")
    print("-" * 40)
    for (incorrect, correct), count in error_patterns.most_common(10):
        print(f"{incorrect} → {correct} : {count}")

    # Create a bar chart of the top 10 fixed error patterns
    top_patterns = error_patterns.most_common(10)
    labels = [f"{orig}->{corr}" for (orig, corr), _ in top_patterns]
    counts = [count for _, count in top_patterns]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.title("Top 10 Error Patterns Fixed with Synthetic Data")
    plt.xlabel("Error Pattern (Original → Correct)")
    plt.ylabel("Number of Fixes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("fixed_error_patterns.png")
    print("Saved error pattern analysis to fixed_error_patterns.png")

def main():
    # Train model with synthetic data
    print("Training model with synthetic data...")
    new_results = train_model_with_synthetic_data()
    
    # Compare with original model
    print("\nComparing with original model...")
    comparison_results = compare_with_original_model(new_results)
    
    if comparison_results:
        # Perform detailed error pattern analysis
        error_pattern_analysis(comparison_results, new_results)
        
        print("\nAnalysis complete. See generated plots for detailed comparisons.")

if __name__ == "__main__":
    main()