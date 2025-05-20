#!/usr/bin/env python
"""
Test Script for Example Selection Strategy

This script demonstrates and evaluates the example selection strategy for
improving word segmentation in Universal Dependencies (UD) POS tagging.
"""

import os
import json
import sys
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Import our modules
from example_selection import select_examples_for_segmentation, extract_features
from ud_pos_tagger_with_examples import tag_sentences_ud_with_examples
try:
    from ud_pos_tagger_grok import tag_sentences_ud as tag_sentences_ud_baseline
except ImportError:
    try:
        from ud_pos_tagger_gemini import tag_sentences_ud as tag_sentences_ud_baseline
    except ImportError:
        print("Error: Neither Grok nor Gemini baseline tagger is available.")
        print("Please make sure either ud_pos_tagger_grok.py or ud_pos_tagger_gemini.py is in the current directory.")
        sys.exit(1)

# Test sentences with various segmentation challenges
TEST_SENTENCES = [
    "I can't believe it's already 5 o'clock!",
    "The cost-benefit analysis shows we'll save $5,000.",
    "Please email john.doe@example.com for more information.",
    "She works at IBM in New York City.",
    "Dr. Smith's paper was published in the Journal of A.I.",
    "Wait - what did you say? That's incredible!",
    "The temperature is 72.5 degrees, which is quite warm.",
    "The mixture is 30% water and 70% alcohol.",
    "\"I'm tired,\" she said as she closed the book.",
    "Check (and double-check) everything before submitting."
]

def print_sentence_features(sentence: str):
    """Print the features extracted from a sentence."""
    features = extract_features(sentence)
    
    print(f"Features for: \"{sentence}\"")
    for key, value in features.items():
        if key == 'punctuation_count':
            print(f"  {key}: {value}")
        elif value:
            print(f"  {key}: {value}")

def print_top_examples(sentence: str, n: int = 3):
    """Print the top N most similar examples for a sentence."""
    examples = select_examples_for_segmentation(sentence, top_n=n)
    
    print(f"Top {n} examples for: \"{sentence}\"")
    for i, example in enumerate(examples):
        print(f"  {i+1}. \"{example['text']}\"")
        print(f"     Similarity: {example['similarity']:.3f}")
        print(f"     Segmentation: {example['segmented']}")

def compare_segmentation(sentence: str):
    """Compare segmentation with and without example selection."""
    print(f"Comparing segmentation methods for: \"{sentence}\"")
    
    # With example selection
    try:
        result_with_examples = tag_sentences_ud_with_examples(sentence)
        if result_with_examples and result_with_examples.sentences:
            tokens_with_examples = [t.token for t in result_with_examples.sentences[0].tokens]
            print(f"  With example selection: {tokens_with_examples}")
        else:
            print("  With example selection: Error")
            tokens_with_examples = []
    except Exception as e:
        print(f"  With example selection: Error - {str(e)}")
        tokens_with_examples = []
    
    # Without example selection (baseline)
    try:
        result_baseline = tag_sentences_ud_baseline(sentence)
        if result_baseline and result_baseline.sentences:
            tokens_baseline = [t.token for t in result_baseline.sentences[0].tokens]
            print(f"  Baseline: {tokens_baseline}")
        else:
            print("  Baseline: Error")
            tokens_baseline = []
    except Exception as e:
        print(f"  Baseline: Error - {str(e)}")
        tokens_baseline = []
    
    # Compare the segmentations
    if tokens_with_examples and tokens_baseline:
        if tokens_with_examples == tokens_baseline:
            print("  Result: Identical segmentation")
        else:
            print("  Result: Different segmentation")
            # Find differences
            max_len = max(len(tokens_with_examples), len(tokens_baseline))
            differences = []
            for i in range(max_len):
                if i >= len(tokens_with_examples) or i >= len(tokens_baseline) or tokens_with_examples[i] != tokens_baseline[i]:
                    differences.append(i)
            
            for i in differences:
                token_with = tokens_with_examples[i] if i < len(tokens_with_examples) else "N/A"
                token_baseline = tokens_baseline[i] if i < len(tokens_baseline) else "N/A"
                print(f"    Index {i}: \"{token_with}\" vs \"{token_baseline}\"")

def evaluate_segmentation(sentences: List[str]):
    """Evaluate segmentation with and without example selection on multiple sentences."""
    results = {
        "with_examples": {
            "total_tokens": 0,
            "errors": 0,
            "segmentation_differences": 0,
            "tokens_per_sentence": [],
            "errors_per_sentence": [],
            "processing_time": 0
        },
        "baseline": {
            "total_tokens": 0,
            "errors": 0,
            "tokens_per_sentence": [],
            "errors_per_sentence": [],
            "processing_time": 0
        },
        "sentences": []
    }
    
    # Simple gold standard for specific tests (manually verified)
    gold_standards = {
        "I can't believe it's already 5 o'clock!": 
            ["I", "ca", "n't", "believe", "it", "'s", "already", "5", "o'clock", "!"],
        "Please email john.doe@example.com for more information.": 
            ["Please", "email", "john.doe@example.com", "for", "more", "information", "."],
        "Dr. Smith's paper was published in the Journal of A.I.":
            ["Dr.", "Smith", "'s", "paper", "was", "published", "in", "the", "Journal", "of", "A.I.", "."]
    }
    
    print("\nEvaluating segmentation on test sentences...\n")
    
    for sentence in sentences:
        print(f"Evaluating: \"{sentence}\"")
        
        # Process with example selection
        start_time = time.time()
        try:
            result_with_examples = tag_sentences_ud_with_examples(sentence)
            if result_with_examples and result_with_examples.sentences:
                tokens_with_examples = [t.token for t in result_with_examples.sentences[0].tokens]
                results["with_examples"]["tokens_per_sentence"].append(len(tokens_with_examples))
            else:
                tokens_with_examples = []
                results["with_examples"]["tokens_per_sentence"].append(0)
        except Exception as e:
            tokens_with_examples = []
            results["with_examples"]["tokens_per_sentence"].append(0)
        results["with_examples"]["processing_time"] += time.time() - start_time
        
        # Process with baseline
        start_time = time.time()
        try:
            result_baseline = tag_sentences_ud_baseline(sentence)
            if result_baseline and result_baseline.sentences:
                tokens_baseline = [t.token for t in result_baseline.sentences[0].tokens]
                results["baseline"]["tokens_per_sentence"].append(len(tokens_baseline))
            else:
                tokens_baseline = []
                results["baseline"]["tokens_per_sentence"].append(0)
        except Exception as e:
            tokens_baseline = []
            results["baseline"]["tokens_per_sentence"].append(0)
        results["baseline"]["processing_time"] += time.time() - start_time
        
        # Compare with gold standard if available
        if sentence in gold_standards:
            gold_tokens = gold_standards[sentence]
            
            # Check example selection version
            if tokens_with_examples:
                errors = sum(1 for i in range(min(len(tokens_with_examples), len(gold_tokens))) 
                           if tokens_with_examples[i] != gold_tokens[i])
                errors += abs(len(tokens_with_examples) - len(gold_tokens))  # Add missing/extra tokens
                results["with_examples"]["errors"] += errors
                results["with_examples"]["total_tokens"] += len(gold_tokens)
                results["with_examples"]["errors_per_sentence"].append(errors)
            else:
                results["with_examples"]["errors_per_sentence"].append(len(gold_tokens))  # All are errors
                results["with_examples"]["errors"] += len(gold_tokens)
                results["with_examples"]["total_tokens"] += len(gold_tokens)
            
            # Check baseline version
            if tokens_baseline:
                errors = sum(1 for i in range(min(len(tokens_baseline), len(gold_tokens))) 
                           if tokens_baseline[i] != gold_tokens[i])
                errors += abs(len(tokens_baseline) - len(gold_tokens))  # Add missing/extra tokens
                results["baseline"]["errors"] += errors
                results["baseline"]["total_tokens"] += len(gold_tokens)
                results["baseline"]["errors_per_sentence"].append(errors)
            else:
                results["baseline"]["errors_per_sentence"].append(len(gold_tokens))  # All are errors
                results["baseline"]["errors"] += len(gold_tokens)
                results["baseline"]["total_tokens"] += len(gold_tokens)
        else:
            # No gold standard, just compare the two methods
            results["with_examples"]["errors_per_sentence"].append(None)
            results["baseline"]["errors_per_sentence"].append(None)
        
        # Compare the two segmentations
        if tokens_with_examples and tokens_baseline:
            if tokens_with_examples != tokens_baseline:
                results["with_examples"]["segmentation_differences"] += 1
        
        # Save detailed results
        results["sentences"].append({
            "text": sentence,
            "with_examples": tokens_with_examples,
            "baseline": tokens_baseline,
            "gold": gold_standards.get(sentence)
        })
        
        print(f"  With examples: {len(tokens_with_examples)} tokens")
        print(f"  Baseline: {len(tokens_baseline)} tokens")
        if sentence in gold_standards:
            print(f"  Gold standard: {len(gold_standards[sentence])} tokens")
        print()
    
    # Calculate error rates
    with_examples_error_rate = (results["with_examples"]["errors"] / results["with_examples"]["total_tokens"] 
                               if results["with_examples"]["total_tokens"] > 0 else 1.0)
    
    baseline_error_rate = (results["baseline"]["errors"] / results["baseline"]["total_tokens"]
                          if results["baseline"]["total_tokens"] > 0 else 1.0)
    
    print("\nSegmentation Evaluation Results:")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Total tokens in gold standard: {results['with_examples']['total_tokens']}")
    print(f"  Segmentation differences between methods: {results['with_examples']['segmentation_differences']} sentences")
    print(f"  Error rate with examples: {with_examples_error_rate:.3f}")
    print(f"  Error rate baseline: {baseline_error_rate:.3f}")
    print(f"  Improvement: {(baseline_error_rate - with_examples_error_rate) * 100:.2f}%")
    print(f"  Processing time with examples: {results['with_examples']['processing_time']:.2f} seconds")
    print(f"  Processing time baseline: {results['baseline']['processing_time']:.2f} seconds")
    
    # Plot the results
    plot_comparison(results)
    
    return results

def plot_comparison(results: Dict[str, Any]):
    """Plot a comparison of segmentation with and without example selection."""
    # Only include sentences with gold standard for error comparison
    error_indices = [i for i, err in enumerate(results["with_examples"]["errors_per_sentence"]) if err is not None]
    with_examples_errors = [results["with_examples"]["errors_per_sentence"][i] for i in error_indices]
    baseline_errors = [results["baseline"]["errors_per_sentence"][i] for i in error_indices]
    sentences = [results["sentences"][i]["text"] for i in error_indices]
    
    # Token count comparison for all sentences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error plot
    x = range(len(sentences))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], with_examples_errors, width, label='With Examples')
    ax1.bar([i + width/2 for i in x], baseline_errors, width, label='Baseline')
    
    ax1.set_xlabel('Sentence')
    ax1.set_ylabel('Segmentation Errors')
    ax1.set_title('Segmentation Errors by Sentence')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i+1}" for i in x])
    ax1.legend()
    
    # Token count plot
    token_counts_with_examples = results["with_examples"]["tokens_per_sentence"]
    token_counts_baseline = results["baseline"]["tokens_per_sentence"]
    
    ax2.bar([i - width/2 for i in range(len(sentences))], token_counts_with_examples, width, label='With Examples')
    ax2.bar([i + width/2 for i in range(len(sentences))], token_counts_baseline, width, label='Baseline')
    
    ax2.set_xlabel('Sentence')
    ax2.set_ylabel('Token Count')
    ax2.set_title('Token Count by Sentence')
    ax2.set_xticks(range(len(sentences)))
    ax2.set_xticklabels([f"{i+1}" for i in range(len(sentences))])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png')
    print("Saved visualization to segmentation_comparison.png")

def main():
    """Main function to test the example selection strategy."""
    print("Example Selection Strategy for Word Segmentation Testing\n")
    
    # Set 1: Feature extraction demonstration
    print("PART 1: Feature Extraction")
    print("-" * 50)
    for sentence in TEST_SENTENCES[:5]:
        print_sentence_features(sentence)
        print()
    
    # Set 2: Example selection demonstration
    print("PART 2: Example Selection")
    print("-" * 50)
    for sentence in TEST_SENTENCES[:5]:
        print_top_examples(sentence, n=2)
        print()
    
    # Set 3: Segmentation comparison
    print("PART 3: Segmentation Comparison")
    print("-" * 50)
    for sentence in TEST_SENTENCES[:3]:
        compare_segmentation(sentence)
        print()
    
    # Set 4: Evaluation
    print("PART 4: Segmentation Evaluation")
    print("-" * 50)
    evaluate_segmentation(TEST_SENTENCES)

if __name__ == "__main__":
    main()