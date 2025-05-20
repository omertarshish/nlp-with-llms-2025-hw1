"""
LLM POS Tagger Error Analysis

This script loads hard sentences identified by the Logistic Regression tagger
and compares the performance of the LLM-based tagger against the LR tagger.

It reports detailed error statistics and visualizations of the comparison.
"""

import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

# Import taggers
from ud_pos_tagger_grok import tag_sentences_ud as grok_tagger
from ud_pos_tagger_grok import TaggedSentences, TokenPOS, SentencePOS
# Uncomment to use Gemini instead
# from ud_pos_tagger_gemini import tag_sentences_ud as gemini_tagger

# Global settings
LLM_MODEL = "grok-3-mini"  # or "gemini"
LLM_TAGGER = grok_tagger  # or gemini_tagger
BATCH_SIZE = 5  # Smaller batches for more reliable API calls
SAVE_RESULTS = True  # Save results to file
PLOT_FIGURES = True  # Create and save figure visualizations


def load_hard_sentences(filename=None):
    """
    Load the hard sentences identified by the LR tagger.
    
    If no file is provided, it attempts to load from a standard location,
    or creates a dummy set for testing.
    
    Returns:
        List of (sentence, error_count, predicted_tags) tuples
    """
    if filename and os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print("No hard sentences file found. Creating dummy data for testing.")
        # Create dummy data for testing
        return [
            (
                [("I", "PRON"), ("like", "VERB"), ("that", "DET"), ("book", "NOUN")], 
                1, 
                ["PRON", "VERB", "SCONJ", "NOUN"]  # Error: DET -> SCONJ
            ),
            (
                [("The", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("mat", "NOUN")],
                0,
                ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]  # No errors
            ),
            (
                [("She", "PRON"), ("looked", "VERB"), ("up", "ADV"), ("the", "DET"), ("answer", "NOUN")],
                1,
                ["PRON", "VERB", "ADP", "DET", "NOUN"]  # Error: ADV -> ADP
            )
        ]


def process_hard_sentences(hard_sentences, limit=100):
    """
    Process the hard sentences through the LLM tagger.
    
    Args:
        hard_sentences: List of (sentence, error_count, predicted_tags) tuples
        limit: Maximum number of sentences to process (for rate limiting)
        
    Returns:
        Tuple of (gold_sentences, lr_predictions, llm_predictions) 
    """
    print(f"Processing up to {limit} hard sentences using {LLM_MODEL} model...")
    
    # Filter sentences with 1-3 errors
    filtered_sentences = [(s, e, p) for s, e, p in hard_sentences if 1 <= e <= 3]
    filtered_sentences = filtered_sentences[:limit]
    
    # Prepare data structures
    gold_standard = []
    lr_predictions = []
    
    # Format data
    for sentence, error_count, lr_pred in filtered_sentences:
        # Create sentence text
        text = " ".join([token for token, _ in sentence])
        
        # Format gold standard
        gold_tokens = []
        for token, tag in sentence:
            gold_tokens.append(TokenPOS(token=token, pos_tag=tag))
        gold_standard.append(SentencePOS(tokens=gold_tokens))
        
        # Format LR predictions
        lr_tokens = []
        for (token, _), tag in zip([pair for pair in sentence], lr_pred):
            lr_tokens.append(TokenPOS(token=token, pos_tag=tag))
        lr_predictions.append(SentencePOS(tokens=lr_tokens))
    
    # Process sentences in batches
    sentences_to_process = [" ".join([token for token, _ in s]) for s, _, _ in filtered_sentences]
    llm_results = []
    
    # Process in smaller batches to be safe with API limits
    for i in range(0, len(sentences_to_process), BATCH_SIZE):
        batch = sentences_to_process[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(sentences_to_process) + BATCH_SIZE - 1)//BATCH_SIZE}...")
        
        try:
            # Process each sentence individually for more reliable results
            for sentence in batch:
                result = LLM_TAGGER(sentence)
                if result and hasattr(result, 'sentences'):
                    llm_results.extend(result.sentences)
            
            # Wait between batches for rate limiting
            if i + BATCH_SIZE < len(sentences_to_process):
                print("Waiting between batches...")
                import time
                time.sleep(2)  # Adjust based on API limits
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    # Check if results align
    if len(llm_results) != len(gold_standard):
        print(f"Warning: Different number of results - LLM: {len(llm_results)}, Gold: {len(gold_standard)}")
        # Truncate to match lengths
        min_len = min(len(llm_results), len(gold_standard))
        llm_results = llm_results[:min_len]
        gold_standard = gold_standard[:min_len]
        lr_predictions = lr_predictions[:min_len]
    
    # Create the TaggedSentences objects
    gold_tagged = TaggedSentences(sentences=gold_standard)
    lr_tagged = TaggedSentences(sentences=lr_predictions)
    llm_tagged = TaggedSentences(sentences=llm_results)
    
    return gold_tagged, lr_tagged, llm_tagged


def token_level_metrics(gold_tagged, predictions_tagged, model_name="LLM"):
    """
    Compute token-level evaluation metrics.
    
    Args:
        gold_tagged: Ground truth tags
        predictions_tagged: Predicted tags 
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with token-level metrics
    """
    total_tokens = 0
    correct_tokens = 0
    errors_by_tag = defaultdict(int)
    total_by_tag = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Analyze each token
    for gold_sent, pred_sent in zip(gold_tagged.sentences, predictions_tagged.sentences):
        for gold_token, pred_token in zip(gold_sent.tokens, pred_sent.tokens):
            tag_gold = gold_token.pos_tag
            tag_pred = pred_token.pos_tag
            token = gold_token.token
            
            total_tokens += 1
            total_by_tag[tag_gold] += 1
            
            if tag_gold == tag_pred:
                correct_tokens += 1
            else:
                errors_by_tag[tag_gold] += 1
                confusion_matrix[tag_gold][tag_pred] += 1
    
    # Calculate metrics
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    accuracy_by_tag = {}
    
    for tag in total_by_tag:
        correct = total_by_tag[tag] - errors_by_tag[tag]
        accuracy_by_tag[tag] = correct / total_by_tag[tag] if total_by_tag[tag] > 0 else 0
    
    # Calculate per-tag precision, recall, F1
    precision = {}
    recall = {}
    f1_score = {}
    
    for tag in total_by_tag:
        # Calculate true positives, false positives, false negatives
        tp = total_by_tag[tag] - errors_by_tag[tag]
        fp = sum(confusion_matrix[t][tag] for t in confusion_matrix if t != tag)
        fn = errors_by_tag[tag]
        
        # Calculate metrics
        precision[tag] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[tag] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[tag] = 2 * precision[tag] * recall[tag] / (precision[tag] + recall[tag]) if (precision[tag] + recall[tag]) > 0 else 0
    
    # Calculate macro-average metrics
    macro_precision = sum(precision.values()) / len(precision) if precision else 0
    macro_recall = sum(recall.values()) / len(recall) if recall else 0
    macro_f1 = sum(f1_score.values()) / len(f1_score) if f1_score else 0
    
    results = {
        "model": model_name,
        "total_tokens": total_tokens,
        "correct_tokens": correct_tokens,
        "accuracy": accuracy,
        "errors_by_tag": dict(errors_by_tag),
        "total_by_tag": dict(total_by_tag),
        "accuracy_by_tag": accuracy_by_tag,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
    }
    
    return results


def sentence_level_metrics(gold_tagged, predictions_tagged, model_name="LLM"):
    """
    Compute sentence-level evaluation metrics.
    
    Args:
        gold_tagged: Ground truth tags
        predictions_tagged: Predicted tags 
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with sentence-level metrics
    """
    perfect_sentences = 0
    error_counts = defaultdict(int)
    
    for gold_sent, pred_sent in zip(gold_tagged.sentences, predictions_tagged.sentences):
        errors = 0
        
        for gold_token, pred_token in zip(gold_sent.tokens, pred_sent.tokens):
            if gold_token.pos_tag != pred_token.pos_tag:
                errors += 1
        
        if errors == 0:
            perfect_sentences += 1
        
        error_counts[errors] += 1
    
    sentence_accuracy = perfect_sentences / len(gold_tagged.sentences) if gold_tagged.sentences else 0
    
    results = {
        "model": model_name,
        "total_sentences": len(gold_tagged.sentences),
        "perfect_sentences": perfect_sentences,
        "sentence_accuracy": sentence_accuracy,
        "error_counts": dict(error_counts)
    }
    
    return results


def compare_tagger_errors(gold_tagged, lr_tagged, llm_tagged):
    """
    Detailed comparison between LR and LLM taggers.
    
    Args:
        gold_tagged: Ground truth tags
        lr_tagged: LR model predictions 
        llm_tagged: LLM model predictions
        
    Returns:
        Dictionary with comparative metrics
    """
    # Track different error categories
    fixed_by_llm = []
    new_in_llm = []
    common_errors = []
    
    # Track token-level patterns
    lr_error_tokens = []
    llm_error_tokens = []
    
    # Analyze each token
    for sent_idx, (gold_sent, lr_sent, llm_sent) in enumerate(zip(gold_tagged.sentences, lr_tagged.sentences, llm_tagged.sentences)):
        for token_idx, (gold_token, lr_token, llm_token) in enumerate(zip(gold_sent.tokens, lr_sent.tokens, llm_sent.tokens)):
            gold_tag = gold_token.pos_tag
            lr_tag = lr_token.pos_tag
            llm_tag = llm_token.pos_tag
            token = gold_token.token
            
            # Check error patterns
            lr_error = (gold_tag != lr_tag)
            llm_error = (gold_tag != llm_tag)
            
            # Collect error tokens
            if lr_error:
                lr_error_tokens.append((token, gold_tag, lr_tag))
            
            if llm_error:
                llm_error_tokens.append((token, gold_tag, llm_tag))
            
            # Analyze error patterns
            if lr_error and not llm_error:
                # LLM fixed an error made by LR
                fixed_by_llm.append((token, gold_tag, lr_tag, sent_idx, token_idx))
            elif not lr_error and llm_error:
                # LLM introduced a new error
                new_in_llm.append((token, gold_tag, llm_tag, sent_idx, token_idx))
            elif lr_error and llm_error:
                # Both made errors
                common_errors.append((token, gold_tag, lr_tag, llm_tag, sent_idx, token_idx))
    
    # Summarize common errors by tag
    lr_errors_by_tag = Counter([g for _, g, _ in lr_error_tokens])
    llm_errors_by_tag = Counter([g for _, g, _ in llm_error_tokens])

    # Frequent error patterns
    lr_error_patterns = Counter([f"{g}->{p}" for _, g, p in lr_error_tokens])
    llm_error_patterns = Counter([f"{g}->{p}" for _, g, p in llm_error_tokens])
    
    # Track which tokens are most often misclassified
    lr_difficult_tokens = Counter([t for t, _, _ in lr_error_tokens])
    llm_difficult_tokens = Counter([t for t, _, _ in llm_error_tokens])
    
    # Analyze tags that LLM handles better/worse
    tag_differences = {}
    for tag in set(lr_errors_by_tag.keys()) | set(llm_errors_by_tag.keys()):
        lr_errors = lr_errors_by_tag.get(tag, 0)
        llm_errors = llm_errors_by_tag.get(tag, 0)
        tag_differences[tag] = {
            "lr_errors": lr_errors,
            "llm_errors": llm_errors,
            "difference": lr_errors - llm_errors
        }
    
    results = {
        "fixed_by_llm": {
            "count": len(fixed_by_llm),
            "examples": fixed_by_llm[:20]  # Limit examples
        },
        "new_in_llm": {
            "count": len(new_in_llm),
            "examples": new_in_llm[:20]
        },
        "common_errors": {
            "count": len(common_errors),
            "examples": common_errors[:20]
        },
        "lr_errors": len(lr_error_tokens),
        "llm_errors": len(llm_error_tokens),
        "error_reduction": len(lr_error_tokens) - len(llm_error_tokens),
        "error_reduction_percent": (len(lr_error_tokens) - len(llm_error_tokens)) / len(lr_error_tokens) if lr_error_tokens else 0,
        "tag_differences": tag_differences,
        "lr_difficult_tokens": dict(lr_difficult_tokens.most_common(20)),
        "llm_difficult_tokens": dict(llm_difficult_tokens.most_common(20)),
        "lr_error_patterns": dict(lr_error_patterns.most_common(20)),
        "llm_error_patterns": dict(llm_error_patterns.most_common(20))
    }
    
    return results


def plot_error_comparison(lr_metrics, llm_metrics, save_path=None):
    """
    Plot comparative visualizations of tagger errors.
    
    Args:
        lr_metrics: LR tagger metrics
        llm_metrics: LLM tagger metrics
        save_path: Path to save figures
    """
    # Create output directory if needed
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 1. Plot overall accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ['LR Tagger', 'LLM Tagger']
    accuracies = [lr_metrics['accuracy'], llm_metrics['accuracy']]
    
    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylim(0.5, 1.0)  # Assuming accuracy is above 50%
    plt.title('Accuracy Comparison on Hard Sentences')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # 2. Plot accuracy by tag
    plt.figure(figsize=(14, 8))
    
    # Get common tags
    all_tags = sorted(set(lr_metrics['accuracy_by_tag'].keys()) | set(llm_metrics['accuracy_by_tag'].keys()))
    
    x = np.arange(len(all_tags))
    width = 0.35
    
    lr_acc_by_tag = [lr_metrics['accuracy_by_tag'].get(tag, 0) for tag in all_tags]
    llm_acc_by_tag = [llm_metrics['accuracy_by_tag'].get(tag, 0) for tag in all_tags]
    
    plt.bar(x - width/2, lr_acc_by_tag, width, label='LR Tagger')
    plt.bar(x + width/2, llm_acc_by_tag, width, label='LLM Tagger')
    
    plt.xlabel('POS Tags')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by POS Tag')
    plt.xticks(x, all_tags, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'accuracy_by_tag.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # 3. Plot sentence error distribution
    plt.figure(figsize=(12, 6))
    
    # Get error counts up to 10 errors per sentence
    error_bins = range(11)
    lr_counts = [lr_metrics['sentence_level']['error_counts'].get(i, 0) for i in error_bins]
    llm_counts = [llm_metrics['sentence_level']['error_counts'].get(i, 0) for i in error_bins]
    
    x = np.arange(len(error_bins))
    width = 0.35
    
    plt.bar(x - width/2, lr_counts, width, label='LR Tagger')
    plt.bar(x + width/2, llm_counts, width, label='LLM Tagger')
    
    plt.xlabel('Number of Errors per Sentence')
    plt.ylabel('Number of Sentences')
    plt.title('Distribution of Errors per Sentence')
    plt.xticks(x, error_bins)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'sentence_error_distribution.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()


def run_error_analysis(hard_sentences_file=None, limit=100):
    """
    Run the complete error analysis workflow.
    
    Args:
        hard_sentences_file: Path to the file with hard sentences
        limit: Maximum number of sentences to process
    
    Returns:
        Dictionary with complete analysis results
    """
    print(f"Starting error analysis using {LLM_MODEL} model")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load hard sentences
    hard_sentences = load_hard_sentences(hard_sentences_file)
    print(f"Loaded {len(hard_sentences)} hard sentences")
    
    # 2. Process hard sentences through LLM tagger
    gold_tagged, lr_tagged, llm_tagged = process_hard_sentences(hard_sentences, limit)
    print(f"Processed {len(gold_tagged.sentences)} sentences through {LLM_MODEL} tagger")
    
    # 3. Compute token-level metrics
    lr_token_metrics = token_level_metrics(gold_tagged, lr_tagged, "LR")
    llm_token_metrics = token_level_metrics(gold_tagged, llm_tagged, "LLM")
    print(f"LR Accuracy: {lr_token_metrics['accuracy']:.4f}, LLM Accuracy: {llm_token_metrics['accuracy']:.4f}")
    
    # 4. Compute sentence-level metrics
    lr_sentence_metrics = sentence_level_metrics(gold_tagged, lr_tagged, "LR")
    llm_sentence_metrics = sentence_level_metrics(gold_tagged, llm_tagged, "LLM")
    print(f"LR Sentence Accuracy: {lr_sentence_metrics['sentence_accuracy']:.4f}, "
          f"LLM Sentence Accuracy: {llm_sentence_metrics['sentence_accuracy']:.4f}")
    
    # 5. Detailed error comparison
    comparison = compare_tagger_errors(gold_tagged, lr_tagged, llm_tagged)
    print(f"LLM fixed {comparison['fixed_by_llm']['count']} errors and introduced {comparison['new_in_llm']['count']} new errors")
    
    # 6. Combine all results
    results = {
        "timestamp": timestamp,
        "llm_model": LLM_MODEL,
        "num_processed": len(gold_tagged.sentences),
        "token_level": {
            "lr": lr_token_metrics,
            "llm": llm_token_metrics
        },
        "sentence_level": {
            "lr": lr_sentence_metrics,
            "llm": llm_sentence_metrics
        },
        "comparison": comparison
    }
    
    # 7. Save results
    if SAVE_RESULTS:
        results_file = f"error_analysis_{LLM_MODEL}_{timestamp}.json"
        with open(results_file, "w") as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            json.dump(results, f, default=lambda o: dict(o) if isinstance(o, defaultdict) else o, indent=2)
        print(f"Results saved to {results_file}")
    
    # 8. Create visualizations
    if PLOT_FIGURES:
        save_path = f"figures_{LLM_MODEL}_{timestamp}"
        plot_error_comparison(
            {"accuracy": lr_token_metrics["accuracy"], 
             "accuracy_by_tag": lr_token_metrics["accuracy_by_tag"],
             "sentence_level": lr_sentence_metrics},
            {"accuracy": llm_token_metrics["accuracy"], 
             "accuracy_by_tag": llm_token_metrics["accuracy_by_tag"],
             "sentence_level": llm_sentence_metrics},
            save_path
        )
        print(f"Figures saved to {save_path}")
    
    return results


def generate_enhanced_prompt(error_analysis_results=None):
    """
    Generate an enhanced prompt based on error analysis results.

    Args:
        error_analysis_results: Results from error analysis

    Returns:
        Improved prompt text for LLM tagger
    """
    # Extract common error patterns if available
    common_patterns = []
    if error_analysis_results and 'comparison' in error_analysis_results:
        # Extract common error patterns from the analysis
        if 'llm_error_patterns' in error_analysis_results['comparison']:
            common_patterns = error_analysis_results['comparison']['llm_error_patterns']

    # Base prompt with enhanced instructions
    prompt = """You are a linguist specializing in Universal Dependencies (UD) part-of-speech tagging.
Your task is to analyze the following text and assign a UD POS tag to each token.

Here are the Universal Dependencies POS tags and their meanings:
- ADJ: adjective (e.g., big, old, green, incomprehensible)
- ADP: adposition (e.g., in, to, during)
- ADV: adverb (e.g., very, tomorrow, down, where, when)
- AUX: auxiliary verb (e.g., is, has been, should, would, could, might, must)
- CCONJ: coordinating conjunction (e.g., and, or, but, nor, so, yet)
- DET: determiner (e.g., a, the, this, these, my, your, which)
- INTJ: interjection (e.g., psst, ouch, bravo, hello)
- NOUN: noun (e.g., girl, cat, tree, air, beauty, software)
- NUM: numeral (e.g., 1, 2017, one, seven, III, first, second)
- PART: particle (e.g., 's, not, to [in infinitives], 've, 'll, 'd)
- PRON: pronoun (e.g., I, you, he, herself, themselves, who, what)
- PROPN: proper noun (e.g., Mary, John, London, NASA, Google)
- PUNCT: punctuation (e.g., ., (, ), ?, !, :, ;)
- SCONJ: subordinating conjunction (e.g., if, while, that, because, since)
- SYM: symbol (e.g., $, %, §, ©, +, :), =, @, emoji)
- VERB: verb (e.g., run, eat, works, accepted, thinking, typed)
- X: other (e.g., foreign words, typos, abbreviations)

Important tagging rules:
1. Pay special attention to frequently confused tags:
   - "that" can be:
     * SCONJ when introducing a clause (e.g., "She said that he left.")
     * DET when determining a noun (e.g., "That car is red.")
     * PRON when acting as a relative pronoun (e.g., "The car that I bought is red.")

   - "to" can be:
     * PART when it's an infinitive marker (e.g., "I want to go.")
     * ADP when it's a preposition (e.g., "I went to the store.")

   - Words like "up", "out", "down" can be:
     * ADV when modifying a verb/action (e.g., "Look up at the sky")
     * ADP when acting as a preposition (e.g., "Go up the stairs")
     * PART when part of a phrasal verb (e.g., "Pick up the book")

   - "like" can be:
     * VERB when expressing preference (e.g., "I like this book.")
     * ADP when expressing similarity (e.g., "It looks like a cat.")
     * SCONJ when introducing a clause of manner (e.g., "Do it like I showed you.")

   - "as" is typically:
     * SCONJ when introducing a clause (e.g., "As I was saying...")
     * ADP when used in comparisons (e.g., "As tall as a tree")

2. Remember that auxiliary verbs (be, have, do, will, shall, would, can, could, may, might, must, should) are tagged as AUX, not VERB.

3. For proper nouns (PROPN), include names of specific entities, organizations, locations, etc.

4. For ambiguous cases, consider the syntactic function in the sentence.

5. Special tokenization rules:
   - Currency symbols like $ should be separate tokens tagged as SYM
   - Percentages like 25% should be split into the number (25) and the symbol (%)
   - Punctuation marks should be their own tokens, even when part of abbreviations (e.g. "Inc." → "Inc" + ".")
   - Hyphens in compound words should be retained (e.g., "state-of-the-art" remains one token)

Here are examples of sentences with challenging cases:

Example 1: "She claimed that that car that we saw yesterday was a Tesla."
Tags: [PRON, VERB, SCONJ, DET, NOUN, PRON, PRON, VERB, ADV, AUX, DET, PROPN, PUNCT]
Explanation: First "that" is SCONJ introducing a clause, second "that" is DET modifying "car", third "that" is PRON (relative pronoun).

Example 2: "I like to run like I used to when I was like 20 years old."
Tags: [PRON, VERB, PART, VERB, SCONJ, PRON, VERB, PART, ADV, PRON, AUX, ADP, NUM, NOUN, ADJ, PUNCT]
Explanation: First "like" is VERB (preference), second "like" is SCONJ (manner), third "like" is ADP (approximation).

Example 3: "She looked up the answer to see if it was correct."
Tags: [PRON, VERB, ADV, DET, NOUN, PART, VERB, SCONJ, PRON, AUX, ADJ, PUNCT]
Explanation: "up" is ADV (direction), first "to" is PART (infinitive marker).

Return the result in the specified JSON format with each sentence as a list of tokens and their POS tags.
"""

    # Add common error patterns from analysis
    if common_patterns:
        prompt += "\n\nBased on error analysis, pay special attention to these patterns:\n"
        for pattern, count in common_patterns.items():
            prompt += f"- {pattern}: This pattern was confused {count} times\n"

    return prompt


def generate_difficult_examples():
    """
    Generate examples of sentences that are likely to be difficult
    for the LLM tagger based on error analysis patterns.

    Returns:
        List of challenging sentences with explanations
    """
    challenging_examples = [
        {
            "sentence": "She hoped that that report that she submitted was comprehensive.",
            "challenge": "Multiple 'that' tokens with different grammatical roles (subordinating conjunction, determiner, relative pronoun)",
            "expected_tags": [
                ("She", "PRON"),
                ("hoped", "VERB"),
                ("that", "SCONJ"),
                ("that", "DET"),
                ("report", "NOUN"),
                ("that", "PRON"),
                ("she", "PRON"),
                ("submitted", "VERB"),
                ("was", "AUX"),
                ("comprehensive", "ADJ"),
                (".", "PUNCT")
            ]
        },
        {
            "sentence": "The up-to-date report highlighted that up to 25% of users had signed up for the program.",
            "challenge": "Multiple uses of 'up' in different contexts (as part of compound adjective, as part of multiword preposition, as adverbial particle)",
            "expected_tags": [
                ("The", "DET"),
                ("up-to-date", "ADJ"),
                ("report", "NOUN"),
                ("highlighted", "VERB"),
                ("that", "SCONJ"),
                ("up", "ADP"),
                ("to", "ADP"),
                ("25", "NUM"),
                ("%", "SYM"),
                ("of", "ADP"),
                ("users", "NOUN"),
                ("had", "AUX"),
                ("signed", "VERB"),
                ("up", "ADV"),
                ("for", "ADP"),
                ("the", "DET"),
                ("program", "NOUN"),
                (".", "PUNCT")
            ]
        },
        {
            "sentence": "The average age is 25, not 15, and the rate is 10%, not 5%, which makes a significant difference.",
            "challenge": "Mixed numeric expressions with punctuation and percentages",
            "expected_tags": [
                ("The", "DET"),
                ("average", "ADJ"),
                ("age", "NOUN"),
                ("is", "AUX"),
                ("25", "NUM"),
                (",", "PUNCT"),
                ("not", "PART"),
                ("15", "NUM"),
                (",", "PUNCT"),
                ("and", "CCONJ"),
                ("the", "DET"),
                ("rate", "NOUN"),
                ("is", "AUX"),
                ("10", "NUM"),
                ("%", "SYM"),
                (",", "PUNCT"),
                ("not", "PART"),
                ("5", "NUM"),
                ("%", "SYM"),
                (",", "PUNCT"),
                ("which", "PRON"),
                ("makes", "VERB"),
                ("a", "DET"),
                ("significant", "ADJ"),
                ("difference", "NOUN"),
                (".", "PUNCT")
            ]
        },
        {
            "sentence": "I like to run like I used to when I was like 20 years old.",
            "challenge": "Multiple 'like' tokens with different grammatical roles (verb, adposition, adverb)",
            "expected_tags": [
                ("I", "PRON"),
                ("like", "VERB"),
                ("to", "PART"),
                ("run", "VERB"),
                ("like", "SCONJ"),
                ("I", "PRON"),
                ("used", "VERB"),
                ("to", "PART"),
                ("when", "ADV"),
                ("I", "PRON"),
                ("was", "AUX"),
                ("like", "ADP"),
                ("20", "NUM"),
                ("years", "NOUN"),
                ("old", "ADJ"),
                (".", "PUNCT")
            ]
        },
        {
            "sentence": "Regarding light-weight materials, carbon-fiber composites are the most cost-effective at $100/kg.",
            "challenge": "Hyphenated compounds, technical terms, and special symbols",
            "expected_tags": [
                ("Regarding", "VERB"),
                ("light-weight", "ADJ"),
                ("materials", "NOUN"),
                (",", "PUNCT"),
                ("carbon-fiber", "ADJ"),
                ("composites", "NOUN"),
                ("are", "AUX"),
                ("the", "DET"),
                ("most", "ADV"),
                ("cost-effective", "ADJ"),
                ("at", "ADP"),
                ("$", "SYM"),
                ("100", "NUM"),
                ("/", "SYM"),
                ("kg", "NOUN"),
                (".", "PUNCT")
            ]
        }
    ]
    
    return challenging_examples


if __name__ == "__main__":
    try:
        # Run the complete analysis
        results = run_error_analysis(limit=20)  # Start with small limit for testing

        # Print error highlights
        if results:
            print("\n=== ERROR ANALYSIS HIGHLIGHTS ===")
            print(f"LLM Model: {results['llm_model']}")
            print(f"Total sentences analyzed: {results['num_processed']}")
            print(f"LR Token Accuracy: {results['token_level']['lr']['accuracy']:.4f}")
            print(f"LLM Token Accuracy: {results['token_level']['llm']['accuracy']:.4f}")
            print(f"Errors fixed by LLM: {results['comparison']['fixed_by_llm']['count']}")
            print(f"New errors introduced by LLM: {results['comparison']['new_in_llm']['count']}")
            print("===================================")

            # Generate and print challenging examples
            print("\n=== CHALLENGING EXAMPLES FOR LLM TAGGERS ===")
            difficult_examples = generate_difficult_examples()
            for i, example in enumerate(difficult_examples, 1):
                print(f"\nExample {i}: {example['sentence']}")
                print(f"Challenge: {example['challenge']}")
                print("Expected tags:")
                for token, tag in example['expected_tags']:
                    print(f"  {token:<15} {tag}")
            print("===================================")

            # Generate enhanced prompt based on error analysis
            enhanced_prompt = generate_enhanced_prompt(results)

            # Save the enhanced prompt to a file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"enhanced_prompt_{LLM_MODEL}_{timestamp}.txt", "w") as f:
                f.write(enhanced_prompt)
            print(f"\nEnhanced prompt saved to enhanced_prompt_{LLM_MODEL}_{timestamp}.txt")

            # Create few-shot examples for segmentation
            print("\nGenerating few-shot examples for segmentation...")
            segmentation_examples = [
                {
                    "original": "The company's CEO doesn't want to go public until Q4 2025.",
                    "segmented": ["The", "company", "'s", "CEO", "does", "n't", "want", "to", "go", "public", "until", "Q4", "2025", "."],
                    "explanation": "Splits possessive 's and contraction n't, keeps Q4 as a single token"
                },
                {
                    "original": "Apple Inc. reported $100.5 billion in revenue, a 25% increase.",
                    "segmented": ["Apple", "Inc", ".", "reported", "$", "100.5", "billion", "in", "revenue", ",", "a", "25", "%", "increase", "."],
                    "explanation": "Splits Inc., $, and % as separate tokens"
                },
                {
                    "original": "The state-of-the-art AI model can't process e-mail attachments.",
                    "segmented": ["The", "state-of-the-art", "AI", "model", "can", "n't", "process", "e-mail", "attachments", "."],
                    "explanation": "Keeps hyphenated compounds together, splits contraction"
                }
            ]

            segmentation_prompt = """You are an expert in linguistic segmentation according to Universal Dependencies (UD) guidelines.
Your task is to segment the input text according to UD tokenization rules.

Key segmentation rules:
1. Split contractions: don't → do n't, I'm → I 'm, she's → she 's
2. Split possessives: company's → company 's
3. Split punctuation from words: Inc. → Inc .
4. Split symbols from numbers: $100 → $ 100, 25% → 25 %
5. Keep hyphenated compounds together: state-of-the-art, e-mail
6. Each punctuation mark should be its own token

Here are some examples:

"""

            for ex in segmentation_examples:
                segmentation_prompt += f"Original: {ex['original']}\n"
                segmentation_prompt += f"Segmented: {' '.join(ex['segmented'])}\n"
                segmentation_prompt += f"Explanation: {ex['explanation']}\n\n"

            segmentation_prompt += "Please segment the following text according to these rules, returning a list of tokens:"

            # Save the segmentation prompt
            with open(f"segmentation_prompt_{timestamp}.txt", "w") as f:
                f.write(segmentation_prompt)
            print(f"Segmentation prompt saved to segmentation_prompt_{timestamp}.txt")

    except Exception as e:
        print(f"Error in error analysis: {e}")