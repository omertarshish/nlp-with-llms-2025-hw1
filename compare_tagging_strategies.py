#!/usr/bin/env python
"""
Comparison of Pipeline vs. Joint Tagging Strategies

This script compares two approaches to POS tagging with segmentation:
1. Pipeline approach: segmentation and tagging in separate steps
2. Joint approach: segmentation and tagging in a single step

It evaluates both approaches on a set of "hard sentences" and reports metrics.
"""

import os
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

# Import our modules
from example_selection import select_examples_for_segmentation
from ud_pos_tagger_with_examples import tag_sentences_ud_with_examples

# Define data models for structured output
class TaggedToken(BaseModel):
    token: str
    pos: str

class TaggedSentence(BaseModel):
    sentence: str
    tokens: List[TaggedToken]

class TaggedSentences(BaseModel):
    sentences: List[TaggedSentence]

class SegmentedSentence(BaseModel):
    original: str
    segments: List[str]

class SegmentedSentences(BaseModel):
    sentences: List[SegmentedSentence]

# Initialize API client
try:
    # Try Grok API first
    import anthropic
    USE_GROK = True
    try:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            with open("grok_key.ini", "r") as f:
                for line in f:
                    if line.startswith("GROK_API_KEY="):
                        api_key = line.split("=")[1].strip()
                        break
        
        client = anthropic.Anthropic(api_key=api_key)
        print("✓ Successfully configured Grok API with key")
    except Exception as e:
        print(f"Error configuring Grok API: {e}")
        USE_GROK = False
except ImportError:
    USE_GROK = False

if not USE_GROK:
    try:
        # Try Gemini API
        from google import generativeai as genai
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                with open("gemini_key.ini", "r") as f:
                    for line in f:
                        if line.startswith("GOOGLE_API_KEY="):
                            api_key = line.split("=")[1].strip()
                            break
            
            genai.configure(api_key=api_key)
            print("✓ Successfully configured Gemini API with key")
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            sys.exit(1)
    except ImportError:
        print("Error: Neither Grok nor Gemini API libraries are available.")
        print("Please install one of the following:")
        print("  pip install anthropic  # For Grok")
        print("  pip install google-generativeai  # For Gemini")
        sys.exit(1)

# Pipeline Approach: Stage 1 - Segmentation Only
def segment_sentence(text: str) -> Optional[SegmentedSentence]:
    """
    Segment a sentence using example-based approach.
    
    Args:
        text: The text to segment
        
    Returns:
        SegmentedSentence object or None if there was an error
    """
    # Select relevant examples for segmentation
    relevant_examples = select_examples_for_segmentation(text, top_n=3)
    
    # Format examples for the prompt
    examples_text = "\n".join([
        f"Example {i+1}: \"{example['text']}\"\n"
        f"Segmentation: {example['segmented']}\n"
        for i, example in enumerate(relevant_examples)
    ])
    
    # Construct the prompt for segmentation only
    prompt = f"""You are a linguistic expert specializing in Universal Dependencies (UD) word segmentation.
Your task is to segment the input text into tokens according to UD guidelines, without assigning POS tags.

I've selected the following examples that contain segmentation patterns similar to your input text. 
Use these as reference for how to correctly segment similar constructions:

{examples_text}

Segmentation Guidelines:
1. Most punctuation marks should be separate tokens
2. Contractions should be split (e.g., "don't" → "do" + "n't")
3. Possessive 's should be split from its noun (e.g., "John's" → "John" + "'s")
4. Hyphenated compounds should usually stay as one token
5. Multi-word proper nouns (e.g., "New York") should be separate tokens
6. Numbers with decimal points or commas stay as one token (e.g., "3.14")
7. Currency or percentage symbols are separate from the number (e.g., "$" + "100")
8. Email addresses and URLs should be kept as single tokens

Input text to segment:
"{text}"

IMPORTANT: Only segment the text, do not assign POS tags.

Your response MUST be valid JSON with the following format:
{{
  "original": "original sentence text",
  "segments": ["token1", "token2", ...]
}}
"""
    
    # Call the appropriate API
    try:
        if USE_GROK:
            # Grok API call
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are a linguistics expert who specializes in Universal Dependencies (UD) word segmentation.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result_text = response.content[0].text
            
            # Parse the result
            try:
                result_json = json.loads(result_text)
                return SegmentedSentence(**result_json)
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {result_text[:100]}...")
                return None
                
        else:
            # Gemini API call
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            
            # Parse the result
            try:
                result_json = json.loads(response.text)
                return SegmentedSentence(**result_json)
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {response.text[:100]}...")
                return None
                
    except Exception as e:
        print(f"API call error: {e}")
        return None

# Pipeline Approach: Stage 2 - Tagging Only
def tag_segmented_sentence(segments: List[str]) -> Optional[TaggedSentence]:
    """
    Tag a pre-segmented sentence.
    
    Args:
        segments: List of segmented tokens
        
    Returns:
        TaggedSentence object or None if there was an error
    """
    # Construct the prompt for tagging only
    prompt = f"""You are a linguistic expert specializing in Universal Dependencies (UD) part-of-speech tagging.
Your task is to assign appropriate UD POS tags to the pre-segmented tokens.

Here are the Universal Dependencies POS tags:
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

Pre-segmented tokens to tag:
{segments}

IMPORTANT: Assign an appropriate POS tag to each token based on UD guidelines.

Your response MUST be valid JSON with the following format:
{{
  "sentence": "original joined sentence",
  "tokens": [
    {{ "token": "token1", "pos": "POSTAG" }},
    {{ "token": "token2", "pos": "POSTAG" }},
    ...
  ]
}}
"""
    
    # Call the appropriate API
    try:
        if USE_GROK:
            # Grok API call
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are a linguistics expert who specializes in Universal Dependencies (UD) POS tagging.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            result_text = response.content[0].text
            
            # Parse the result
            try:
                result_json = json.loads(result_text)
                return TaggedSentence(**result_json)
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {result_text[:100]}...")
                return None
                
        else:
            # Gemini API call
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            
            # Parse the result
            try:
                result_json = json.loads(response.text)
                return TaggedSentence(**result_json)
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {response.text[:100]}...")
                return None
                
    except Exception as e:
        print(f"API call error: {e}")
        return None

# Pipeline approach (complete)
def pipeline_approach(text: str) -> Optional[TaggedSentence]:
    """
    Process a sentence using the pipeline approach:
    1. Segment
    2. Tag
    
    Args:
        text: The original text to process
        
    Returns:
        TaggedSentence object or None if there was an error
    """
    print(f"Pipeline: Processing \"{text}\"")
    
    # Step 1: Segmentation
    start_time = time.time()
    segmented = segment_sentence(text)
    seg_time = time.time() - start_time
    
    if not segmented:
        print("  Pipeline: Segmentation failed")
        return None
    
    print(f"  Pipeline: Segmentation completed in {seg_time:.2f}s")
    print(f"  Pipeline: Segments: {segmented.segments}")
    
    # Step 2: Tagging
    start_time = time.time()
    tagged = tag_segmented_sentence(segmented.segments)
    tag_time = time.time() - start_time
    
    if not tagged:
        print("  Pipeline: Tagging failed")
        return None
    
    print(f"  Pipeline: Tagging completed in {tag_time:.2f}s")
    print(f"  Pipeline: Total time: {seg_time + tag_time:.2f}s")
    
    return tagged

# Joint approach
def joint_approach(text: str) -> Optional[TaggedSentence]:
    """
    Process a sentence using the joint approach:
    Segment and tag in a single step
    
    Args:
        text: The original text to process
        
    Returns:
        TaggedSentence object or None if there was an error
    """
    print(f"Joint: Processing \"{text}\"")
    
    # Use the existing joint implementation
    start_time = time.time()
    result = tag_sentences_ud_with_examples(text)
    joint_time = time.time() - start_time
    
    if not result or not result.sentences:
        print("  Joint: Processing failed")
        return None
    
    tagged = result.sentences[0]
    print(f"  Joint: Processing completed in {joint_time:.2f}s")
    print(f"  Joint: Produced {len(tagged.tokens)} tokens")
    
    return tagged

# Compare results with gold standard
def compare_with_gold(pipeline_result: Optional[TaggedSentence], 
                     joint_result: Optional[TaggedSentence],
                     gold_tokens: List[str],
                     gold_tags: List[str]) -> Dict[str, Any]:
    """
    Compare pipeline and joint approaches with gold standard.
    
    Args:
        pipeline_result: Result from pipeline approach
        joint_result: Result from joint approach
        gold_tokens: Gold standard tokens
        gold_tags: Gold standard tags
        
    Returns:
        Dictionary with comparison metrics
    """
    results = {
        "pipeline": {
            "segmentation_accuracy": 0.0,
            "tagging_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "token_count": 0
        },
        "joint": {
            "segmentation_accuracy": 0.0,
            "tagging_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "token_count": 0
        }
    }
    
    # Check pipeline results
    if pipeline_result:
        pipeline_tokens = [t.token for t in pipeline_result.tokens]
        pipeline_tags = [t.pos for t in pipeline_result.tokens]
        
        # Simple exact match for tokens (crude approximation)
        results["pipeline"]["token_count"] = len(pipeline_tokens)
        
        # For tokens that match in position, compute segmentation accuracy
        min_len = min(len(pipeline_tokens), len(gold_tokens))
        seg_correct = sum(1 for i in range(min_len) if pipeline_tokens[i] == gold_tokens[i])
        results["pipeline"]["segmentation_accuracy"] = seg_correct / len(gold_tokens) if gold_tokens else 0
        
        # For tokens that match in segmentation, compute tagging accuracy
        tag_correct = sum(1 for i in range(min_len) 
                          if pipeline_tokens[i] == gold_tokens[i] and pipeline_tags[i] == gold_tags[i])
        results["pipeline"]["tagging_accuracy"] = tag_correct / seg_correct if seg_correct > 0 else 0
        
        # Overall accuracy: tokens must be correctly segmented AND tagged
        results["pipeline"]["overall_accuracy"] = tag_correct / len(gold_tokens) if gold_tokens else 0
    
    # Check joint results
    if joint_result:
        joint_tokens = [t.token for t in joint_result.tokens]
        joint_tags = [t.pos for t in joint_result.tokens]
        
        results["joint"]["token_count"] = len(joint_tokens)
        
        # For tokens that match in position, compute segmentation accuracy
        min_len = min(len(joint_tokens), len(gold_tokens))
        seg_correct = sum(1 for i in range(min_len) if joint_tokens[i] == gold_tokens[i])
        results["joint"]["segmentation_accuracy"] = seg_correct / len(gold_tokens) if gold_tokens else 0
        
        # For tokens that match in segmentation, compute tagging accuracy
        tag_correct = sum(1 for i in range(min_len) 
                         if joint_tokens[i] == gold_tokens[i] and joint_tags[i] == gold_tags[i])
        results["joint"]["tagging_accuracy"] = tag_correct / seg_correct if seg_correct > 0 else 0
        
        # Overall accuracy: tokens must be correctly segmented AND tagged
        results["joint"]["overall_accuracy"] = tag_correct / len(gold_tokens) if gold_tokens else 0
    
    return results

# Evaluate and compare approaches on a set of hard sentences
def evaluate_approaches(sentences_with_gold: List[Tuple[str, List[str], List[str]]]):
    """
    Evaluate pipeline and joint approaches on a set of sentences.
    
    Args:
        sentences_with_gold: List of (sentence, gold_tokens, gold_tags) tuples
    """
    results = []
    
    for idx, (sentence, gold_tokens, gold_tags) in enumerate(sentences_with_gold):
        print(f"\nProcessing sentence {idx+1}/{len(sentences_with_gold)}")
        
        # Run both approaches
        pipeline_result = pipeline_approach(sentence)
        
        # Wait to respect API rate limits
        time.sleep(5)
        
        joint_result = joint_approach(sentence)
        
        # Compare results
        comparison = compare_with_gold(pipeline_result, joint_result, gold_tokens, gold_tags)
        
        # Add to results
        results.append({
            "sentence": sentence,
            "gold_tokens": gold_tokens,
            "gold_tags": gold_tags,
            "pipeline_result": pipeline_result.dict() if pipeline_result else None,
            "joint_result": joint_result.dict() if joint_result else None,
            "comparison": comparison
        })
        
        # Print comparison
        print("\nComparison:")
        print(f"  Pipeline segmentation accuracy: {comparison['pipeline']['segmentation_accuracy']:.3f}")
        print(f"  Pipeline tagging accuracy: {comparison['pipeline']['tagging_accuracy']:.3f}")
        print(f"  Pipeline overall accuracy: {comparison['pipeline']['overall_accuracy']:.3f}")
        print(f"  Joint segmentation accuracy: {comparison['joint']['segmentation_accuracy']:.3f}")
        print(f"  Joint tagging accuracy: {comparison['joint']['tagging_accuracy']:.3f}")
        print(f"  Joint overall accuracy: {comparison['joint']['overall_accuracy']:.3f}")
        
        # Wait to respect API rate limits
        time.sleep(10)
    
    # Save results
    with open("tagging_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Compute aggregate metrics
    pipeline_seg_acc = np.mean([r["comparison"]["pipeline"]["segmentation_accuracy"] for r in results])
    pipeline_tag_acc = np.mean([r["comparison"]["pipeline"]["tagging_accuracy"] for r in results])
    pipeline_overall_acc = np.mean([r["comparison"]["pipeline"]["overall_accuracy"] for r in results])
    
    joint_seg_acc = np.mean([r["comparison"]["joint"]["segmentation_accuracy"] for r in results])
    joint_tag_acc = np.mean([r["comparison"]["joint"]["tagging_accuracy"] for r in results])
    joint_overall_acc = np.mean([r["comparison"]["joint"]["overall_accuracy"] for r in results])
    
    print("\n===== Aggregate Results =====")
    print(f"Pipeline segmentation accuracy: {pipeline_seg_acc:.3f}")
    print(f"Pipeline tagging accuracy: {pipeline_tag_acc:.3f}")
    print(f"Pipeline overall accuracy: {pipeline_overall_acc:.3f}")
    print(f"Joint segmentation accuracy: {joint_seg_acc:.3f}")
    print(f"Joint tagging accuracy: {joint_tag_acc:.3f}")
    print(f"Joint overall accuracy: {joint_overall_acc:.3f}")
    
    # Plot comparative metrics
    plt.figure(figsize=(12, 6))
    
    metrics = ["Segmentation", "Tagging", "Overall"]
    pipeline_values = [pipeline_seg_acc, pipeline_tag_acc, pipeline_overall_acc]
    joint_values = [joint_seg_acc, joint_tag_acc, joint_overall_acc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, pipeline_values, width, label='Pipeline')
    plt.bar(x + width/2, joint_values, width, label='Joint')
    
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Pipeline vs. Joint Approaches')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("tagging_comparison.png")
    print("Saved visualization to tagging_comparison.png")
    
    # Write a detailed analysis report
    write_analysis_report(results, {
        "pipeline_seg_acc": pipeline_seg_acc,
        "pipeline_tag_acc": pipeline_tag_acc,
        "pipeline_overall_acc": pipeline_overall_acc,
        "joint_seg_acc": joint_seg_acc,
        "joint_tag_acc": joint_tag_acc,
        "joint_overall_acc": joint_overall_acc
    })

def write_analysis_report(results, aggregate_metrics):
    """
    Write a detailed analysis report of the comparison.
    
    Args:
        results: List of result dictionaries
        aggregate_metrics: Dictionary of aggregate metrics
    """
    with open("tagging_comparison_report.md", "w") as f:
        f.write("# Pipeline vs. Joint Tagging Approaches: Comparative Analysis\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write("This report compares two approaches to POS tagging with segmentation:\n\n")
        f.write("1. **Pipeline Approach**: Segmentation and tagging in separate steps\n")
        f.write("2. **Joint Approach**: Segmentation and tagging in a single step\n\n")
        
        # Aggregate metrics
        f.write("## Aggregate Metrics\n\n")
        f.write("| Metric | Pipeline | Joint | Difference |\n")
        f.write("|--------|----------|-------|------------|\n")
        
        # Calculate differences (joint - pipeline)
        seg_diff = aggregate_metrics["joint_seg_acc"] - aggregate_metrics["pipeline_seg_acc"]
        tag_diff = aggregate_metrics["joint_tag_acc"] - aggregate_metrics["pipeline_tag_acc"]
        overall_diff = aggregate_metrics["joint_overall_acc"] - aggregate_metrics["pipeline_overall_acc"]
        
        f.write(f"| Segmentation Accuracy | {aggregate_metrics['pipeline_seg_acc']:.3f} | {aggregate_metrics['joint_seg_acc']:.3f} | {seg_diff:+.3f} |\n")
        f.write(f"| Tagging Accuracy | {aggregate_metrics['pipeline_tag_acc']:.3f} | {aggregate_metrics['joint_tag_acc']:.3f} | {tag_diff:+.3f} |\n")
        f.write(f"| Overall Accuracy | {aggregate_metrics['pipeline_overall_acc']:.3f} | {aggregate_metrics['joint_overall_acc']:.3f} | {overall_diff:+.3f} |\n\n")
        
        # Observations about general patterns
        f.write("## General Observations\n\n")
        
        # Decide which approach is generally better based on overall accuracy
        better_approach = "joint" if overall_diff > 0 else "pipeline" if overall_diff < 0 else "neither"
        
        if better_approach == "joint":
            f.write("The **joint approach** generally performed better overall, with particular strengths in:\n\n")
        elif better_approach == "pipeline":
            f.write("The **pipeline approach** generally performed better overall, with particular strengths in:\n\n")
        else:
            f.write("Neither approach consistently outperformed the other, with trade-offs in different areas:\n\n")
        
        # Highlight strengths of the joint approach
        if seg_diff > 0:
            f.write("- The joint approach achieved better segmentation accuracy\n")
        
        if tag_diff > 0:
            f.write("- The joint approach achieved better tagging accuracy on correctly segmented tokens\n")
        
        # Highlight strengths of the pipeline approach
        if seg_diff < 0:
            f.write("- The pipeline approach achieved better segmentation accuracy\n")
        
        if tag_diff < 0:
            f.write("- The pipeline approach achieved better tagging accuracy on correctly segmented tokens\n")
        
        f.write("\n")
        
        # Sentence-level analysis
        f.write("## Sentence-Level Analysis\n\n")
        
        # Find sentences where approaches differed significantly
        significant_diff_threshold = 0.2
        significant_diffs = []
        
        for result in results:
            seg_diff = (result["comparison"]["joint"]["segmentation_accuracy"] - 
                        result["comparison"]["pipeline"]["segmentation_accuracy"])
            
            overall_diff = (result["comparison"]["joint"]["overall_accuracy"] - 
                           result["comparison"]["pipeline"]["overall_accuracy"])
            
            if abs(seg_diff) > significant_diff_threshold or abs(overall_diff) > significant_diff_threshold:
                significant_diffs.append({
                    "sentence": result["sentence"],
                    "seg_diff": seg_diff,
                    "overall_diff": overall_diff,
                    "result": result
                })
        
        # Write about sentences with significant differences
        if significant_diffs:
            f.write("### Sentences with Significant Performance Differences\n\n")
            
            for diff in significant_diffs:
                f.write(f"#### \"{diff['sentence']}\"\n\n")
                
                pipeline_tokens = []
                if diff["result"]["pipeline_result"]:
                    pipeline_tokens = [t["token"] for t in diff["result"]["pipeline_result"]["tokens"]]
                
                joint_tokens = []
                if diff["result"]["joint_result"]:
                    joint_tokens = [t["token"] for t in diff["result"]["joint_result"]["tokens"]]
                
                f.write("**Segmentation Results:**\n\n")
                f.write(f"- Gold tokens: {diff['result']['gold_tokens']}\n")
                f.write(f"- Pipeline tokens: {pipeline_tokens}\n")
                f.write(f"- Joint tokens: {joint_tokens}\n\n")
                
                # Characterize the main differences
                if diff["seg_diff"] > 0:
                    f.write("The joint approach performed significantly better at segmentation, likely because:\n")
                elif diff["seg_diff"] < 0:
                    f.write("The pipeline approach performed significantly better at segmentation, likely because:\n")
                
                # Add likely reasons (would need manual analysis in a real implementation)
                f.write("- The sentence contains complex segmentation patterns\n")
                f.write("- Different example selection may have influenced the results\n")
                f.write("- The interplay between segmentation and tagging decisions affects results\n\n")
                
                f.write("**Metrics:**\n\n")
                f.write(f"- Segmentation accuracy difference: {diff['seg_diff']:+.3f}\n")
                f.write(f"- Overall accuracy difference: {diff['overall_diff']:+.3f}\n\n")
        
        # Analysis of specific cases
        f.write("## Error Analysis\n\n")
        
        # Categorize challenging segmentation patterns
        f.write("### Challenging Segmentation Patterns\n\n")
        
        # List of common challenging patterns (simplified)
        challenges = ["Contractions", "Hyphenated compounds", "Punctuation", "Multi-word expressions", 
                     "Special characters", "URLs/emails", "Possessives"]
        
        for challenge in challenges:
            f.write(f"#### {challenge}\n\n")
            
            # In a real implementation, this would analyze which approach handles each case better
            # Here we provide a placeholder analysis
            f.write(f"Analysis of how pipeline and joint approaches handle {challenge.lower()}:\n\n")
            f.write("- Pipeline approach tends to handle this pattern [better/worse] because...\n")
            f.write("- Joint approach tends to handle this pattern [better/worse] because...\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        if better_approach == "joint":
            f.write("The joint approach generally outperforms the pipeline approach, suggesting that:\n\n")
            f.write("1. Segmentation and tagging decisions benefit from being made together\n")
            f.write("2. The model can leverage mutual information between the two tasks\n")
            f.write("3. Error propagation from segmentation to tagging is minimized\n\n")
        elif better_approach == "pipeline":
            f.write("The pipeline approach generally outperforms the joint approach, suggesting that:\n\n")
            f.write("1. Specializing each model for a single task improves performance\n")
            f.write("2. Segmentation is better handled as a standalone task\n")
            f.write("3. Example selection can be more targeted for each specific task\n\n")
        else:
            f.write("Neither approach consistently outperforms the other, suggesting that:\n\n")
            f.write("1. The optimal approach depends on the specific characteristics of the sentence\n")
            f.write("2. A hybrid approach might be beneficial\n")
            f.write("3. Different segmentation patterns may benefit from different approaches\n\n")
        
        f.write("### Recommendations\n\n")
        
        if better_approach == "joint":
            f.write("Based on these findings, the joint approach is recommended for most cases, with the following considerations:\n\n")
        elif better_approach == "pipeline":
            f.write("Based on these findings, the pipeline approach is recommended for most cases, with the following considerations:\n\n")
        else:
            f.write("Based on these findings, a hybrid approach might be optimal with the following considerations:\n\n")
        
        f.write("1. Example selection is critical for both approaches\n")
        f.write("2. Certain patterns may benefit from specialized handling\n")
        f.write("3. The choice of approach should consider the specific characteristics of the text\n")
    
    print("Saved analysis report to tagging_comparison_report.md")

# Hard sentences with gold standard for testing
HARD_SENTENCES = [
    {
        "text": "I can't believe it's already 5 o'clock!",
        "gold_tokens": ["I", "ca", "n't", "believe", "it", "'s", "already", "5", "o'clock", "!"],
        "gold_tags": ["PRON", "AUX", "PART", "VERB", "PRON", "AUX", "ADV", "NUM", "NOUN", "PUNCT"]
    },
    {
        "text": "The cost-benefit analysis shows we'll save $5,000.",
        "gold_tokens": ["The", "cost-benefit", "analysis", "shows", "we", "'ll", "save", "$", "5,000", "."],
        "gold_tags": ["DET", "NOUN", "NOUN", "VERB", "PRON", "AUX", "VERB", "SYM", "NUM", "PUNCT"]
    },
    {
        "text": "Please email john.doe@example.com for more information.",
        "gold_tokens": ["Please", "email", "john.doe@example.com", "for", "more", "information", "."],
        "gold_tags": ["INTJ", "VERB", "PROPN", "ADP", "ADJ", "NOUN", "PUNCT"]
    },
    {
        "text": "Dr. Smith's paper was published in the Journal of A.I.",
        "gold_tokens": ["Dr.", "Smith", "'s", "paper", "was", "published", "in", "the", "Journal", "of", "A.I.", "."],
        "gold_tags": ["PROPN", "PROPN", "PART", "NOUN", "AUX", "VERB", "ADP", "DET", "PROPN", "ADP", "PROPN", "PUNCT"]
    },
    {
        "text": "Wait - what did you say? That's incredible!",
        "gold_tokens": ["Wait", "-", "what", "did", "you", "say", "?", "That", "'s", "incredible", "!"],
        "gold_tags": ["VERB", "PUNCT", "PRON", "AUX", "PRON", "VERB", "PUNCT", "PRON", "AUX", "ADJ", "PUNCT"]
    }
]

def main():
    """Main function to run the comparison."""
    print("\nComparing Pipeline vs. Joint Tagging Approaches\n")
    
    # Prepare sentences with gold standard
    sentences_with_gold = [(s["text"], s["gold_tokens"], s["gold_tags"]) for s in HARD_SENTENCES]
    
    # Run the evaluation
    evaluate_approaches(sentences_with_gold)

if __name__ == "__main__":
    main()