# See https://docs.x.ai/docs/guides/structured-outputs
# --- Imports ---
import os
import json
import datetime
from openai import OpenAI

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
import time
from collections import defaultdict
import re

# Limit: 5 requests per second
# Context: 131,072 tokens
# Text input: $0.30 per million
# Text output: $0.50 per million
model = 'grok-3-mini'


# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"       # adjective
    ADP = "ADP"       # adposition
    ADV = "ADV"       # adverb
    AUX = "AUX"       # auxiliary
    CCONJ = "CCONJ"   # coordinating conjunction
    DET = "DET"       # determiner
    INTJ = "INTJ"     # interjection
    NOUN = "NOUN"     # noun
    NUM = "NUM"       # numeral
    PART = "PART"     # particle
    PRON = "PRON"     # pronoun
    PROPN = "PROPN"   # proper noun
    PUNCT = "PUNCT"   # punctuation
    SCONJ = "SCONJ"   # subordinating conjunction
    SYM = "SYM"       # symbol
    VERB = "VERB"     # verb
    X = "X"           # other

class TokenPOS(BaseModel):
    """Represents a token with its part-of-speech tag."""
    token: str = Field(description="The word or token.")
    pos_tag: UDPosTag = Field(description="The Universal Dependencies POS tag for this token.")

class SentencePOS(BaseModel):
    """Represents a sentence with its tokens and their POS tags."""
    tokens: List[TokenPOS] = Field(description="List of tokens with their POS tags.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")



# --- Configure the Grok API ---
# Get a key https://console.x.ai/team 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GROK_API_KEY = userdata.get('GROK_API_KEY')
# genai.configure(api_key=GROK_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Load API key from config file
    with open("grok_key.ini", "r") as f:
        for line in f:
            if line.startswith("GROK_API_KEY="):
                api_key = line.strip().split("=")[1]

    # If not found in file, check environment variable
    if not api_key:
        api_key = os.environ.get("GROK_API_KEY")

    # Check if we have a valid key
    if not api_key or api_key == "YOUR_API_KEY":
        print("⚠️ Warning: API key not found. Using placeholder.")
        print("   Please set the GROK_API_KEY in the grok_key.ini file.")
        api_key = "YOUR_API_KEY"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )
    print("✓ Successfully configured Grok API with key")

except Exception as e:
    print(f"Error configuring API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input list of sentences using the Grok API and
    returns the result structured according to the TaggedSentences Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt with clearer examples and guidance
    prompt = f"""You are a linguist specializing in Universal Dependencies (UD) part-of-speech tagging.
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
   - "that" can be DET, PRON, or SCONJ depending on usage
   - "to" as infinitive marker is PART, but as a preposition is ADP
   - Words like "up", "out", "down" can be ADV or ADP based on context
   - "like" can be VERB, SCONJ, or ADP depending on context
   - "as" is typically SCONJ or ADP depending on usage

2. Remember that auxiliary verbs (be, have, do, will, shall, would, can, could, may, might, must, should) are tagged as AUX, not VERB.

3. For proper nouns (PROPN), include names of specific entities, organizations, locations, etc.

4. For ambiguous cases, consider the syntactic function in the sentence.

5. Special tokenization rules:
   - Currency symbols like $ should be separate tokens tagged as SYM
   - Percentages like 25% should be split into the number (25) and the symbol (%)
   - Punctuation marks should be their own tokens, even when part of abbreviations (e.g. "Inc." → "Inc" + ".")

First, split the text into sentences. Then, for each sentence, tokenize it (preserving the original tokenization) and assign the most appropriate UD POS tag to each token.

Return the result in the specified JSON format with each sentence as a list of tokens and their POS tags.
"""

    try:
        completion = client.beta.chat.completions.parse(
            model="grok-3-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_to_tag},
            ],
            response_format=TaggedSentences,
        )
        # print(completion)
        res = completion.choices[0].message.parsed
        return res
    except Exception as e:
        print(f"Error during Grok API call: {e}")
        return None

def generate_few_shot_prompt(examples: List[Dict[str, List[Tuple[str, str]]]], text: str) -> str:
    """
    Generate a prompt with few-shot examples for better POS tagging.
    
    Args:
        examples: List of dictionaries containing example sentences with their gold POS tags
        text: The text to be tagged
    
    Returns:
        A prompt string with few-shot examples
    """
    few_shot_prompt = "Here are some example sentences with their correct POS tags:\n\n"
    
    for example in examples:
        sentence = ""
        tags = ""
        for token, tag in example["tokens"]:
            sentence += f"{token} "
            tags += f"{tag} "
        few_shot_prompt += f"Sentence: {sentence.strip()}\n"
        few_shot_prompt += f"Tags: {tags.strip()}\n\n"
    
    few_shot_prompt += f"Now, please tag the following text using the same format:\n{text}"
    return few_shot_prompt

def batch_tag_sentences(sentences: List[str], batch_size: int = 10) -> List[TaggedSentences]:
    """
    Process sentences in batches to respect API rate limits.

    Args:
        sentences: List of sentences to tag
        batch_size: Number of sentences to process in each batch (default 10 for Grok)

    Returns:
        List of TaggedSentences objects with tagged sentences
    """
    results = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size  # Ceiling division

    print(f"Processing {len(sentences)} sentences in {total_batches} batches (batch size: {batch_size})")

    for i in range(0, len(sentences), batch_size):
        batch_num = i // batch_size + 1
        batch = sentences[i:i + batch_size]

        # Join sentences with a clear separator to help the model identify sentence boundaries
        batch_text = " [SENTENCE] ".join(batch)

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} sentences)")

        try:
            batch_result = tag_sentences_ud(batch_text)
            if batch_result:
                results.append(batch_result)
                print(f"✓ Successfully processed batch {batch_num}")
            else:
                print(f"✗ Failed to process batch {batch_num}")

            # Respect rate limits - wait between batches (Grok: 5 requests per second)
            if i + batch_size < len(sentences):
                sleep_time = 0.25  # Conservative rate limiting (4 requests per second)
                print(f"Waiting {sleep_time}s before next batch...")
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Error processing batch {batch_num}/{total_batches}: {e}")

            # If we hit a rate limit, wait longer before retrying
            if "rate limit" in str(e).lower():
                print("Rate limit exceeded. Waiting 5 seconds before continuing...")
                time.sleep(5)

                # Try again with a smaller batch
                try:
                    half_size = len(batch) // 2
                    if half_size > 0:
                        print(f"Retrying with smaller batch size ({half_size})")
                        for j in range(0, len(batch), half_size):
                            small_batch = batch[j:j + half_size]
                            small_batch_text = " [SENTENCE] ".join(small_batch)
                            small_batch_result = tag_sentences_ud(small_batch_text)
                            if small_batch_result:
                                results.append(small_batch_result)
                            time.sleep(1)  # Be extra cautious with rate limits
                except Exception as inner_e:
                    print(f"Failed to process reduced batch: {inner_e}")

    # Flatten results if needed (combine all sentence POS data into a single structure)
    flattened_results = []
    for result in results:
        if result and hasattr(result, 'sentences'):
            flattened_results.extend(result.sentences)

    return flattened_results

def analyze_tagging_errors(predicted: TaggedSentences, ground_truth: TaggedSentences) -> Dict:
    """
    Compare predictions with ground truth and collect error statistics.

    Args:
        predicted: Predicted POS tags from LLM
        ground_truth: Ground truth POS tags from gold data

    Returns:
        Dictionary containing detailed error statistics
    """
    error_stats = {
        "total_tokens": 0,
        "correct_tokens": 0,
        "errors_by_tag": defaultdict(int),
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "sentence_error_counts": defaultdict(int),  # Count sentences with N errors
        "error_tokens": [],  # Store specific error tokens for analysis
        "difficult_tokens": defaultdict(int)  # Frequency of tokens that are often misclassified
    }

    for sent_idx, (pred_sent, true_sent) in enumerate(zip(predicted.sentences, ground_truth.sentences)):
        sent_errors = 0

        for token_idx, (pred_token, true_token) in enumerate(zip(pred_sent.tokens, true_sent.tokens)):
            error_stats["total_tokens"] += 1

            if pred_token.pos_tag == true_token.pos_tag:
                error_stats["correct_tokens"] += 1
            else:
                # Token-level error tracking
                sent_errors += 1
                error_stats["errors_by_tag"][true_token.pos_tag] += 1
                error_stats["confusion_matrix"][true_token.pos_tag][pred_token.pos_tag] += 1
                error_stats["difficult_tokens"][true_token.token] += 1

                # Store detailed error information for analysis
                error_stats["error_tokens"].append({
                    "sentence_idx": sent_idx,
                    "token_idx": token_idx,
                    "token": true_token.token,
                    "gold_tag": true_token.pos_tag,
                    "pred_tag": pred_token.pos_tag
                })

        # Sentence-level error tracking
        if sent_errors > 0:
            error_stats["sentence_error_counts"][sent_errors] += 1

    # Calculate accuracy metrics
    error_stats["accuracy"] = error_stats["correct_tokens"] / error_stats["total_tokens"] if error_stats["total_tokens"] > 0 else 0
    error_stats["error_rate"] = 1 - error_stats["accuracy"]

    # Calculate per-tag metrics
    tag_metrics = {}
    all_tags = set(list(error_stats["errors_by_tag"].keys()) +
                  [tag for tag_dict in error_stats["confusion_matrix"].values() for tag in tag_dict.keys()])

    for tag in all_tags:
        # True positives, false positives, false negatives
        tp = sum(error_stats["confusion_matrix"][tag][tag] for tag in error_stats["confusion_matrix"] if tag in error_stats["confusion_matrix"][tag])
        fp = sum(error_stats["confusion_matrix"][gold][tag] for gold in error_stats["confusion_matrix"] for pred in error_stats["confusion_matrix"][gold] if gold != tag)
        fn = sum(error_stats["confusion_matrix"][tag][pred] for pred in error_stats["confusion_matrix"][tag] if pred != tag)

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tag_metrics[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "errors": error_stats["errors_by_tag"].get(tag, 0)
        }

    error_stats["tag_metrics"] = tag_metrics

    # Calculate overall F1 score (macro average)
    error_stats["macro_f1"] = sum(tag["f1"] for tag in tag_metrics.values()) / len(tag_metrics) if tag_metrics else 0

    return dict(error_stats)


def compare_taggers(llm_predictions: TaggedSentences, lr_predictions: TaggedSentences,
                   ground_truth: TaggedSentences) -> Dict:
    """
    Compare performance between LLM and LR taggers against ground truth.

    Args:
        llm_predictions: Predictions from the LLM-based tagger
        lr_predictions: Predictions from the Logistic Regression tagger
        ground_truth: Ground truth data

    Returns:
        Dictionary with comparative analysis
    """
    result = {
        "llm_analysis": analyze_tagging_errors(llm_predictions, ground_truth),
        "lr_analysis": analyze_tagging_errors(lr_predictions, ground_truth),
        "comparative": {}
    }

    # Extract token-level differences
    llm_errors = set((e["token"], e["gold_tag"], e["pred_tag"]) for e in result["llm_analysis"]["error_tokens"])
    lr_errors = set((e["token"], e["gold_tag"], e["pred_tag"]) for e in result["lr_analysis"]["error_tokens"])

    # Find errors fixed by LLM (in LR but not in LLM)
    fixed_by_llm = lr_errors - llm_errors

    # Find new errors introduced by LLM (in LLM but not in LR)
    new_in_llm = llm_errors - lr_errors

    # Find common errors (in both LLM and LR)
    common_errors = llm_errors.intersection(lr_errors)

    result["comparative"] = {
        "fixed_by_llm": len(fixed_by_llm),
        "fixed_by_llm_examples": list(fixed_by_llm)[:20],  # Show just 20 examples
        "new_in_llm": len(new_in_llm),
        "new_in_llm_examples": list(new_in_llm)[:20],
        "common_errors": len(common_errors),
        "accuracy_improvement": result["llm_analysis"]["accuracy"] - result["lr_analysis"]["accuracy"],
        "f1_improvement": result["llm_analysis"]["macro_f1"] - result["lr_analysis"]["macro_f1"]
    }

    # Sentence-level comparison
    result["comparative"]["sentence_improvements"] = {}
    for error_count in range(1, 11):  # Track improvements for sentences with 1-10 errors
        llm_count = result["llm_analysis"]["sentence_error_counts"].get(error_count, 0)
        lr_count = result["lr_analysis"]["sentence_error_counts"].get(error_count, 0)
        result["comparative"]["sentence_improvements"][error_count] = {
            "llm_count": llm_count,
            "lr_count": lr_count,
            "difference": lr_count - llm_count
        }

    return result

def test_on_hard_sentences(hard_sentences, lr_tagger, save_results=True):
    """
    Test the LLM tagger on sentences that are difficult for the LR tagger.

    Args:
        hard_sentences: List of difficult sentences from the LR tagger evaluation
        lr_tagger: The LR tagger function (for comparison)
        save_results: Whether to save results to a file for later analysis

    Returns:
        Comprehensive comparison between LLM and LR taggers on hard sentences
    """
    print(f"Testing on {len(hard_sentences)} hard sentences...")

    # Extract original sentences and gold standard tags
    original_sentences = []
    gold_standard = []
    lr_predictions = []

    for sentence, error_count, lr_pred in hard_sentences:
        # Convert to format required by taggers
        original_text = " ".join([token for token, _ in sentence])
        original_sentences.append(original_text)

        # Create gold standard tags in the right format
        gold_tokens = []
        for token, tag in sentence:
            gold_tokens.append(TokenPOS(token=token, pos_tag=tag))
        gold_standard.append(SentencePOS(tokens=gold_tokens))

        # Store LR predictions in the right format
        lr_tokens = []
        for (token, _), tag in zip([pair for pair in sentence], lr_pred):
            lr_tokens.append(TokenPOS(token=token, pos_tag=tag))
        lr_predictions.append(SentencePOS(tokens=lr_tokens))

    # Create TaggedSentences objects
    gold_tagged = TaggedSentences(sentences=gold_standard)
    lr_tagged = TaggedSentences(sentences=lr_predictions)

    # Process with LLM in batches (focus on sentences with 1-3 errors)
    sentences_to_process = [s for s, e, _ in hard_sentences if 1 <= e <= 3]
    print(f"Processing subset of {len(sentences_to_process)} sentences with 1-3 errors...")

    # Extract raw text for LLM processing
    texts_to_process = [" ".join([token for token, _ in s]) for s in sentences_to_process]

    # Process in batches
    batch_size = 5  # Adjust based on API limits
    llm_results = batch_tag_sentences(texts_to_process, batch_size)

    print(f"Completed LLM tagging with {len(llm_results)} sentences processed")

    # Convert to expected format if needed
    if not isinstance(llm_results, TaggedSentences):
        llm_tagged = TaggedSentences(sentences=llm_results)
    else:
        llm_tagged = llm_results

    # Compare taggers
    comparison = compare_taggers(llm_tagged, lr_tagged, gold_tagged)

    # Save results if requested
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"tagger_comparison_{timestamp}.json", "w") as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            serializable_comparison = json.dumps(comparison, default=lambda o: dict(o) if isinstance(o, defaultdict) else o, indent=2)
            f.write(serializable_comparison)
            print(f"Results saved to tagger_comparison_{timestamp}.json")

    return comparison


def segment_text(text: str) -> List[str]:
    """
    Segment text according to Universal Dependencies guidelines.

    This function implements basic UD segmentation rules:
    1. Splits contractions (e.g., don't -> do n't)
    2. Handles special punctuation
    3. Preserves multi-word tokens when needed

    Args:
        text: Input text to segment

    Returns:
        List of segmented tokens
    """
    # Pre-process contractions
    text = re.sub(r"n't", " n't", text)  # don't -> do n't
    text = re.sub(r"'ll", " 'll", text)  # they'll -> they 'll
    text = re.sub(r"'ve", " 've", text)  # I've -> I 've
    text = re.sub(r"'re", " 're", text)  # they're -> they 're
    text = re.sub(r"'m", " 'm", text)    # I'm -> I 'm

    # Handle special punctuation
    text = re.sub(r"([.,!?()])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Split into tokens
    tokens = text.strip().split()

    # Post-process special cases
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Handle multi-word expressions
        if i + 2 < len(tokens) and token.lower() in ["in", "of", "at"] and tokens[i+1].lower() == "the":
            # Keep some common multi-word expressions together
            processed_tokens.append(f"{token} {tokens[i+1]} {tokens[i+2]}")
            i += 3
        else:
            processed_tokens.append(token)
            i += 1

    return processed_tokens

def segment_and_tag(text: str) -> TaggedSentences:
    """
    Perform UD-compliant word segmentation before tagging.
    
    Args:
        text: Input text to segment and tag
        
    Returns:
        TaggedSentences object with properly segmented and tagged tokens
    """
    # First segment the text
    tokens = segment_text(text)
    
    # Join tokens back with spaces for the LLM
    segmented_text = " ".join(tokens)
    
    # Get POS tags
    return tag_sentences_ud(segmented_text)

# --- Example Usage ---
if __name__ == "__main__":
    # Test the Grok POS tagger
    print("Grok POS Tagger Test")
    print("===================")

    # Example sentences with different difficulties
    test_sentences = [
        # Simple sentence
        "The quick brown fox jumps over the lazy dog.",

        # Sentence with ambiguous words (that, like, as)
        "I like the book that you recommended, as it was very engaging.",

        # Sentence with technical terms
        "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?",

        # Sentence with challenging prepositions/particles
        "She looked up the answer to the question that had been bothering her for days.",

        # Sentence with proper nouns and symbols
        "Apple Inc. announced its Q3 earnings of $100.5 million, surprising investors with a 25% increase.",
    ]

    print(f"Testing {len(test_sentences)} sentences...\n")

    # Process each sentence individually for demonstration
    for i, example_text in enumerate(test_sentences):
        print(f"\nSentence {i+1}: \"{example_text}\"")

        try:
            # Tag the sentence
            tagged_result = tag_sentences_ud(example_text)

            if tagged_result and isinstance(tagged_result, TaggedSentences):
                print("\n--- Tagging Results ---")
                for s in tagged_result.sentences:
                    for token_pos in s.tokens:
                        token = token_pos.token
                        tag = token_pos.pos_tag
                        # Handle potential None for pos_tag if model couldn't assign one
                        ctag = tag if tag is not None else "UNKNOWN"
                        print(f"Token: {token:<15} {str(ctag)}")
                    print("----------------------")
            else:
                print("\nFailed to get valid POS tagging results.")
        except Exception as e:
            print(f"Error processing sentence: {e}")

    print("\nGrok POS Tagger test complete!")