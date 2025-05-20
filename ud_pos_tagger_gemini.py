# --- Imports ---
import os
import json
import datetime
from google import generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Tuple, Any
from enum import Enum
import time
from collections import defaultdict
import re

#
gemini_model = 'gemini-2.0-flash-lite'

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

# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Load API key from config file
    with open("gemini_key.ini", "r") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                api_key = line.strip().split("=")[1]

    # If not found in file, check environment variable
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    # Check if we have a valid key
    if not api_key or api_key == "YOUR_API_KEY":
        print("⚠️ Warning: API key not found. Using placeholder.")
        print("   Please set the GOOGLE_API_KEY in the gemini_key.ini file.")
        api_key = "YOUR_API_KEY"

    genai.configure(api_key=api_key)
    print("✓ Successfully configured Gemini API with key")

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt with enhanced instructions and examples
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

First, split the text into sentences. Then, for each sentence, tokenize it (preserving the original tokenization) and assign the most appropriate UD POS tag to each token.

Return the result in the specified JSON format with each sentence as a list of tokens and their POS tags.

Text to analyze:
{text_to_tag}
"""

    try:
        # Configure the model
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model)

        # Generate structured response
        response = model.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )

        # Parse the JSON response and convert to TaggedSentences
        if response.text:
            print(f"Response from API: {response.text[:100]}...")

            try:
                # Parse the JSON response
                response_json = json.loads(response.text)
                print(f"Response structure: {type(response_json)}")

                # Handle different response structures
                if isinstance(response_json, dict) and "sentences" in response_json:
                    sentences_data = response_json["sentences"]
                elif isinstance(response_json, list):
                    sentences_data = response_json
                else:
                    print(f"Unexpected response structure")
                    return None

                # Create the sentences list
                sentences_list = []
                for sent_data in sentences_data:
                    print(f"Sentence data: {type(sent_data)}")

                    # Handle the actual response structure from the API (list of tokens)
                    if isinstance(sent_data, list):
                        # Process tokens in the list
                        tokens_list = []
                        for token_data in sent_data:
                            # The API may use either "pos" or "tag" for POS tags
                            token = token_data.get("token", "")
                            pos_tag = (token_data.get("pos_tag") or
                                      token_data.get("pos") or
                                      token_data.get("tag", "X"))
                            tokens_list.append(TokenPOS(token=token, pos_tag=pos_tag))

                        sentences_list.append(SentencePOS(tokens=tokens_list))
                    # Still keep the original logic for dict format
                    elif isinstance(sent_data, dict) and "tokens" in sent_data:
                        tokens_data = sent_data["tokens"]

                        # Process tokens
                        tokens_list = []
                        for token_data in tokens_data:
                            token = token_data.get("token", "")
                            pos_tag = (token_data.get("pos_tag") or
                                      token_data.get("pos") or
                                      token_data.get("tag", "X"))
                            tokens_list.append(TokenPOS(token=token, pos_tag=pos_tag))

                        sentences_list.append(SentencePOS(tokens=tokens_list))

                return TaggedSentences(sentences=sentences_list)
            except Exception as e:
                print(f"JSON parsing error: {e}")
                return None
        else:
            print("Empty response from Gemini API")
            return None
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
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

def batch_tag_sentences(sentences: List[str], batch_size: int = 5) -> List[TaggedSentences]:
    """
    Process sentences in batches to respect API rate limits.

    Args:
        sentences: List of sentences to tag
        batch_size: Number of sentences to process in each batch (default 5 for Gemini free tier)

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

            # Respect rate limits - wait between batches
            # Gemini free tier: 15 RPM (4 seconds between requests to be safe)
            if i + batch_size < len(sentences):
                sleep_time = 4
                print(f"Waiting {sleep_time}s before next batch (Gemini free tier limit)...")
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Error processing batch {batch_num}/{total_batches}: {e}")

            # If we hit a rate limit, wait longer before retrying
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                print("Rate limit or quota exceeded. Waiting 30 seconds before continuing...")
                time.sleep(30)

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
                            time.sleep(5)  # Be extra cautious with rate limits
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
        predicted: Predicted POS tags
        ground_truth: Ground truth POS tags
    
    Returns:
        Dictionary containing error statistics
    """
    error_stats = {
        "total_tokens": 0,
        "correct_tokens": 0,
        "errors_by_tag": defaultdict(int),
        "confusion_matrix": defaultdict(lambda: defaultdict(int))
    }
    
    for pred_sent, true_sent in zip(predicted.sentences, ground_truth.sentences):
        for pred_token, true_token in zip(pred_sent.tokens, true_sent.tokens):
            error_stats["total_tokens"] += 1
            
            if pred_token.pos_tag == true_token.pos_tag:
                error_stats["correct_tokens"] += 1
            else:
                error_stats["errors_by_tag"][true_token.pos_tag] += 1
                error_stats["confusion_matrix"][true_token.pos_tag][pred_token.pos_tag] += 1
    
    error_stats["accuracy"] = error_stats["correct_tokens"] / error_stats["total_tokens"]
    return dict(error_stats)

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
    # Test the Gemini POS tagger
    print("Gemini POS Tagger Test")
    print("======================")

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

            if tagged_result and hasattr(tagged_result, 'sentences'):
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

    print("\nGemini POS Tagger test complete!")