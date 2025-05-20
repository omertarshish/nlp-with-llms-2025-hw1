#!/usr/bin/env python
"""
Enhanced UD POS Tagger with Example Selection

This script enhances the POS tagging process by incorporating:
1. Smart example selection for word segmentation
2. Improved prompting with relevant examples
3. Proper handling of complex segmentation cases
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import time

# Import based on available API
try:
    # Grok API
    import anthropic
    USE_GROK = True
except ImportError:
    try:
        # Gemini API
        from google import generativeai as genai
        USE_GROK = False
    except ImportError:
        print("Error: Neither Grok nor Gemini API libraries are available.")
        print("Please install one of the following:")
        print("  pip install anthropic  # For Grok")
        print("  pip install google-generativeai  # For Gemini")
        sys.exit(1)

# Import our example selection module
from example_selection import select_examples_for_segmentation

# Initialize API clients
if USE_GROK:
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
        sys.exit(1)
else:
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

# Define data models for structured output
class TaggedToken(BaseModel):
    token: str
    pos: str

class TaggedSentence(BaseModel):
    sentence: str
    tokens: List[TaggedToken]

class TaggedSentences(BaseModel):
    sentences: List[TaggedSentence]

# Function to segment and tag a text with example selection
def tag_sentences_ud_with_examples(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using selected similar examples
    for improved segmentation and tagging.
    
    Args:
        text_to_tag: The text to tag (one or more sentences)
        
    Returns:
        TaggedSentences object or None if there was an error
    """
    # First, select relevant examples for segmentation
    relevant_examples = select_examples_for_segmentation(text_to_tag, top_n=3)
    
    # Format examples for the prompt
    examples_text = "\n".join([
        f"Example {i+1}: \"{example['text']}\"\n"
        f"Segmentation: {example['segmented']}\n"
        for i, example in enumerate(relevant_examples)
    ])
    
    # Construct the prompt with enhanced instructions and selected examples
    prompt = f"""You are a linguist specializing in Universal Dependencies (UD) part-of-speech tagging.
Your task is to segment the input text into tokens according to UD guidelines and then assign appropriate UD POS tags.

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

I've selected the following examples that contain segmentation patterns similar to your input text. 
Use these as reference for how to correctly segment and tag similar constructions:

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

Input text to segment and tag:
"{text_to_tag}"

IMPORTANT: First correctly segment the text according to UD guidelines, paying special attention to the examples provided. Then assign the appropriate POS tag to each token.

Your response MUST be valid JSON with the following format:
{{
  "sentences": [
    {{
      "sentence": "original sentence text",
      "tokens": [
        {{ "token": "token1", "pos": "POSTAG" }},
        {{ "token": "token2", "pos": "POSTAG" }},
        ...
      ]
    }},
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
                max_tokens=4000,
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
                return TaggedSentences(**result_json)
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
                return TaggedSentences(**result_json)
            except Exception as e:
                print(f"Error parsing API response: {e}")
                print(f"Response text: {response.text[:100]}...")
                return None
                
    except Exception as e:
        print(f"API call error: {e}")
        return None

# Process multiple sentences with rate limiting
def batch_process_sentences(sentences: List[str], batch_size: int = 5, wait_time: int = 10) -> List[Dict]:
    """
    Process multiple sentences with appropriate batching and rate limiting.
    
    Args:
        sentences: List of sentences to process
        batch_size: Number of sentences to process in each batch
        wait_time: Time to wait between batches (in seconds)
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size}...")
        
        for sentence in batch:
            result = tag_sentences_ud_with_examples(sentence)
            if result:
                results.extend(result.sentences)
            
        # Wait between batches to respect rate limits
        if i + batch_size < len(sentences):
            print(f"Waiting {wait_time} seconds for rate limiting...")
            time.sleep(wait_time)
    
    return results

# Convert to CoNLL-U format
def convert_to_conllu(tagged_sentences: List[TaggedSentence]) -> str:
    """
    Convert tagged sentences to CoNLL-U format.
    
    Args:
        tagged_sentences: List of tagged sentences
        
    Returns:
        String in CoNLL-U format
    """
    conllu_output = []
    
    for i, sentence in enumerate(tagged_sentences):
        # Add sentence ID and text
        conllu_output.append(f"# sent_id = {i+1}")
        conllu_output.append(f"# text = {sentence.sentence}")
        
        # Add tokens
        for j, token in enumerate(sentence.tokens):
            # Format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            conllu_output.append(f"{j+1}\t{token.token}\t_\t{token.pos}\t_\t_\t_\t_\t_\t_")
        
        # Add empty line between sentences
        conllu_output.append("")
    
    return "\n".join(conllu_output)

# Evaluate against gold standard
def evaluate(predicted_conllu: str, gold_conllu: str) -> Dict[str, float]:
    """
    Evaluate predictions against gold standard.
    
    Args:
        predicted_conllu: Predicted CoNLL-U format
        gold_conllu: Gold standard CoNLL-U format
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Parse CoNLL-U files
    pred_lines = predicted_conllu.strip().split("\n")
    gold_lines = gold_conllu.strip().split("\n")
    
    # Extract token/tag pairs from both files
    pred_tokens = []
    gold_tokens = []
    
    for line in pred_lines:
        if line and not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) >= 4:
                pred_tokens.append((fields[1], fields[3]))  # (form, upos)
    
    for line in gold_lines:
        if line and not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) >= 4:
                gold_tokens.append((fields[1], fields[3]))  # (form, upos)
    
    # Calculate metrics
    if len(pred_tokens) != len(gold_tokens):
        print(f"Warning: Token count mismatch. Predicted: {len(pred_tokens)}, Gold: {len(gold_tokens)}")
        # Use the shorter length for comparison
        min_len = min(len(pred_tokens), len(gold_tokens))
        pred_tokens = pred_tokens[:min_len]
        gold_tokens = gold_tokens[:min_len]
    
    # Segmentation accuracy
    correct_seg = sum(1 for i in range(len(pred_tokens)) if pred_tokens[i][0] == gold_tokens[i][0])
    seg_accuracy = correct_seg / len(gold_tokens) if gold_tokens else 0
    
    # Tagging accuracy (only for correctly segmented tokens)
    correct_pos = sum(1 for i in range(len(pred_tokens)) 
                      if pred_tokens[i][0] == gold_tokens[i][0] and pred_tokens[i][1] == gold_tokens[i][1])
    pos_accuracy = correct_pos / correct_seg if correct_seg > 0 else 0
    
    # Overall accuracy
    overall_accuracy = correct_pos / len(gold_tokens) if gold_tokens else 0
    
    return {
        "segmentation_accuracy": seg_accuracy,
        "pos_tagging_accuracy": pos_accuracy,
        "overall_accuracy": overall_accuracy,
        "total_tokens": len(gold_tokens),
        "correctly_segmented": correct_seg,
        "correctly_tagged": correct_pos
    }

# Main function to demonstrate functionality
def main():
    # Example sentences that demonstrate various segmentation challenges
    test_sentences = [
        "I don't think we'll be able to make it to John's party on Friday.",
        "The cost-benefit analysis shows we might save $5,000 in the first quarter of 2023.",
        "Please email john.doe@example.com or visit https://example.org for more information.",
        "She works at IBM in New York City and earns well-above average.",
        "Dr. Smith's paper was published in the Journal of A.I. Research in the U.S."
    ]
    
    print("\nUD POS Tagger with Example Selection\n")
    print("This tagger uses similar examples to guide segmentation decisions\n")
    
    for i, sentence in enumerate(test_sentences):
        print(f"Example {i+1}: \"{sentence}\"")
        
        # Process one sentence at a time
        result = tag_sentences_ud_with_examples(sentence)
        
        if result and result.sentences:
            tagged = result.sentences[0]
            print("Segmentation and tagging result:")
            for token in tagged.tokens:
                print(f"  {token.token} → {token.pos}")
        else:
            print("  Error processing sentence")
        
        print()
    
    # Process a batch of sentences
    print("\nBatch processing example:")
    results = batch_process_sentences(test_sentences[:2])
    
    if results:
        conllu_output = convert_to_conllu(results)
        print("\nCoNLL-U format output sample:")
        print(conllu_output[:500] + "..." if len(conllu_output) > 500 else conllu_output)

if __name__ == "__main__":
    main()