"""
Error Explanation for POS Tagging Errors

This script analyzes tagging errors and uses an LLM to explain 
the likely causes of these errors in a structured JSON format.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
import sys

# Import the appropriate LLM API (Grok or Gemini)
from ud_pos_tagger_grok import client as grok_client
# Uncomment to use Gemini instead
# from google import generativeai as genai

def format_error_for_explanation(token: str, correct_tag: str, predicted_tag: str, context: str) -> str:
    """
    Formats an error for explanation by the LLM.
    
    Args:
        token: The word/token that was incorrectly tagged
        correct_tag: The gold standard correct tag
        predicted_tag: The predicted (incorrect) tag
        context: The sentence or context where the error occurred
        
    Returns:
        Formatted error description
    """
    return f"""Token: {token}
Correct tag: {correct_tag}
Predicted tag: {predicted_tag}
Context: {context}
"""


def explain_pos_tagging_error(token: str, correct_tag: str, predicted_tag: str, context: str) -> Dict:
    """
    Use LLM to explain why a particular token was tagged incorrectly.
    
    Args:
        token: The word/token that was incorrectly tagged
        correct_tag: The gold standard correct tag
        predicted_tag: The predicted (incorrect) tag
        context: The sentence or context where the error occurred
        
    Returns:
        Dictionary containing the explanation
    """
    # Define the prompt for error explanation
    prompt = f"""You are a linguistic expert specializing in Universal Dependencies (UD) part-of-speech tagging.
Your task is to explain why a tagger might have incorrectly tagged a specific word in a sentence.

Provided POS tagging error:
{format_error_for_explanation(token, correct_tag, predicted_tag, context)}

Universal Dependencies POS tags:
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

Please explain the error and categorize it.
Your response should be a JSON object with the following structure:
{{
  "word": "<the token>",
  "correct_tag": "<the correct tag>",
  "predicted_tag": "<the predicted tag>",
  "explanation": "<A detailed explanation of why this error might have occurred, discussing ambiguity, context, and linguistic factors. Include references to how the word functions grammatically in this specific context and why the model might have confused the tags.>",
  "category": "<A short category label for this type of error, e.g., 'Ambiguity (NOUN/VERB)', 'Context Dependence', 'Rare Usage', etc.>"
}}

Make sure the explanation is detailed (at least 3-4 sentences), insightful, and educational about linguistic features.
"""

    try:
        # Call LLM API based on which one is available
        if 'grok_client' in globals():
            # Grok API
            response = grok_client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
        elif 'genai' in globals():
            # Gemini API
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
        else:
            raise ImportError("No LLM API client available")
        
        return result
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        # Return a basic structure in case of error
        return {
            "word": token,
            "correct_tag": correct_tag,
            "predicted_tag": predicted_tag,
            "explanation": f"Error generating explanation: {e}",
            "category": "Error"
        }


def batch_explain_errors(errors, max_errors=10, output_file=None):
    """
    Generate explanations for a batch of errors.
    
    Args:
        errors: List of error tuples (token, correct_tag, predicted_tag, context)
        max_errors: Maximum number of errors to explain (for rate limiting)
        output_file: File to save explanations to
        
    Returns:
        List of explanation dictionaries
    """
    print(f"Generating explanations for {min(len(errors), max_errors)} errors...")
    explanations = []
    
    for i, (token, correct_tag, predicted_tag, context) in enumerate(errors[:max_errors]):
        print(f"Explaining error {i+1}/{min(len(errors), max_errors)}: {token} ({correct_tag} vs {predicted_tag})")
        explanation = explain_pos_tagging_error(token, correct_tag, predicted_tag, context)
        explanations.append(explanation)
        
        # Respect rate limits
        if i < min(len(errors), max_errors) - 1:
            import time
            print("Waiting for rate limits...")
            time.sleep(5)  # Adjust based on API limits
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(explanations, f, indent=2)
        print(f"Saved {len(explanations)} explanations to {output_file}")
    
    return explanations


def extract_error_categories(explanations):
    """
    Extract and count error categories from explanations.
    
    Args:
        explanations: List of explanation dictionaries
        
    Returns:
        Counter of error categories
    """
    from collections import Counter
    categories = [explanation.get("category", "Unknown") for explanation in explanations]
    return Counter(categories)


def generate_synthetic_examples(categories, total_examples=200, errors_per_sentence=3):
    """
    Generate synthetic examples with multiple error categories per sentence.

    Args:
        categories: List of error categories
        total_examples: Total number of sentences to generate
        errors_per_sentence: Number of challenging words per sentence

    Returns:
        Dictionary of synthetic examples with full POS tagging
    """
    examples = {}

    # Calculate how many examples to generate in each batch
    batch_size = 20  # Generate 20 sentences per API call to avoid context limits
    num_batches = (total_examples + batch_size - 1) // batch_size  # Ceiling division

    all_examples = []

    # Convert categories to list if it's not already
    categories_list = list(categories)

    for batch in range(num_batches):
        # Determine number of examples for this batch
        current_batch_size = min(batch_size, total_examples - batch * batch_size)
        if current_batch_size <= 0:
            break

        print(f"Generating batch {batch+1}/{num_batches} ({current_batch_size} sentences)...")

        # Construct prompt for generating examples with multiple error categories per sentence
        categories_str = "\n".join([f"- {category}" for category in categories_list])

        prompt = f"""You are a linguistic expert specializing in Universal Dependencies (UD) part-of-speech tagging.
Your task is to generate {current_batch_size} challenging synthetic sentences, each containing {errors_per_sentence} words
that would be difficult to tag correctly, spanning multiple error categories.

These are the error categories to choose from:
{categories_str}

For each sentence:
1. Include exactly {errors_per_sentence} challenging words from different error categories
2. Provide the complete POS tagging for the entire sentence (for all words, not just the challenging ones)
3. Make the sentences realistic and grammatically correct

Your response should be a JSON array with the following structure:
[
  {{
    "sentence": "<example sentence>",
    "complete_pos_tagging": [
      ["word1", "CORRECT_TAG"],
      ["word2", "CORRECT_TAG"],
      ...
    ],
    "challenging_words": [
      {{
        "word": "<challenging word 1>",
        "correct_tag": "<correct UD tag>",
        "likely_error_tag": "<tag a model might incorrectly assign>",
        "error_category": "<the category this error falls under>",
        "explanation": "<brief explanation of why this is challenging>"
      }},
      {{
        "word": "<challenging word 2>",
        "correct_tag": "<correct UD tag>",
        "likely_error_tag": "<tag a model might incorrectly assign>",
        "error_category": "<the category this error falls under>",
        "explanation": "<brief explanation of why this is challenging>"
      }},
      {{
        "word": "<challenging word 3>",
        "correct_tag": "<correct UD tag>",
        "likely_error_tag": "<tag a model might incorrectly assign>",
        "error_category": "<the category this error falls under>",
        "explanation": "<brief explanation of why this is challenging>"
      }}
    ]
  }}
]

Universal Dependencies POS tags:
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

Make sure each example is realistic, grammatically correct, and clearly demonstrates multiple error categories.
"""

        try:
            # Call LLM API
            if 'grok_client' in globals():
                response = grok_client.chat.completions.create(
                    model="grok-3-mini",
                    messages=[
                        {"role": "system", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                batch_examples = json.loads(response.choices[0].message.content)

                # Add to our collection
                if isinstance(batch_examples, list):
                    all_examples.extend(batch_examples)
                else:
                    print(f"Warning: Unexpected response format in batch {batch+1}")

                # Respect rate limits
                if batch < num_batches - 1:
                    import time
                    print("Waiting for rate limits...")
                    time.sleep(10)  # Adjust based on API limits

            elif 'genai' in globals():
                # Gemini API implementation
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('gemini-2.0-flash-lite')
                response = model.generate_content(
                    contents=prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                batch_examples = json.loads(response.text)

                # Add to our collection
                if isinstance(batch_examples, list):
                    all_examples.extend(batch_examples)
                else:
                    print(f"Warning: Unexpected response format in batch {batch+1}")

                # Respect rate limits
                if batch < num_batches - 1:
                    import time
                    print("Waiting for rate limits...")
                    time.sleep(10)  # Adjust based on API limits
            else:
                raise ImportError("No LLM API client available")

        except Exception as e:
            print(f"Error generating batch {batch+1}: {e}")

    # Save all batches to file
    examples = {"examples": all_examples}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"synthetic_examples_{total_examples}_{timestamp}.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(all_examples)} synthetic examples to synthetic_examples_{total_examples}_{timestamp}.json")

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="POS Tagging Error Analysis and Synthetic Data Generation")
    parser.add_argument('--generate', action='store_true', help='Generate synthetic examples')
    parser.add_argument('--examples', type=int, default=200, help='Number of examples to generate (default: 200)')
    parser.add_argument('--errors-per-example', type=int, default=3, help='Number of errors per example (default: 3)')
    args = parser.parse_args()

    if args.generate:
        # Use predefined error categories for generating synthetic data
        error_categories = [
            "Ambiguity (ADJ/NOUN)",
            "Symbol-Punctuation Ambiguity",
            "Proper Noun Recognition",
            "Numeral vs. Adjective Confusion",
            "Adverb vs. Adjective Confusion",
            "Particle vs. Adposition",
            "Proper Noun vs. Common Noun",
            "Auxiliary vs. Main Verb",
            "Subordinating vs. Coordinating Conjunction",
            "Determiner vs. Pronoun"
        ]
        print(f"Generating {args.examples} synthetic examples with {args.errors_per_example} errors per example...")
        examples = generate_synthetic_examples(error_categories, total_examples=args.examples, errors_per_sentence=args.errors_per_example)
        print("\nSynthetic data generation complete!")
        return

    # Default behavior: Analyze example errors
    print("Running error analysis on example errors...")
    example_errors = [
        ("ONLINE", "ADJ", "NOUN", "WE HAVE A DATE FOR THE RELEASE OF RAGNAROK ONLINE 2 ( beta anyway )"),
        ("-", "SYM", "PUNCT", "September 16 - 18 , this was announced by Gravity"),
        ("Gravity", "PROPN", "NOUN", "this was announced by Gravity CEO Kim Jung - Ryool"),
        ("16th", "NOUN", "ADJ", "on either 16th or 17th of july"),
        ("july", "PROPN", "ADV", "on either 16th or 17th of july")
    ]

    # Generate explanations
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    explanations = batch_explain_errors(
        example_errors,
        max_errors=5,
        output_file=f"error_explanations_{timestamp}.json"
    )

    # Extract and display categories
    categories = extract_error_categories(explanations)
    print("\nError Categories:")
    for category, count in categories.items():
        print(f"- {category}: {count}")

    # Generate a few synthetic examples from the identified categories
    if len(categories) > 0:
        examples = generate_synthetic_examples(categories.keys(), total_examples=5, errors_per_sentence=2)
        print("\nGenerated a few sample synthetic examples for each category")
        print("To generate the full dataset of 200 examples, run: python error_explanation.py --generate")


if __name__ == "__main__":
    main()