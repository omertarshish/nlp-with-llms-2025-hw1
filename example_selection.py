#!/usr/bin/env python
"""
Example Selection for Word Segmentation

This module implements a strategy to select the most similar examples
from a collection of well-segmented sentences, to help with segmenting
new sentences according to UD guidelines.

Key components:
1. A collection of good examples with proper segmentation
2. A similarity function to determine relevance
3. Methods to select the top-N most similar examples
"""

import os
import re
import json
import string
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set, Any
import conllu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# Try to download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# UD segmentation patterns collection
# These are examples of specific segmentation challenges
SEGMENTATION_EXAMPLES = {
    "contractions": [
        {"text": "don't worry about it", "segmented": ["do", "n't", "worry", "about", "it"]},
        {"text": "I'll be there soon", "segmented": ["I", "'ll", "be", "there", "soon"]},
        {"text": "she's already finished", "segmented": ["she", "'s", "already", "finished"]},
        {"text": "we've been waiting", "segmented": ["we", "'ve", "been", "waiting"]},
        {"text": "they'd rather stay home", "segmented": ["they", "'d", "rather", "stay", "home"]},
    ],
    "multi_word_expressions": [
        {"text": "New York is a big city", "segmented": ["New", "York", "is", "a", "big", "city"]},
        {"text": "she lives in San Francisco", "segmented": ["she", "lives", "in", "San", "Francisco"]},
        {"text": "the United States of America", "segmented": ["the", "United", "States", "of", "America"]},
    ],
    "hyphenated_compounds": [
        {"text": "it's a well-known fact", "segmented": ["it", "'s", "a", "well-known", "fact"]},
        {"text": "we need up-to-date information", "segmented": ["we", "need", "up-to-date", "information"]},
        {"text": "the cost-benefit analysis shows", "segmented": ["the", "cost-benefit", "analysis", "shows"]},
    ],
    "numerals": [
        {"text": "chapter 3.2 explains it", "segmented": ["chapter", "3.2", "explains", "it"]},
        {"text": "the temperature is 72.5 degrees", "segmented": ["the", "temperature", "is", "72.5", "degrees"]},
        {"text": "about 4,000 people attended", "segmented": ["about", "4,000", "people", "attended"]},
        {"text": "it costs $5.99", "segmented": ["it", "costs", "$", "5.99"]},
    ],
    "abbreviations": [
        {"text": "Dr. Smith lives on Main St.", "segmented": ["Dr.", "Smith", "lives", "on", "Main", "St."]},
        {"text": "I work for IBM in the U.S.", "segmented": ["I", "work", "for", "IBM", "in", "the", "U.S."]},
        {"text": "she has a Ph.D. in physics", "segmented": ["she", "has", "a", "Ph.D.", "in", "physics"]},
    ],
    "punctuation": [
        {"text": "Wait - what did you say?", "segmented": ["Wait", "-", "what", "did", "you", "say", "?"]},
        {"text": "\"I'm tired,\" she said.", "segmented": ["\"", "I", "'m", "tired", ",", "\"", "she", "said", "."]},
        {"text": "check (and double-check) everything", "segmented": ["check", "(", "and", "double-check", ")", "everything"]},
    ],
    "urls_emails": [
        {"text": "visit https://example.com for details", "segmented": ["visit", "https://example.com", "for", "details"]},
        {"text": "email us at info@example.org", "segmented": ["email", "us", "at", "info@example.org"]},
    ],
    "possessives": [
        {"text": "John's book is on the table", "segmented": ["John", "'s", "book", "is", "on", "the", "table"]},
        {"text": "the children's toys are everywhere", "segmented": ["the", "children", "'s", "toys", "are", "everywhere"]},
    ],
    "special_characters": [
        {"text": "it costs €50 or $60", "segmented": ["it", "costs", "€", "50", "or", "$", "60"]},
        {"text": "the mixture is 30% water", "segmented": ["the", "mixture", "is", "30", "%", "water"]},
    ]
}

# Load examples from UD dataset
def load_ud_examples(file_path: str = "../UD_English-EWT/en_ewt-ud-train.conllu", 
                     max_examples: int = 100) -> List[Dict]:
    """
    Load real examples from UD dataset with their correct segmentation.
    
    Args:
        file_path: Path to the UD dataset file
        max_examples: Maximum number of examples to load
    
    Returns:
        List of examples with text and segmentation
    """
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            parsed_data = conllu.parse(f.read())
            
            # Select a subset of sentences
            selected = parsed_data[:max_examples]
            
            for sentence in selected:
                # Get original text and segmented tokens
                text = sentence.metadata.get('text', '')
                segmented = [token['form'] for token in sentence if token['form']]
                
                if text and segmented:
                    examples.append({
                        "text": text,
                        "segmented": segmented
                    })
    except Exception as e:
        print(f"Error loading UD examples: {e}")
    
    return examples

# Extract features from a sentence for similarity computation
def extract_features(text: str) -> Dict[str, Any]:
    """
    Extract features from a sentence that are relevant for segmentation decisions.
    
    Args:
        text: The input text
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Check for contractions (tokens with apostrophes)
    features['has_contractions'] = bool(re.search(r"\b\w+['']\w+\b|\b\w+['']\b|\b['']\w+\b", text))
    
    # Check for hyphenated compounds
    features['has_hyphens'] = bool(re.search(r"\b\w+-\w+\b", text))
    
    # Check for multi-word expressions (capitalized consecutive words)
    features['has_multi_word_expr'] = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    
    # Check for numbers and measurements
    features['has_numbers'] = bool(re.search(r"\b\d+[\.,]?\d*\b", text))
    
    # Check for currency symbols
    features['has_currency'] = bool(re.search(r"[$€£¥]", text))
    
    # Check for abbreviations
    features['has_abbreviations'] = bool(re.search(r"\b[A-Za-z](\.[A-Za-z])+\.", text))
    
    # Check for quotations
    features['has_quotes'] = bool(re.search(r"[\"'']", text))
    
    # Check for parentheses
    features['has_parentheses'] = bool(re.search(r"[()[\]{}]", text))
    
    # Check for URL or email patterns
    features['has_urls_emails'] = bool(re.search(r"https?://|www\.|@\w+\.\w+", text))
    
    # Check for possessives
    features['has_possessives'] = bool(re.search(r"\b\w+['']s\b", text))
    
    # Count punctuation
    features['punctuation_count'] = sum(1 for char in text if char in string.punctuation)
    
    return features

# Compute feature similarity between two sentences
def compute_feature_similarity(features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
    """
    Compute similarity based on segmentation-relevant features.
    
    Args:
        features1: Features of the first sentence
        features2: Features of the second sentence
        
    Returns:
        Similarity score (0 to 1)
    """
    # Define feature weights (importance for segmentation decisions)
    weights = {
        'has_contractions': 0.2,
        'has_hyphens': 0.15,
        'has_multi_word_expr': 0.1,
        'has_numbers': 0.1,
        'has_currency': 0.1,
        'has_abbreviations': 0.1,
        'has_quotes': 0.05,
        'has_parentheses': 0.05,
        'has_urls_emails': 0.1,
        'has_possessives': 0.1,
        'punctuation_count': 0.05
    }
    
    total_weight = sum(weights.values())
    feature_sim = 0.0
    
    # Calculate weighted feature similarity
    for feature, weight in weights.items():
        if feature == 'punctuation_count':
            # Normalize and compare punctuation counts
            max_count = max(features1[feature], features2[feature]) if max(features1[feature], features2[feature]) > 0 else 1
            sim = 1.0 - abs(features1[feature] - features2[feature]) / max_count
            feature_sim += weight * sim
        else:
            # Boolean features - exact match
            feature_sim += weight if features1[feature] == features2[feature] else 0
    
    return feature_sim / total_weight

# Compute lexical similarity between two sentences
def compute_lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute lexical similarity between two sentences using TF-IDF and cosine similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0 to 1)
    """
    # Create a vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Compute cosine similarity
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Compute overall similarity between two sentences
def compute_similarity(text: str, example: Dict[str, Any]) -> float:
    """
    Compute overall similarity between a text and an example.
    
    Args:
        text: The input text
        example: The example with text and segmentation
        
    Returns:
        Similarity score (0 to 1)
    """
    # Extract features
    features1 = extract_features(text)
    features2 = extract_features(example['text'])
    
    # Compute feature similarity
    feature_sim = compute_feature_similarity(features1, features2)
    
    # Compute lexical similarity
    lexical_sim = compute_lexical_similarity(text, example['text'])
    
    # Compute length similarity
    len1 = len(word_tokenize(text))
    len2 = len(example['segmented'])
    length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
    
    # Weight the different similarity components
    return 0.5 * feature_sim + 0.3 * lexical_sim + 0.2 * length_sim

# Find the most similar examples
def find_similar_examples(text: str, examples: List[Dict], top_n: int = 3) -> List[Dict]:
    """
    Find the top-N most similar examples for a given text.
    
    Args:
        text: The input text
        examples: List of example dictionaries
        top_n: Number of examples to return
        
    Returns:
        List of the most similar examples with similarity scores
    """
    # Compute similarities with all examples
    examples_with_sim = [
        {**example, 'similarity': compute_similarity(text, example)}
        for example in examples
    ]
    
    # Sort by similarity (descending)
    sorted_examples = sorted(examples_with_sim, key=lambda x: x['similarity'], reverse=True)
    
    # Return top N examples
    return sorted_examples[:top_n]

# Select relevant examples based on segmentation challenges
def select_relevant_examples(text: str, top_n: int = 3) -> List[Dict]:
    """
    Select relevant segmentation examples for a given text.
    
    Args:
        text: The text to be segmented
        top_n: Number of examples to select
        
    Returns:
        List of relevant examples with similarity scores
    """
    # Extract features to identify the types of challenges
    features = extract_features(text)
    
    # Collect all applicable examples
    all_examples = []
    
    # Add examples based on detected features
    if features['has_contractions']:
        all_examples.extend(SEGMENTATION_EXAMPLES['contractions'])
    
    if features['has_hyphens']:
        all_examples.extend(SEGMENTATION_EXAMPLES['hyphenated_compounds'])
    
    if features['has_multi_word_expr']:
        all_examples.extend(SEGMENTATION_EXAMPLES['multi_word_expressions'])
    
    if features['has_numbers']:
        all_examples.extend(SEGMENTATION_EXAMPLES['numerals'])
    
    if features['has_abbreviations']:
        all_examples.extend(SEGMENTATION_EXAMPLES['abbreviations'])
    
    if features['has_quotes'] or features['has_parentheses'] or features['punctuation_count'] > 2:
        all_examples.extend(SEGMENTATION_EXAMPLES['punctuation'])
    
    if features['has_urls_emails']:
        all_examples.extend(SEGMENTATION_EXAMPLES['urls_emails'])
    
    if features['has_possessives']:
        all_examples.extend(SEGMENTATION_EXAMPLES['possessives'])
    
    if features['has_currency']:
        all_examples.extend(SEGMENTATION_EXAMPLES['special_characters'])
    
    # If no specific examples were selected, use all categories
    if not all_examples:
        for category in SEGMENTATION_EXAMPLES.values():
            all_examples.extend(category)
    
    # Find similar examples from the collected set
    return find_similar_examples(text, all_examples, top_n)

# Load additional examples from UD dataset and combine with manually curated ones
def load_all_examples() -> List[Dict]:
    """
    Load and combine examples from both the manually curated set and the UD dataset.
    
    Returns:
        Combined list of examples
    """
    # Start with manually curated examples
    all_examples = []
    for category in SEGMENTATION_EXAMPLES.values():
        all_examples.extend(category)
    
    # Load examples from UD dataset
    ud_examples = load_ud_examples()
    all_examples.extend(ud_examples)
    
    return all_examples

# Main function to select examples for a sentence to be segmented
def select_examples_for_segmentation(text: str, top_n: int = 3, use_ud_examples: bool = True) -> List[Dict]:
    """
    Select the most relevant examples for segmenting a given text.
    
    Args:
        text: The text to be segmented
        top_n: Number of examples to select
        use_ud_examples: Whether to include examples from UD dataset
        
    Returns:
        List of selected examples with similarity scores
    """
    if use_ud_examples:
        # Use both manually curated and UD examples
        examples = load_all_examples()
        return find_similar_examples(text, examples, top_n)
    else:
        # Use only manually curated examples based on detected features
        return select_relevant_examples(text, top_n)

# Example usage and testing
def main():
    # Test sentences with various segmentation challenges
    test_sentences = [
        "I can't believe it's already 5 o'clock!",
        "The cost-benefit analysis shows we'll save $5,000.",
        "Please email john.doe@example.com for more information.",
        "She works at IBM in New York City.",
        "Dr. Smith's paper was published in the Journal of A.I."
    ]
    
    print("Example Selection for Word Segmentation\n")
    
    for i, sentence in enumerate(test_sentences):
        print(f"Test {i+1}: \"{sentence}\"")
        print("Features:", extract_features(sentence))
        
        # Select examples using only manually curated examples
        print("\nTop 3 manually curated examples:")
        examples1 = select_examples_for_segmentation(sentence, use_ud_examples=False)
        for j, ex in enumerate(examples1):
            print(f"  {j+1}. \"{ex['text']}\" (similarity: {ex['similarity']:.3f})")
            print(f"     Segmentation: {ex['segmented']}")
        
        # With UD examples (if available)
        try:
            print("\nTop 3 examples (including UD dataset):")
            examples2 = select_examples_for_segmentation(sentence, use_ud_examples=True)
            for j, ex in enumerate(examples2):
                print(f"  {j+1}. \"{ex['text']}\" (similarity: {ex['similarity']:.3f})")
                print(f"     Segmentation: {ex['segmented']}")
        except Exception as e:
            print(f"Error loading UD examples: {e}")
            
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()