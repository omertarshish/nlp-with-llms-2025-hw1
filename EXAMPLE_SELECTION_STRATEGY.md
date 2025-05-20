# Example Selection Strategy for Word Segmentation

This document explains the implementation of example selection for improving word segmentation in Universal Dependencies (UD) POS tagging tasks. The strategy selects the most similar examples to a sentence being segmented, focusing on patterns and features that are relevant to segmentation decisions.

## Overview

Effective word segmentation is critical for accurate POS tagging. By selecting examples that share similar segmentation challenges with the input sentence, we can guide the LLM to make better decisions about token boundaries.

The implementation consists of three key components:

1. **Example Collection**: A set of well-segmented examples covering various segmentation patterns
2. **Similarity Function**: A function that determines the relevance of examples to the input
3. **Example Selection**: A strategy that selects the top-N most similar examples to include in the prompt

## Implementation Details

### 1. Example Collection

The example collection combines:

- **Manually Curated Examples**: Carefully chosen to illustrate specific segmentation challenges:
  - Contractions (`don't` → `do` + `n't`)
  - Multi-word expressions (`New York`)
  - Hyphenated compounds (`well-known`)
  - Numerals (`3.14`, `$5.99`)
  - Abbreviations (`Dr.`, `U.S.`)
  - Punctuation (quotes, parentheses, etc.)
  - URLs and emails
  - Possessives (`John's` → `John` + `'s`)
  - Special characters (currency symbols, %)

- **Real Examples from UD Dataset**: Examples from the UD English-EWT dataset with their correct segmentation, providing authentic and diverse examples.

### 2. Similarity Function

The similarity function combines multiple factors to estimate how relevant an example is for segmenting a given sentence:

```python
def compute_similarity(text: str, example: Dict[str, Any]) -> float:
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
```

The similarity score is a weighted combination of:

- **Feature Similarity (50%)**: Compares segmentation-relevant features of the sentences
- **Lexical Similarity (30%)**: Measures TF-IDF/cosine similarity between texts
- **Length Similarity (20%)**: Considers sentence length, as similar lengths often indicate similar complexity

#### Feature Extraction

The `extract_features` function extracts characteristics that are relevant to segmentation decisions:

```python
def extract_features(text: str) -> Dict[str, Any]:
    features = {}
    
    # Check for contractions (tokens with apostrophes)
    features['has_contractions'] = bool(re.search(r"\b\w+['']\w+\b|\b\w+['']\b|\b['']\w+\b", text))
    
    # Check for hyphenated compounds
    features['has_hyphens'] = bool(re.search(r"\b\w+-\w+\b", text))
    
    # Check for multi-word expressions (capitalized consecutive words)
    features['has_multi_word_expr'] = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text))
    
    # Check for numbers and measurements
    features['has_numbers'] = bool(re.search(r"\b\d+[\.,]?\d*\b", text))
    
    # Additional features...
    
    return features
```

Each feature is given a weight based on its importance for segmentation decisions:

```python
weights = {
    'has_contractions': 0.2,      # High importance
    'has_hyphens': 0.15,          # High importance
    'has_multi_word_expr': 0.1,   # Medium importance
    'has_numbers': 0.1,           # Medium importance
    'has_currency': 0.1,          # Medium importance
    'has_abbreviations': 0.1,     # Medium importance
    'has_quotes': 0.05,           # Lower importance
    'has_parentheses': 0.05,      # Lower importance
    'has_urls_emails': 0.1,       # Medium importance
    'has_possessives': 0.1,       # Medium importance
    'punctuation_count': 0.05     # Lower importance
}
```

### 3. Example Selection

The selection algorithm:

1. Extracts features from the input sentence
2. Computes similarity scores with all examples
3. Selects the top-N most similar examples
4. Returns these examples along with their similarity scores

```python
def select_examples_for_segmentation(text: str, top_n: int = 3, use_ud_examples: bool = True) -> List[Dict]:
    if use_ud_examples:
        # Use both manually curated and UD examples
        examples = load_all_examples()
        return find_similar_examples(text, examples, top_n)
    else:
        # Use only manually curated examples based on detected features
        return select_relevant_examples(text, top_n)
```

## Assumptions about Relevance in Segmentation

Our notion of "relevance" for example selection is based on several key assumptions:

1. **Feature-Based Relevance**: Examples with similar segmentation-critical features are more relevant. For instance, if a sentence contains contractions, examples showing how to segment contractions are highly relevant.

2. **Pattern Matching**: The presence of specific patterns (like hyphens, apostrophes, punctuation) is more important than general textual similarity.

3. **Structural Similarity**: Examples with similar structures (such as similar types of named entities, numbers, or punctuation patterns) are more relevant than examples with similar topics or vocabulary.

4. **Context-Sensitive Segmentation**: Segmentation decisions often depend on context, so similar patterns in similar contexts are particularly relevant.

5. **Frequency of Challenges**: Common segmentation challenges (contractions, possessives) are weighted more heavily than rare ones.

6. **Complexity Matching**: Sentences with similar complexity (measured partly by length and punctuation density) tend to pose similar segmentation challenges.

7. **Feature Hierarchy**: Not all features are equally important:
   - Contractions, hyphens, and possessives are given the highest weights
   - Special symbols and multi-word expressions come next
   - General punctuation and length are given less weight

8. **Format-Specific Rules**: URLs, email addresses, abbreviations, and other special formats have their own segmentation rules, so matching these patterns is essential.

## Integration with POS Tagging

The selected examples are included in the prompt sent to the LLM:

```
I've selected the following examples that contain segmentation patterns similar to your input text. 
Use these as reference for how to correctly segment and tag similar constructions:

Example 1: "don't worry about it"
Segmentation: ['do', "n't", 'worry', 'about', 'it']

Example 2: "John's book is on the table"
Segmentation: ['John', "'s", 'book', 'is', 'on', 'the', 'table']

Example 3: "email us at info@example.org"
Segmentation: ['email', 'us', 'at', 'info@example.org']
```

This enables the LLM to see concrete examples of proper segmentation for patterns similar to those in the input sentence, improving segmentation accuracy.

## Rationale for the Approach

This approach is based on the following considerations:

1. **Few-Shot Learning**: LLMs learn more effectively from concrete examples than from abstract rules.

2. **Contextual Transfer**: Examples similar to the input provide better transfer of segmentation knowledge.

3. **Ambiguity Resolution**: Many segmentation decisions are ambiguous and depend on UD conventions rather than grammar rules. Examples help clarify these conventions.

4. **Efficiency**: By focusing only on relevant examples, we avoid cluttering the prompt with irrelevant information.

5. **Generalization**: The similarity function balances pattern-specific matching with general sentence similarity, promoting good generalization.

## Evaluation

The effectiveness of this approach can be measured by:

1. **Segmentation Accuracy**: The percentage of tokens correctly segmented
2. **POS Tagging Accuracy**: The percentage of correctly segmented tokens that are also correctly tagged
3. **Overall Accuracy**: The percentage of tokens that are both correctly segmented and correctly tagged

Our implementation includes functions to evaluate against gold standard data in CoNLL-U format.

## Limitations and Future Work

1. **Feature Engineering**: The current approach relies on manually designed features. A more advanced approach might learn these features from data.

2. **Limited Context**: The similarity function considers sentences in isolation, while document-level context might be valuable.

3. **Language Specificity**: The current implementation focuses on English. Different languages may require different features and examples.

4. **Computational Cost**: Computing similarity with many examples increases latency. Future work could explore more efficient indexing and retrieval methods.

5. **Example Quality**: The quality of the example set is crucial. A more systematic approach to collecting diverse, high-quality examples would be beneficial.

## Conclusion

The example selection strategy provides a systematic way to improve segmentation accuracy by including relevant examples in the prompt. By defining "relevance" in terms of segmentation-critical features and patterns, we can help the LLM make better decisions about token boundaries, ultimately improving POS tagging accuracy.