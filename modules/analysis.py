"""
Analysis Module

This module handles the analysis of text data, including sentiment analysis,
intent detection, and text comparison.
"""

import logging
import difflib
import spacy
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Spacy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def analyze_text(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single text document.
    
    Args:
        processed_data: Dictionary containing processed text data
        
    Returns:
        Dict: Dictionary containing analysis results
    """
    logger.info("Analyzing text...")
    
    # Extract sentiment
    sentiment_results = analyze_sentiment(processed_data)
    
    # Extract intent (hawkish vs. dovish for Fed statements)
    intent_results = analyze_intent(processed_data)
    
    # Extract key phrases
    key_phrases = extract_key_phrases(processed_data)
    
    # Create analysis results dictionary
    analysis_results = {
        "sentiment": sentiment_results,
        "intent": intent_results,
        "key_phrases": key_phrases,
        "metadata": {
            "token_count": len(processed_data["tokens"]),
            "sentence_count": len(processed_data["sentences"]),
            "average_sentence_length": len(processed_data["tokens"]) / len(processed_data["sentences"]) if processed_data["sentences"] else 0
        }
    }
    
    logger.info("Text analysis complete.")
    
    return analysis_results


def compare_texts(processed_data1: Dict[str, Any], processed_data2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two text documents to identify changes and similarities.
    
    Args:
        processed_data1: Dictionary containing processed text data for the first document
        processed_data2: Dictionary containing processed text data for the second document
        
    Returns:
        Dict: Dictionary containing comparison results
    """
    logger.info("Comparing texts...")
    
    # Calculate similarity scores
    similarity_scores = calculate_similarity(processed_data1, processed_data2)
    
    # Identify changes at the sentence level
    sentence_changes = identify_sentence_changes(processed_data1["sentences"], processed_data2["sentences"])
    
    # Identify phrase-level changes
    phrase_changes = identify_phrase_changes(processed_data1["cleaned_text"], processed_data2["cleaned_text"])
    
    # Compare sentiment and intent
    sentiment1 = analyze_sentiment(processed_data1)
    sentiment2 = analyze_sentiment(processed_data2)
    intent1 = analyze_intent(processed_data1)
    intent2 = analyze_intent(processed_data2)
    
    # Create comparison results dictionary
    comparison_results = {
        "similarity_scores": similarity_scores,
        "sentence_changes": sentence_changes,
        "phrase_changes": phrase_changes,
        "sentiment_comparison": {
            "document1": sentiment1,
            "document2": sentiment2,
            "change": sentiment2["compound"] - sentiment1["compound"]
        },
        "intent_comparison": {
            "document1": intent1,
            "document2": intent2,
            "change": "More hawkish" if intent2["hawkish_score"] > intent1["hawkish_score"] else "More dovish"
        }
    }
    
    logger.info("Text comparison complete.")
    
    return comparison_results


def analyze_sentiment(processed_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze the sentiment of a text document.
    
    Args:
        processed_data: Dictionary containing processed text data
        
    Returns:
        Dict: Dictionary containing sentiment scores
    """
    # Use spaCy for basic sentiment analysis
    # Note: For more advanced sentiment analysis, consider using a dedicated library like VADER or TextBlob
    
    doc = processed_data.get("doc")
    if not doc:
        doc = nlp(processed_data["cleaned_text"])
    
    # Calculate positive, negative, and neutral scores based on token attributes
    positive_words = ["increase", "growth", "strong", "robust", "improve", "positive", "recovery", "expand"]
    negative_words = ["decrease", "decline", "weak", "deteriorate", "negative", "contraction", "recession"]
    
    positive_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
    negative_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)
    total_count = len([token for token in doc if token.is_alpha])
    
    positive_score = positive_count / total_count if total_count > 0 else 0
    negative_score = negative_count / total_count if total_count > 0 else 0
    neutral_score = 1 - (positive_score + negative_score)
    compound_score = positive_score - negative_score
    
    return {
        "positive": positive_score,
        "negative": negative_score,
        "neutral": neutral_score,
        "compound": compound_score
    }


def analyze_intent(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the intent of a text document (e.g., hawkish vs. dovish for Fed statements).
    
    Args:
        processed_data: Dictionary containing processed text data
        
    Returns:
        Dict: Dictionary containing intent analysis results
    """
    # Define hawkish and dovish terms
    hawkish_terms = [
        "inflation", "tightening", "raise rates", "increase rates", "higher rates",
        "restrictive", "overheating", "price stability", "combat inflation"
    ]
    
    dovish_terms = [
        "growth", "employment", "labor market", "accommodative", "lower rates",
        "reduce rates", "cut rates", "stimulus", "support", "maximum employment"
    ]
    
    # Count occurrences of hawkish and dovish terms
    text = processed_data["cleaned_text"].lower()
    
    hawkish_count = sum(text.count(term) for term in hawkish_terms)
    dovish_count = sum(text.count(term) for term in dovish_terms)
    
    total_count = hawkish_count + dovish_count
    
    # Calculate scores
    hawkish_score = hawkish_count / total_count if total_count > 0 else 0.5
    dovish_score = dovish_count / total_count if total_count > 0 else 0.5
    
    # Determine overall intent
    if hawkish_score > dovish_score:
        overall_intent = "hawkish"
    elif dovish_score > hawkish_score:
        overall_intent = "dovish"
    else:
        overall_intent = "neutral"
    
    return {
        "hawkish_score": hawkish_score,
        "dovish_score": dovish_score,
        "overall_intent": overall_intent,
        "hawkish_terms_found": hawkish_count,
        "dovish_terms_found": dovish_count
    }


def extract_key_phrases(processed_data: Dict[str, Any]) -> List[str]:
    """
    Extract key phrases from a text document.
    
    Args:
        processed_data: Dictionary containing processed text data
        
    Returns:
        List: List of key phrases
    """
    doc = processed_data.get("doc")
    if not doc:
        doc = nlp(processed_data["cleaned_text"])
    
    # Extract noun phrases as key phrases
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    
    # Limit to top 10 phrases
    return key_phrases[:10]


def calculate_similarity(processed_data1: Dict[str, Any], processed_data2: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate similarity scores between two text documents.
    
    Args:
        processed_data1: Dictionary containing processed text data for the first document
        processed_data2: Dictionary containing processed text data for the second document
        
    Returns:
        Dict: Dictionary containing similarity scores
    """
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([
        processed_data1["cleaned_text"],
        processed_data2["cleaned_text"]
    ])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Calculate Jaccard similarity for tokens
    tokens1 = set(processed_data1["tokens"])
    tokens2 = set(processed_data2["tokens"])
    jaccard_similarity = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2)) if tokens1 or tokens2 else 0
    
    return {
        "tfidf_similarity": float(tfidf_similarity),
        "jaccard_similarity": jaccard_similarity,
        "overall_similarity": (float(tfidf_similarity) + jaccard_similarity) / 2
    }


def identify_sentence_changes(sentences1: List[str], sentences2: List[str]) -> List[Dict[str, Any]]:
    """
    Identify changes between sentences in two documents.
    
    Args:
        sentences1: List of sentences from the first document
        sentences2: List of sentences from the second document
        
    Returns:
        List: List of dictionaries containing sentence change information
    """
    # Use difflib to identify sentence-level changes
    matcher = difflib.SequenceMatcher(None, sentences1, sentences2)
    changes = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            for i in range(i1, i2):
                for j in range(j1, j2):
                    changes.append({
                        "type": "replace",
                        "old_sentence": sentences1[i],
                        "new_sentence": sentences2[j],
                        "old_index": i,
                        "new_index": j
                    })
        elif tag == 'delete':
            for i in range(i1, i2):
                changes.append({
                    "type": "delete",
                    "old_sentence": sentences1[i],
                    "new_sentence": None,
                    "old_index": i,
                    "new_index": None
                })
        elif tag == 'insert':
            for j in range(j1, j2):
                changes.append({
                    "type": "insert",
                    "old_sentence": None,
                    "new_sentence": sentences2[j],
                    "old_index": None,
                    "new_index": j
                })
    
    return changes


def identify_phrase_changes(text1: str, text2: str) -> Dict[str, Any]:
    """
    Identify phrase-level changes between two documents.
    
    Args:
        text1: Text of the first document
        text2: Text of the second document
        
    Returns:
        Dict: Dictionary containing phrase change information
    """
    # Generate a unified diff
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        lineterm='',
        n=0
    )
    
    # Parse the diff to extract additions and deletions
    additions = []
    deletions = []
    
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            additions.append(line[1:])
        elif line.startswith('-') and not line.startswith('---'):
            deletions.append(line[1:])
    
    return {
        "additions": additions,
        "deletions": deletions,
        "addition_count": len(additions),
        "deletion_count": len(deletions)
    }
