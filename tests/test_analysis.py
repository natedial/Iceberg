"""
Test Analysis Module

This module contains tests for the analysis module.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.preprocessing import preprocess_text
from modules.analysis import analyze_text, compare_texts


def test_analyze_text():
    """Test the analyze_text function."""
    # Create a sample processed data dictionary
    processed_data = {
        "original_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time.",
        "cleaned_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time.",
        "tokens": ["The", "Federal", "Reserve", "is", "committed", "to", "using", "its", "full", "range", "of", "tools", "to", "support", "the", "U.S.", "economy", "in", "this", "challenging", "time", "."],
        "lemmas": ["the", "federal", "reserve", "be", "commit", "to", "use", "its", "full", "range", "of", "tool", "to", "support", "the", "U.S.", "economy", "in", "this", "challenge", "time", "."],
        "pos_tags": ["DET", "PROPN", "PROPN", "AUX", "VERB", "ADP", "VERB", "DET", "ADJ", "NOUN", "ADP", "NOUN", "ADP", "VERB", "DET", "PROPN", "NOUN", "ADP", "DET", "ADJ", "NOUN", "PUNCT"],
        "sentences": ["The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time."]
    }
    
    # Analyze the text
    analysis_results = analyze_text(processed_data)
    
    # Check that the analysis results contain the expected keys
    assert "sentiment" in analysis_results
    assert "intent" in analysis_results
    assert "key_phrases" in analysis_results
    assert "metadata" in analysis_results
    
    # Check that the sentiment analysis contains the expected keys
    assert "positive" in analysis_results["sentiment"]
    assert "negative" in analysis_results["sentiment"]
    assert "neutral" in analysis_results["sentiment"]
    assert "compound" in analysis_results["sentiment"]
    
    # Check that the intent analysis contains the expected keys
    assert "hawkish_score" in analysis_results["intent"]
    assert "dovish_score" in analysis_results["intent"]
    assert "overall_intent" in analysis_results["intent"]


def test_compare_texts():
    """Test the compare_texts function."""
    # Create sample processed data dictionaries
    processed_data1 = {
        "original_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time.",
        "cleaned_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time.",
        "tokens": ["The", "Federal", "Reserve", "is", "committed", "to", "using", "its", "full", "range", "of", "tools", "to", "support", "the", "U.S.", "economy", "in", "this", "challenging", "time", "."],
        "lemmas": ["the", "federal", "reserve", "be", "commit", "to", "use", "its", "full", "range", "of", "tool", "to", "support", "the", "U.S.", "economy", "in", "this", "challenge", "time", "."],
        "pos_tags": ["DET", "PROPN", "PROPN", "AUX", "VERB", "ADP", "VERB", "DET", "ADJ", "NOUN", "ADP", "NOUN", "ADP", "VERB", "DET", "PROPN", "NOUN", "ADP", "DET", "ADJ", "NOUN", "PUNCT"],
        "sentences": ["The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time."]
    }
    
    processed_data2 = {
        "original_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time and to promote its maximum employment and price stability goals.",
        "cleaned_text": "The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time and to promote its maximum employment and price stability goals.",
        "tokens": ["The", "Federal", "Reserve", "is", "committed", "to", "using", "its", "full", "range", "of", "tools", "to", "support", "the", "U.S.", "economy", "in", "this", "challenging", "time", "and", "to", "promote", "its", "maximum", "employment", "and", "price", "stability", "goals", "."],
        "lemmas": ["the", "federal", "reserve", "be", "commit", "to", "use", "its", "full", "range", "of", "tool", "to", "support", "the", "U.S.", "economy", "in", "this", "challenge", "time", "and", "to", "promote", "its", "maximum", "employment", "and", "price", "stability", "goal", "."],
        "pos_tags": ["DET", "PROPN", "PROPN", "AUX", "VERB", "ADP", "VERB", "DET", "ADJ", "NOUN", "ADP", "NOUN", "ADP", "VERB", "DET", "PROPN", "NOUN", "ADP", "DET", "ADJ", "NOUN", "CCONJ", "ADP", "VERB", "DET", "ADJ", "NOUN", "CCONJ", "NOUN", "NOUN", "NOUN", "PUNCT"],
        "sentences": ["The Federal Reserve is committed to using its full range of tools to support the U.S. economy in this challenging time and to promote its maximum employment and price stability goals."]
    }
    
    # Compare the texts
    comparison_results = compare_texts(processed_data1, processed_data2)
    
    # Check that the comparison results contain the expected keys
    assert "similarity_scores" in comparison_results
    assert "sentence_changes" in comparison_results
    assert "phrase_changes" in comparison_results
    assert "sentiment_comparison" in comparison_results
    assert "intent_comparison" in comparison_results
    
    # Check that the similarity scores contain the expected keys
    assert "tfidf_similarity" in comparison_results["similarity_scores"]
    assert "jaccard_similarity" in comparison_results["similarity_scores"]
    assert "overall_similarity" in comparison_results["similarity_scores"]
    
    # Check that the phrase changes contain the expected keys
    assert "additions" in comparison_results["phrase_changes"]
    assert "deletions" in comparison_results["phrase_changes"]
    assert "addition_count" in comparison_results["phrase_changes"]
    assert "deletion_count" in comparison_results["phrase_changes"]


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-v", __file__])
