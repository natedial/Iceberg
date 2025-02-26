"""
Preprocessing Module

This module handles text preprocessing tasks such as cleaning, tokenization,
and other NLP preprocessing steps.
"""

import re
import logging
import spacy
import nltk
from pathlib import Path
from typing import List, Dict, Any, Union

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.warning("Spacy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Preprocess text by cleaning, tokenizing, and performing other NLP tasks.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Dict: Dictionary containing the processed text and metadata
    """
    logger.info("Preprocessing text...")
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Process with spaCy
    doc = nlp(cleaned_text)
    
    # Extract tokens, sentences, and other information
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    sentences = [sent.text for sent in doc.sents]
    
    # Create processed data dictionary
    processed_data = {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "sentences": sentences,
        "doc": doc  # Include the spaCy doc for further processing
    }
    
    logger.info(f"Text preprocessing complete. Extracted {len(tokens)} tokens and {len(sentences)} sentences.")
    
    return processed_data


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that aren't relevant for analysis
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"\(\)\[\]\{\}]', '', text)
    
    return text.strip()


def save_processed_text(processed_data: Dict[str, Any], original_file_path: Path) -> Path:
    """
    Save the processed text data to the data/processed directory.
    
    Args:
        processed_data: Dictionary containing processed text data
        original_file_path: Path to the original file
        
    Returns:
        Path: Path to the saved processed data file
    """
    import json
    
    # Create a filename based on the original file
    filename = original_file_path.stem + "_processed.json"
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a serializable version of the processed data
    serializable_data = {
        "original_text": processed_data["original_text"],
        "cleaned_text": processed_data["cleaned_text"],
        "tokens": processed_data["tokens"],
        "lemmas": processed_data["lemmas"],
        "pos_tags": processed_data["pos_tags"],
        "sentences": processed_data["sentences"]
    }
    
    # Save the processed data
    output_path = processed_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)
    
    logger.info(f"Processed data saved to: {output_path}")
    return output_path
