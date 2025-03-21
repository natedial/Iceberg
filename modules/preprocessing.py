"""
Preprocessing Module

This module handles text preprocessing tasks such as cleaning, tokenization,
and other NLP preprocessing steps.
"""

import re
import logging
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from typing import List, Dict, Any, Union

import os
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk_resources = [
    'punkt',           # For sentence tokenization
    'punkt_tab',       # Additional tokenization data for NLTK 3.9+
    'stopwords',       # Common stopwords
    'averaged_perceptron_tagger',  # For POS tagging
    'averaged_perceptron_tagger_eng',  # Additional POS tagging data for NLTK 3.9+
    'wordnet',         # For lemmatization
    'vader_lexicon'    # For sentiment analysis
]

def ensure_nltk_resources():
    """Ensure all required NLTK resources are available, downloading them if necessary."""
    for resource in nltk_resources:
        try:
            # Different resources are stored in different directories
            if resource in ['stopwords', 'wordnet', 'vader_lexicon']:
                nltk.data.find(f'corpora/{resource}')
            elif resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'taggers/{resource}')
            
            logger.info(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            logger.info(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)
            logger.info(f"NLTK resource '{resource}' has been downloaded.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Preprocess text by cleaning, tokenizing, and performing other NLP tasks.
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Dict: Dictionary containing the processed text and metadata
    """
    logger.info("Preprocessing text...")
    
    # Ensure all NLTK resources are available
    ensure_nltk_resources()
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Output cleaned text to a file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, f"cleaned_text_{timestamp}.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    logger.info(f"Cleaned text saved to {output_file}")
    
    # Process with NLTK
    try:
        sentences = sent_tokenize(cleaned_text)
        tokens = word_tokenize(cleaned_text)
        pos_tags_tuples = pos_tag(tokens)
        pos_tags = [tag for _, tag in pos_tags_tuples]
        
        # Get lemmas - convert POS tags to WordNet format first
        lemmas = []
        for word, tag in pos_tags_tuples:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag:
                lemmas.append(lemmatizer.lemmatize(word, wn_tag))
            else:
                lemmas.append(lemmatizer.lemmatize(word))
        
        # Create processed data dictionary
        processed_data = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "sentences": sentences
        }
        
        logger.info(f"Text preprocessing complete. Extracted {len(tokens)} tokens and {len(sentences)} sentences.")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error during text preprocessing: {str(e)}")
        raise


def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convert Penn Treebank POS tags to WordNet POS tags.
    
    Args:
        treebank_tag: Penn Treebank POS tag
        
    Returns:
        str: WordNet POS tag or empty string if no match
    """
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        # For other tags, return empty string
        return ''


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove everything prior to "For release at 2:00 p.m." for Federal Reserve press releases
    start_pattern = r"For release at 2:00 p\.m\."
    match = re.search(start_pattern, text)
    if match:
        start_index = match.start()
        text = text[start_index:]

    end_pattern = r"About the FedNews"
    match = re.search(end_pattern, text)
    if match:
        end_index = match.start()
        text = text[:end_index]
    
    # Normalize paragraph structure (replace 2+ newlines with double newline)
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Remove special characters that aren't relevant for analysis
    # Added $ % / to the allowed characters for financial texts
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"\(\)\[\]\{\}\$\%\/]', '', text)
    
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
