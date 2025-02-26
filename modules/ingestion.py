"""
Ingestion Module

This module handles the ingestion of different file types (PDF, HTML, text)
and extracts the text content for further processing.
"""

import os
from pathlib import Path
import pdfplumber
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_file(file_path: Path, file_type: str) -> str:
    """
    Extract text from a file based on its type.
    
    Args:
        file_path: Path to the file to ingest
        file_type: Type of the file ('pdf', 'html', or 'text')
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Ingesting file: {file_path} (type: {file_type})")
    
    try:
        if file_type == "pdf":
            return extract_from_pdf(file_path)
        elif file_type == "html":
            return extract_from_html(file_path)
        else:
            return extract_from_text(file_path)
    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {str(e)}")
        raise


def extract_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise


def extract_from_html(file_path: Path) -> str:
    """
    Extract text from an HTML file.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Extracting text from HTML: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML {file_path}: {str(e)}")
        raise


def extract_from_text(file_path: Path) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Extracting text from text file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {str(e)}")
        raise


def save_raw_text(text: str, original_file_path: Path) -> Path:
    """
    Save the extracted raw text to the data/raw directory.
    
    Args:
        text: The extracted text content
        original_file_path: Path to the original file
        
    Returns:
        Path: Path to the saved raw text file
    """
    # Create a filename based on the original file
    filename = original_file_path.stem + ".txt"
    raw_dir = Path("data/raw")
    raw_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the raw text
    output_path = raw_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Raw text saved to: {output_path}")
    return output_path
