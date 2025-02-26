# Product Requirements Document (PRD): Federal Reserve Text Change Tracker

## 1. Project Overview

### 1.1 Goals
- Track changes in FOMC statements released every 6 weeks.
- Provide historical context for language changes.
- Analyze sentiment and intent in Fed official speeches.

### 1.2 Use Cases
1. **FOMC Statement Change Tracking**:
   - Detect additions, deletions, and substitutions at the phrase level.
   - Reference historical introduction of terms or phrases.
2. **Speech Sentiment and Intent Analysis**:
   - Measure sentiment (positive, negative, neutral).
   - Detect intent (e.g., hawkish vs. dovish).

## 2. Requirements

### 2.1 Functional Requirements
- Extract text from HTML and PDF files.
- Detect changes in FOMC statements at the phrase level using spaCy.
- Store historical statements and query for term origins.
- Analyze sentiment and intent in speeches using spaCy with custom rules.
- Generate reports in CSV and HTML formats.

### 2.2 Non-Functional Requirements
- Use Python with spaCy, pdfplumber, BeautifulSoup, pandas, and sqlite3.
- Process a pair of FOMC statements or a single speech in <10 seconds.
- Store data in a lightweight SQLite database.

## 3. Implementation Approach
- **Text Extraction**: Use pdfplumber for PDFs and BeautifulSoup for HTML.
- **Change Detection**: Use spaCy for phrase-level diffing and historical context.
- **Sentiment and Intent Analysis**: Use spaCy with custom rules for intent detection.
- **Data Storage**: Use SQLite to store historical statements and term introductions.
- **Output**: Use pandas to generate CSV and HTML reports.

## 4. Implementation Plan

### 4.1 Tech Stack
- Python 3.8+
- spaCy (en_core_web_sm model)
- pdfplumber
- BeautifulSoup4
- pandas
- sqlite3

### 4.2 Milestones
1. **Setup**:
   - Install dependencies.
   - Download spaCy model.
2. **Text Extraction**:
   - Implement functions to extract text from PDFs and HTML.
3. **FOMC Change Tracker**:
   - Use spaCy to tokenize and diff statements.
   - Store statements in SQLite.
   - Query for historical term introductions.
4. **Speech Analysis**:
   - Define custom rules for intent detection.
   - Use spaCy for sentiment analysis.
5. **Output Generation**:
   - Use pandas to create CSV and HTML reports.

### 4.3 Sample Code Structure
```python
import spacy
import pdfplumber
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def extract_text(file_path, file_type):
    if file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages)
    elif file_type == "html":
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()

def track_changes(prev_text, curr_text, db_conn):
    prev_doc = nlp(prev_text)
    curr_doc = nlp(curr_text)
    changes = []
    for prev_sent, curr_sent in zip(prev_doc.sents, curr_doc.sents):
        if prev_sent.text != curr_sent.text:
            changes.append({"prev": prev_sent.text, "curr": curr_sent.text})
    cursor = db_conn.cursor()
    for change in changes:
        term = change["curr"].split()[0]  # Simplified
        cursor.execute("SELECT date FROM history WHERE term=?", (term,))
        change["introduced"] = cursor.fetchone()
    return changes

def analyze_speech(text):
    doc = nlp(text)
    sentiment = sum(token.sentiment for token in doc if token.sentiment) / len(doc)
    intent = "hawkish" if "tightening" in text else "dovish"  # Simplified
    return {"sentiment": sentiment, "intent": intent}

# Main execution
db = sqlite3.connect("fed_history.db")
fomc_changes = track_changes(extract_text("prev.pdf", "pdf"), extract_text("curr.pdf", "pdf"), db)
speech_analysis = analyze_speech(extract_text("speech.html", "html"))
pd.DataFrame(fomc_changes).to_csv("fomc_changes.csv")
print(speech_analysis)