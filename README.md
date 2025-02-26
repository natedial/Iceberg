# Iceberg: Federal Reserve Text Change Tracker

Iceberg is a tool for tracking changes in Federal Reserve statements and analyzing sentiment in speeches. It allows users to analyze individual documents or compare two documents to identify changes, sentiment shifts, and intent changes.

## Features

- **Text Extraction**: Extract text from PDF and HTML files
- **Change Detection**: Detect changes in FOMC statements at the phrase level
- **Sentiment Analysis**: Analyze sentiment in speeches and statements
- **Intent Detection**: Detect intent (hawkish vs. dovish) in Fed communications
- **Comparison**: Compare two documents to identify changes and similarities
- **Reporting**: Generate reports in text, CSV, and HTML formats

## Project Structure

```
├── data/                # Raw and cleaned speech files
│   ├── raw/             # Original extracted text
│   └── processed/       # Processed text data
├── modules/             # Core modules for analysis
│   ├── ingestion.py     # Text extraction from different file formats
│   ├── preprocessing.py # Text cleaning and tokenization
│   ├── analysis.py      # Sentiment, intent, and comparison analysis
│   └── reporting.py     # Report generation in different formats
├── tests/               # Test scripts
│   └── test_analysis.py # Tests for the analysis module
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project overview and setup steps
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd iceberg
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

### Analyzing a Single Document

```
python main.py analyze <file_path> --output-format [text|csv|html] --save-path [optional_path]
```

Example:
```
python main.py analyze data/raw/fomc_statement_2023_03_22.pdf
```

### Comparing Two Documents

```
python main.py compare <file_path1> <file_path2> --output-format [text|csv|html] --save-path [optional_path]
```

Example:
```
python main.py compare data/raw/fomc_statement_2023_01_31.pdf data/raw/fomc_statement_2023_03_22.pdf
```

## Development

### Running Tests

```
pytest tests/
```

Or run a specific test:
```
pytest tests/test_analysis.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses [spaCy](https://spacy.io/) for natural language processing
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [pandas](https://pandas.pydata.org/) for data manipulation and reporting
- [scikit-learn](https://scikit-learn.org/) for text similarity metrics
