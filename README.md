# Iceberg: Federal Reserve Text Change Tracker

Iceberg is a tool for tracking changes in Federal Reserve statements and analyzing sentiment in speeches. It allows users to analyze individual documents or compare two documents to identify changes, sentiment shifts, and intent changes.

## Features

- **Text Extraction**: Extract text from PDF and HTML files
- **URL Analysis**: Directly analyze HTML content from URLs
- **Change Detection**: Detect changes in FOMC statements at the phrase and sentence level
- **Sentiment Analysis**: Analyze sentiment in speeches and statements
- **Intent Detection**: Detect intent (hawkish vs. dovish) in Fed communications
- **Comparison**: Compare two documents to identify changes and similarities
- **Reporting**: Generate reports in text, CSV, and HTML formats
- **Text Cleaning**: Automatic removal of boilerplate content for cleaner analysis

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

   ```bash
   git clone <repository-url>
   cd iceberg
   ```

2. Install the required dependencies (creation of virtual environment recommended):

   ```bash
   pip install -r requirements.txt
   ```

3. Required NLTK resources will be downloaded automatically when running the application for the first time.

## Usage

### Analyzing a Single Document or URL

```bash
python main.py analyze <file_path_or_url> --output-format [text|csv|html] --save-path [optional_path]
```

Examples:

```bash
# Analyze a local file
python main.py analyze data/raw/fomc_statement_2023_03_22.pdf

# Analyze content from a URL
python main.py analyze https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm
```

### Comparing Two Documents or URLs

```bash
python main.py compare <file_path1_or_url1> <file_path2_or_url2> --output-format [text|csv|html] --save-path [optional_path]
```

Examples:

```bash
# Compare two local files
python main.py compare data/raw/fomc_statement_2023_01_31.pdf data/raw/fomc_statement_2023_03_22.pdf

# Compare two URLs (Federal Reserve press releases)
python main.py compare https://www.federalreserve.gov/newsevents/pressreleases/monetary20250129a.htm https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm
```

## Development

### Running Tests

```bash
pytest tests/
```

Or run a specific test:

```bash
pytest tests/test_analysis.py
```

## Dependencies

The project uses Python 3.13 with the following key dependencies:

- **NLTK**: For natural language processing and text analysis
- **requests**: For fetching content from URLs
- **BeautifulSoup**: For HTML parsing
- **pandas**: For data manipulation and reporting
- **scikit-learn**: For TF-IDF vectorization and similarity calculations

## Recent Improvements

- Added support for direct URL analysis of Federal Reserve statements
- Improved text cleaning to remove boilerplate content
- Enhanced sentence-level change detection to better highlight significant changes
- Added automatic saving of cleaned text files for debugging and analysis
- Fixed reporting issues to better display important policy changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
