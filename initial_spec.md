# **Refined PRD: Iceberg (Textual Semantic Analysis Tool)**

## **1. Overview**
Iceberg aims to detect and measure shifts in a speaker’s messaging across multiple speeches. By analyzing linguistic and semantic changes, we capture how content, tone, and word choice evolve over time. The **initial goal** is to produce a **Command-Line Interface (CLI) MVP in Python** that supports **batch processing** (no real-time analysis). Development proceeds in **phases**, with a recommended folder structure to maintain clarity for both human developers and LLM coding agents.

---

## **2. Phases of Development**

### **2.1 Phase 1: Core CLI Setup**

- **Objectives**  
  1. Implement a Python CLI (using `argparse` or Typer).  
  2. Define commands such as `analyze` and `compare`, each with required arguments (e.g., file paths, processing flags).  
  3. Keep external dependencies minimal to enable local execution.

- **Acceptance Criteria**  
  1. CLI accepts at least one speech file path and validates its existence.  
  2. Displays a “Hello World” or simple prompt to confirm CLI readiness.

- **Edge Cases & Considerations**  
  - Handling missing or invalid paths.  
  - Graceful error messages for incorrect command usage.

---

### **2.2 Phase 2: Data Ingestion & Storage**

- **Objectives**  
  1. Accept multiple input formats (e.g., raw text, CSV).  
  2. Validate file format, size, and encoding (UTF-8 assumed as default).  
  3. Organize data by speaker and date in a simple directory structure.

- **Acceptance Criteria**  
  1. Parse incoming files without errors, logging invalid or corrupted files.  
  2. Store parsed data locally with clear directory conventions (e.g., `raw/` for original files, `processed/` for subsequent steps).

- **Edge Cases & Considerations**  
  - Non-UTF-8 encodings or files with unexpected formats.  
  - Large input files requiring memory-friendly parsing.

---

### **2.3 Phase 3: Text Preprocessing**

- **Objectives**  
  1. Clean raw text by removing special characters and HTML tags.  
  2. Tokenize using spaCy.  
  3. Output cleaned and tokenized data into a **dedicated `processed/` folder**.

- **Acceptance Criteria**  
  1. Automatic text cleaning with a log or message confirming success.  
  2. Verified tokenization files stored per speech, named consistently (e.g., `<speaker>_<date>_tokens.json`).

- **Edge Cases & Considerations**  
  - Mixed languages or special symbols that the tokenizer might not handle cleanly.  
  - Incomplete or extremely short text segments.

---

### **2.4 Phase 4: Semantic Analysis**

- **Objectives**  
  1. Implement and configure one or more similarity metrics (e.g., TF-IDF, word embeddings).  
  2. Compare speeches to detect shifts in topics or key phrases.  
  3. Generate a **numeric similarity score** or **difference measure** for each comparison.

- **Acceptance Criteria**  
  1. An analysis module (`analysis.py`) that outputs numeric similarity scores.  
  2. Summaries that highlight top changes between pairs of speeches (e.g., top 5 phrases that differ).

- **Edge Cases & Considerations**  
  - Very short speeches that may not produce reliable similarity scores.  
  - Multiple similarity algorithms (TF-IDF vs. embeddings) requiring synergy or selection logic.

---

### **2.5 Phase 5: Reporting & Output**

- **Objectives**  
  1. Present findings in a **concise CLI report**—either a tabular or text summary.  
  2. List each speech analyzed, its similarity score with other speeches, and key changed phrases.  
  3. Keep output readable, suitable for quick scanning in a terminal.

- **Acceptance Criteria**  
  1. Automatic summary displayed in CLI after each analysis command.  
  2. Clear formatting (e.g., columns or bullet points) that highlight essential metrics.

- **Edge Cases & Considerations**  
  - Large sets of speeches requiring pagination or chunked output.  
  - Ensuring the summary remains readable even for large numeric results.

---

### **2.6 Additional Feature: Fine-Grained Similarity**

- **Objectives**  
  1. Apply diff-based methods to highlight newly added or removed words.  
  2. Label sentences with partial changes (near-duplicates).  
  3. Provide a side-by-side comparison of original vs. altered sections.

- **Acceptance Criteria**  
  1. Automated diff detection triggered for speeches with similarity above a certain threshold (e.g., >80%).  
  2. Clear summary or color-coded diff for subtle textual shifts.

- **Edge Cases & Considerations**  
  - Overly large diffs for lengthy speeches, requiring condensed or summarized output.  
  - Handling special characters or punctuation changes that may not be meaningful shifts.

---

## **3. Recommended Folder Structure**

```plaintext
project-root/
├── ai_docs/             # Project documentation
├── cli/                 # CLI entry points and argument parsing
│   └── main.py
├── data/                # Raw and cleaned speech files
│   ├── raw/
│   └── processed/
├── modules/             # Core modules for analysis
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── analysis.py
│   └── reporting.py
├── tests/               # Test scripts
│   └── test_analysis.py
├── requirements.txt     # Python dependencies
└── README.md            # Project overview and setup steps
