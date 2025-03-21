"""
Reporting Module

This module handles the generation of reports in various formats (text, CSV, HTML)
based on analysis results.
"""

import logging
import json
import pandas as pd
from typing import Dict, List, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_report(
    analysis_results: Dict[str, Any],
    output_format: str = "text"
) -> str:
    """
    Generate a report based on analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        output_format: Format of the report ('text', 'csv', 'html')
        
    Returns:
        str: Generated report as a string
    """
    logger.info(f"Generating {output_format} report...")
    
    if "similarity_scores" in analysis_results:
        # This is a comparison report
        return generate_comparison_report(analysis_results, output_format)
    else:
        # This is a single document analysis report
        return generate_analysis_report(analysis_results, output_format)


def generate_analysis_report(
    analysis_results: Dict[str, Any],
    output_format: str = "text"
) -> str:
    """
    Generate a report for a single document analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        output_format: Format of the report ('text', 'csv', 'html')
        
    Returns:
        str: Generated report as a string
    """
    if output_format == "text":
        return generate_text_analysis_report(analysis_results)
    elif output_format == "csv":
        return generate_csv_analysis_report(analysis_results)
    elif output_format == "html":
        return generate_html_analysis_report(analysis_results)
    else:
        logger.warning(f"Unsupported output format: {output_format}. Defaulting to text.")
        return generate_text_analysis_report(analysis_results)


def generate_comparison_report(
    comparison_results: Dict[str, Any],
    output_format: str = "text"
) -> str:
    """
    Generate a report for a comparison between two documents.
    
    Args:
        comparison_results: Dictionary containing comparison results
        output_format: Format of the report ('text', 'csv', 'html')
        
    Returns:
        str: Generated report as a string
    """
    if output_format == "text":
        return generate_text_comparison_report(comparison_results)
    elif output_format == "csv":
        return generate_csv_comparison_report(comparison_results)
    elif output_format == "html":
        return generate_html_comparison_report(comparison_results)
    else:
        logger.warning(f"Unsupported output format: {output_format}. Defaulting to text.")
        return generate_text_comparison_report(comparison_results)


def generate_text_analysis_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a text report for a single document analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        str: Generated text report
    """
    report = []
    
    # Add sentiment analysis
    sentiment = analysis_results["sentiment"]
    report.append("SENTIMENT ANALYSIS")
    report.append("=" * 20)
    report.append(f"Positive: {sentiment['positive']:.2f}")
    report.append(f"Negative: {sentiment['negative']:.2f}")
    report.append(f"Neutral: {sentiment['neutral']:.2f}")
    report.append(f"Compound: {sentiment['compound']:.2f}")
    report.append("")
    
    # Add intent analysis
    intent = analysis_results["intent"]
    report.append("INTENT ANALYSIS")
    report.append("=" * 20)
    report.append(f"Overall Intent: {intent['overall_intent'].upper()}")
    report.append(f"Hawkish Score: {intent['hawkish_score']:.2f}")
    report.append(f"Dovish Score: {intent['dovish_score']:.2f}")
    if 'hawkish_terms_found' in intent:
        report.append(f"Hawkish Terms Found: {intent['hawkish_terms_found']}")
    if 'dovish_terms_found' in intent:
        report.append(f"Dovish Terms Found: {intent['dovish_terms_found']}")
    report.append("")
    
    # Add key phrases
    key_phrases = analysis_results["key_phrases"]
    report.append("KEY PHRASES")
    report.append("=" * 20)
    for i, phrase in enumerate(key_phrases, 1):
        report.append(f"{i}. {phrase}")
    report.append("")
    
    # Add metadata
    metadata = analysis_results["metadata"]
    report.append("DOCUMENT METADATA")
    report.append("=" * 20)
    report.append(f"Token Count: {metadata['token_count']}")
    report.append(f"Sentence Count: {metadata['sentence_count']}")
    report.append(f"Average Sentence Length: {metadata['average_sentence_length']:.2f} tokens")
    
    return "\n".join(report)


def generate_text_comparison_report(comparison_results: Dict[str, Any]) -> str:
    """
    Generate a text report for a comparison between two documents.
    
    Args:
        comparison_results: Dictionary containing comparison results
        
    Returns:
        str: Generated text report
    """
    report = []
    
    # Add similarity scores
    similarity = comparison_results["similarity_scores"]
    report.append("SIMILARITY ANALYSIS")
    report.append("=" * 20)
    report.append(f"TF-IDF Similarity: {similarity['tfidf_similarity']:.2f}")
    report.append(f"Jaccard Similarity: {similarity['jaccard_similarity']:.2f}")
    report.append(f"Overall Similarity: {similarity['overall_similarity']:.2f}")
    report.append("")
    
    # Add sentiment comparison
    sentiment_comp = comparison_results["sentiment_comparison"]
    report.append("SENTIMENT COMPARISON")
    report.append("=" * 20)
    report.append(f"Document 1 Compound: {sentiment_comp['document1']['compound']:.2f}")
    report.append(f"Document 2 Compound: {sentiment_comp['document2']['compound']:.2f}")
    report.append(f"Change: {sentiment_comp['change']:.2f}")
    report.append("")
    
    # Add intent comparison
    intent_comp = comparison_results["intent_comparison"]
    report.append("INTENT COMPARISON")
    report.append("=" * 20)
    report.append(f"Document 1 Intent: {intent_comp['document1']['overall_intent'].upper()}")
    report.append(f"Document 2 Intent: {intent_comp['document2']['overall_intent'].upper()}")
    report.append(f"Change Direction: {intent_comp['change']}")
    report.append("")
    
    # Add phrase changes
    phrase_changes = comparison_results["phrase_changes"]
    report.append("PHRASE CHANGES")
    report.append("=" * 20)
    report.append(f"Additions: {len(phrase_changes['additions'])}")
    report.append(f"Deletions: {len(phrase_changes['deletions'])}")
    report.append("")
    
    # Add top 5 additions
    report.append("TOP ADDITIONS:")
    for i, addition in enumerate(phrase_changes["additions"][:5], 1):
        report.append(f"{i}. + {addition}")
    report.append("")
    
    # Add top 5 deletions
    report.append("TOP DELETIONS:")
    for i, deletion in enumerate(phrase_changes["deletions"][:5], 1):
        report.append(f"{i}. - {deletion}")
    report.append("")
    
    # Add sentence changes
    sentence_changes = comparison_results["sentence_changes"]
    report.append("SENTENCE CHANGES")
    report.append("=" * 20)
    
    # Filter for only significant changes (not unchanged)
    significant_changes = [change for change in sentence_changes if change["type"] != "unchanged"]
    report.append(f"Total Changes: {len(significant_changes)}")
    
    # Add all significant sentence changes
    for i, change in enumerate(significant_changes[:10], 1):
        report.append(f"\nChange {i}: {change['type']}")
        if change["type"] == "replace" or change["type"] == "modified":
            report.append(f"Old: {change['old_sentence']}")
            report.append(f"New: {change['new_sentence']}")
        elif change["type"] == "deleted":
            report.append(f"Deleted: {change['old_sentence']}")
        elif change["type"] == "added":
            report.append(f"Added: {change['new_sentence']}")
    
    return "\n".join(report)


def generate_csv_analysis_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a CSV report for a single document analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        str: Generated CSV report
    """
    # Create a DataFrame for the report
    data = {
        "Category": [],
        "Metric": [],
        "Value": []
    }
    
    # Add sentiment analysis
    sentiment = analysis_results["sentiment"]
    data["Category"].extend(["Sentiment"] * 4)
    data["Metric"].extend(["Positive", "Negative", "Neutral", "Compound"])
    data["Value"].extend([
        f"{sentiment['positive']:.2f}",
        f"{sentiment['negative']:.2f}",
        f"{sentiment['neutral']:.2f}",
        f"{sentiment['compound']:.2f}"
    ])
    
    # Add intent analysis
    intent = analysis_results["intent"]
    data["Category"].extend(["Intent"] * 3)
    data["Metric"].extend(["Overall Intent", "Hawkish Score", "Dovish Score"])
    data["Value"].extend([
        intent["overall_intent"].upper(),
        f"{intent['hawkish_score']:.2f}",
        f"{intent['dovish_score']:.2f}"
    ])
    
    # Add hawkish and dovish terms if available
    if 'hawkish_terms_found' in intent:
        data["Category"].append("Intent")
        data["Metric"].append("Hawkish Terms Found")
        data["Value"].append(str(intent["hawkish_terms_found"]))
    
    if 'dovish_terms_found' in intent:
        data["Category"].append("Intent")
        data["Metric"].append("Dovish Terms Found")
        data["Value"].append(str(intent["dovish_terms_found"]))
    
    # Add key phrases
    key_phrases = analysis_results["key_phrases"]
    for i, phrase in enumerate(key_phrases, 1):
        data["Category"].append("Key Phrases")
        data["Metric"].append(f"Phrase {i}")
        data["Value"].append(phrase)
    
    # Add metadata
    metadata = analysis_results["metadata"]
    data["Category"].extend(["Metadata"] * 3)
    data["Metric"].extend(["Token Count", "Sentence Count", "Average Sentence Length"])
    data["Value"].extend([
        str(metadata["token_count"]),
        str(metadata["sentence_count"]),
        f"{metadata['average_sentence_length']:.2f}"
    ])
    
    # Create DataFrame and convert to CSV
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def generate_csv_comparison_report(comparison_results: Dict[str, Any]) -> str:
    """
    Generate a CSV report for a comparison between two documents.
    
    Args:
        comparison_results: Dictionary containing comparison results
        
    Returns:
        str: Generated CSV report
    """
    # Create DataFrames for different sections
    
    # Similarity DataFrame
    similarity_df = pd.DataFrame([comparison_results["similarity_scores"]])
    
    # Sentiment Comparison DataFrame
    sentiment_comp = comparison_results["sentiment_comparison"]
    sentiment_df = pd.DataFrame({
        "document": ["Document 1", "Document 2"],
        "compound_score": [
            sentiment_comp["document1"]["compound"],
            sentiment_comp["document2"]["compound"]
        ]
    })
    
    # Intent Comparison DataFrame
    intent_comp = comparison_results["intent_comparison"]
    intent_df = pd.DataFrame({
        "document": ["Document 1", "Document 2"],
        "overall_intent": [
            intent_comp["document1"]["overall_intent"],
            intent_comp["document2"]["overall_intent"]
        ],
        "hawkish_score": [
            intent_comp["document1"]["hawkish_score"],
            intent_comp["document2"]["hawkish_score"]
        ],
        "dovish_score": [
            intent_comp["document1"]["dovish_score"],
            intent_comp["document2"]["dovish_score"]
        ]
    })
    
    # Phrase Changes DataFrame
    phrase_changes = comparison_results["phrase_changes"]
    additions_df = pd.DataFrame({
        "type": ["addition"] * len(phrase_changes["additions"]),
        "text": phrase_changes["additions"]
    })
    deletions_df = pd.DataFrame({
        "type": ["deletion"] * len(phrase_changes["deletions"]),
        "text": phrase_changes["deletions"]
    })
    phrase_changes_df = pd.concat([additions_df, deletions_df])
    
    # Sentence Changes DataFrame
    sentence_changes_df = pd.DataFrame(comparison_results["sentence_changes"])
    
    # Combine all DataFrames into a single CSV
    csv_parts = [
        "SIMILARITY ANALYSIS",
        similarity_df.to_csv(index=False),
        "",
        "SENTIMENT COMPARISON",
        sentiment_df.to_csv(index=False),
        "",
        "INTENT COMPARISON",
        intent_df.to_csv(index=False),
        "",
        "PHRASE CHANGES",
        phrase_changes_df.to_csv(index=False),
        "",
        "SENTENCE CHANGES",
        sentence_changes_df.to_csv(index=False)
    ]
    
    return "\n".join(csv_parts)


def generate_html_analysis_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate an HTML report for a single document analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        str: Generated HTML report
    """
    # Start HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iceberg Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; }
            .value { color: #2c3e50; }
            .positive { color: green; }
            .negative { color: red; }
            .neutral { color: gray; }
        </style>
    </head>
    <body>
        <h1>Iceberg Analysis Report</h1>
    """
    
    # Add sentiment analysis
    sentiment = analysis_results["sentiment"]
    html += """
        <h2>Sentiment Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td class="metric">Positive</td><td class="value positive">{:.2f}</td></tr>
            <tr><td class="metric">Negative</td><td class="value negative">{:.2f}</td></tr>
            <tr><td class="metric">Neutral</td><td class="value neutral">{:.2f}</td></tr>
            <tr><td class="metric">Compound</td><td class="value">{:.2f}</td></tr>
        </table>
    """.format(
        sentiment["positive"],
        sentiment["negative"],
        sentiment["neutral"],
        sentiment["compound"]
    )
    
    # Add intent analysis
    intent = analysis_results["intent"]
    html += """
        <h2>Intent Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td class="metric">Overall Intent</td><td class="value">{}</td></tr>
            <tr><td class="metric">Hawkish Score</td><td class="value">{:.2f}</td></tr>
            <tr><td class="metric">Dovish Score</td><td class="value">{:.2f}</td></tr>
    """.format(
        intent["overall_intent"].upper(),
        intent["hawkish_score"],
        intent["dovish_score"]
    )
    
    # Add hawkish and dovish terms if available
    if 'hawkish_terms_found' in intent:
        html += """
            <tr><td class="metric">Hawkish Terms Found</td><td class="value">{}</td></tr>
        """.format(intent["hawkish_terms_found"])
    
    if 'dovish_terms_found' in intent:
        html += """
            <tr><td class="metric">Dovish Terms Found</td><td class="value">{}</td></tr>
        """.format(intent["dovish_terms_found"])
    
    html += """
        </table>
    """
    
    # Add key phrases
    key_phrases = analysis_results["key_phrases"]
    html += """
        <h2>Key Phrases</h2>
        <table>
            <tr><th>#</th><th>Phrase</th></tr>
    """
    
    for i, phrase in enumerate(key_phrases, 1):
        html += """
            <tr><td>{}</td><td>{}</td></tr>
        """.format(i, phrase)
    
    html += """
        </table>
    """
    
    # Add metadata
    metadata = analysis_results["metadata"]
    html += """
        <h2>Document Metadata</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td class="metric">Token Count</td><td class="value">{}</td></tr>
            <tr><td class="metric">Sentence Count</td><td class="value">{}</td></tr>
            <tr><td class="metric">Average Sentence Length</td><td class="value">{:.2f} tokens</td></tr>
        </table>
    """.format(
        metadata["token_count"],
        metadata["sentence_count"],
        metadata["average_sentence_length"]
    )
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html


def generate_html_comparison_report(comparison_results: Dict[str, Any]) -> str:
    """
    Generate an HTML report for a comparison between two documents.
    
    Args:
        comparison_results: Dictionary containing comparison results
        
    Returns:
        str: Generated HTML report
    """
    # Create DataFrames for different sections
    
    # Similarity DataFrame
    similarity_df = pd.DataFrame([comparison_results["similarity_scores"]])
    
    # Sentiment Comparison DataFrame
    sentiment_comp = comparison_results["sentiment_comparison"]
    sentiment_df = pd.DataFrame({
        "document": ["Document 1", "Document 2"],
        "compound_score": [
            sentiment_comp["document1"]["compound"],
            sentiment_comp["document2"]["compound"]
        ]
    })
    
    # Intent Comparison DataFrame
    intent_comp = comparison_results["intent_comparison"]
    intent_df = pd.DataFrame({
        "document": ["Document 1", "Document 2"],
        "overall_intent": [
            intent_comp["document1"]["overall_intent"],
            intent_comp["document2"]["overall_intent"]
        ],
        "hawkish_score": [
            intent_comp["document1"]["hawkish_score"],
            intent_comp["document2"]["hawkish_score"]
        ],
        "dovish_score": [
            intent_comp["document1"]["dovish_score"],
            intent_comp["document2"]["dovish_score"]
        ]
    })
    
    # Phrase Changes DataFrame
    phrase_changes = comparison_results["phrase_changes"]
    additions_df = pd.DataFrame({
        "type": ["addition"] * len(phrase_changes["additions"]),
        "text": phrase_changes["additions"]
    })
    deletions_df = pd.DataFrame({
        "type": ["deletion"] * len(phrase_changes["deletions"]),
        "text": phrase_changes["deletions"]
    })
    phrase_changes_df = pd.concat([additions_df, deletions_df])
    
    # Sentence Changes DataFrame
    sentence_changes_df = pd.DataFrame(comparison_results["sentence_changes"])
    
    # Generate HTML
    html_parts = [
        "<html>",
        "<head>",
        "<title>Text Comparison Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #3498db; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f2f2f2; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        ".addition { background-color: #dff0d8; color: #3c763d; }",
        ".deletion { background-color: #f2dede; color: #a94442; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Text Comparison Report</h1>",
        
        "<h2>Similarity Analysis</h2>",
        similarity_df.to_html(index=False),
        
        "<h2>Sentiment Comparison</h2>",
        sentiment_df.to_html(index=False),
        
        "<h2>Intent Comparison</h2>",
        intent_df.to_html(index=False),
        
        "<h2>Phrase Changes</h2>",
        phrase_changes_df.to_html(index=False),
        
        "<h2>Sentence Changes</h2>",
        sentence_changes_df.to_html(index=False),
        
        "</body>",
        "</html>"
    ]
    
    return "\n".join(html_parts)
