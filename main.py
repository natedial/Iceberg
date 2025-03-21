#!/usr/bin/env python3
"""
Iceberg: Federal Reserve Text Change Tracker

A tool for tracking changes in FOMC statements and analyzing sentiment in speeches.
"""

import os
import sys
import typer
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse

# Import modules
from modules.ingestion import ingest_file, is_url
from modules.preprocessing import preprocess_text
from modules.analysis import analyze_text, compare_texts
from modules.reporting import generate_report

# Create Typer app
app = typer.Typer(
    name="iceberg",
    help="A tool for tracking changes in Federal Reserve statements and analyzing sentiment in speeches.",
    add_completion=False,
)


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to the file to analyze or URL for HTML content"),
    output_format: str = typer.Option("text", help="Output format (text, csv, html)"),
    save_path: Optional[Path] = typer.Option(None, help="Path to save the analysis results"),
):
    """
    Analyze a single Federal Reserve statement or speech.
    """
    typer.echo(f"Analyzing file or URL: {file_path}")
    
    # Determine if input is a URL or file path
    is_url_input = is_url(file_path)
    
    if is_url_input:
        # If it's a URL, assume it's HTML
        file_type = "html"
    else:
        # Convert to Path object for local files
        file_path_obj = Path(file_path)
        
        # Check if file exists for local files
        if not file_path_obj.exists():
            typer.echo(f"Error: File {file_path} does not exist.", err=True)
            raise typer.Exit(code=1)
        
        # Determine file type
        file_type = "pdf" if file_path_obj.suffix.lower() == ".pdf" else "html"
    
    # Ingest file
    typer.echo("Ingesting file or URL...")
    text = ingest_file(file_path, file_type)
    
    # Preprocess text
    typer.echo("Preprocessing text...")
    processed_text = preprocess_text(text)
    
    # Analyze text
    typer.echo("Analyzing text...")
    analysis_results = analyze_text(processed_text)
    
    # Generate report
    typer.echo("Generating report...")
    report = generate_report(analysis_results, output_format)
    
    # Save or display report
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        typer.echo(f"Report saved to {save_path}")
    else:
        typer.echo("\nAnalysis Results:")
        typer.echo(report)


@app.command()
def compare(
    file_path1: str = typer.Argument(..., help="Path to the first file or URL"),
    file_path2: str = typer.Argument(..., help="Path to the second file or URL"),
    output_format: str = typer.Option("text", help="Output format (text, csv, html)"),
    save_path: Optional[Path] = typer.Option(None, help="Path to save the comparison results"),
):
    """
    Compare two Federal Reserve statements to track changes.
    """
    typer.echo(f"Comparing files or URLs: {file_path1} and {file_path2}")
    
    # Process first input
    is_url_input1 = is_url(file_path1)
    if is_url_input1:
        file_type1 = "html"
    else:
        file_path_obj1 = Path(file_path1)
        if not file_path_obj1.exists():
            typer.echo(f"Error: File {file_path1} does not exist.", err=True)
            raise typer.Exit(code=1)
        file_type1 = "pdf" if file_path_obj1.suffix.lower() == ".pdf" else "html"
    
    # Process second input
    is_url_input2 = is_url(file_path2)
    if is_url_input2:
        file_type2 = "html"
    else:
        file_path_obj2 = Path(file_path2)
        if not file_path_obj2.exists():
            typer.echo(f"Error: File {file_path2} does not exist.", err=True)
            raise typer.Exit(code=1)
        file_type2 = "pdf" if file_path_obj2.suffix.lower() == ".pdf" else "html"
    
    # Ingest files
    typer.echo("Ingesting files or URLs...")
    text1 = ingest_file(file_path1, file_type1)
    text2 = ingest_file(file_path2, file_type2)
    
    # Preprocess texts
    typer.echo("Preprocessing texts...")
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Compare texts
    typer.echo("Comparing texts...")
    comparison_results = compare_texts(processed_text1, processed_text2)
    
    # Generate report
    typer.echo("Generating report...")
    report = generate_report(comparison_results, output_format)
    
    # Save or display report
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        typer.echo(f"Report saved to {save_path}")
    else:
        typer.echo("\nComparison Results:")
        typer.echo(report)


if __name__ == "__main__":
    app()
