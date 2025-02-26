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

# Import modules
from modules.ingestion import ingest_file
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
    file_path: Path = typer.Argument(..., help="Path to the file to analyze"),
    output_format: str = typer.Option("text", help="Output format (text, csv, html)"),
    save_path: Optional[Path] = typer.Option(None, help="Path to save the analysis results"),
):
    """
    Analyze a single Federal Reserve statement or speech.
    """
    typer.echo(f"Analyzing file: {file_path}")
    
    # Check if file exists
    if not file_path.exists():
        typer.echo(f"Error: File {file_path} does not exist.", err=True)
        raise typer.Exit(code=1)
    
    # Determine file type
    file_type = "pdf" if file_path.suffix.lower() == ".pdf" else "html"
    
    # Ingest file
    typer.echo("Ingesting file...")
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
    file_path1: Path = typer.Argument(..., help="Path to the first file"),
    file_path2: Path = typer.Argument(..., help="Path to the second file"),
    output_format: str = typer.Option("text", help="Output format (text, csv, html)"),
    save_path: Optional[Path] = typer.Option(None, help="Path to save the comparison results"),
):
    """
    Compare two Federal Reserve statements to track changes.
    """
    typer.echo(f"Comparing files: {file_path1} and {file_path2}")
    
    # Check if files exist
    for path in [file_path1, file_path2]:
        if not path.exists():
            typer.echo(f"Error: File {path} does not exist.", err=True)
            raise typer.Exit(code=1)
    
    # Determine file types
    file_type1 = "pdf" if file_path1.suffix.lower() == ".pdf" else "html"
    file_type2 = "pdf" if file_path2.suffix.lower() == ".pdf" else "html"
    
    # Ingest files
    typer.echo("Ingesting files...")
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
