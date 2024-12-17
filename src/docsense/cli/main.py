# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
Command Line Interface for DocSense.

This module provides the command-line interface for DocSense, offering three main commands:
- index: Create a vector index from documents in a directory
- ask: Ask a question and get an answer based on indexed documents
- daemon: Run DocSense in daemon mode for faster consecutive queries

The CLI is built using Typer and Rich libraries for better user experience.
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .. import DEFAULT_INDEX_PATH, ask_question, create_index, get_docsense

app = typer.Typer(help="DocSense: An intelligent document assistant powered by Qwen")
console = Console()


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable way.

    Args:
        seconds: Time duration in seconds

    Returns:
        A formatted string showing minutes and seconds or just seconds
        if duration is less than a minute
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes} minutes {seconds:.2f} seconds"


@app.command()
def index(
    path: Path,
    model_name: str = "Qwen/Qwen2-7B",
    device: str = "cuda",
    index_path: Optional[Path] = DEFAULT_INDEX_PATH,
):
    """Create document index from the specified path."""
    try:
        console.print(Panel("Creating document index...", title="DocSense"))
        if index_path == DEFAULT_INDEX_PATH:
            console.print(f"[blue]Using default index path: {DEFAULT_INDEX_PATH}[/blue]")
        start_time = time.time()

        create_index(str(path), model_name=model_name, device=device, index_path=str(index_path))

        elapsed_time = time.time() - start_time
        console.print("[green]Index created successfully![/green]")
        console.print(f"[blue]Index saved to: {index_path}[/blue]")
        console.print(f"[yellow]Time taken: {format_time(elapsed_time)}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error creating index: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command()
def ask(
    question: str,
    model_name: str = "Qwen/Qwen2-7B",
    device: str = "cuda",
    index_path: Optional[Path] = DEFAULT_INDEX_PATH,
):
    """Ask a question and get an answer."""
    try:
        if not isinstance(index_path, Path) or not index_path.exists():
            console.print("[red]No index found. Please create an index first using 'docsense index'[/red]")
            raise typer.Exit(1)

        console.print(Panel("Processing your question...", title="DocSense"))
        start_time = time.time()

        response = ask_question(question, model_name=model_name, device=device, index_path=str(index_path))

        elapsed_time = time.time() - start_time
        console.print(
            Panel(
                response["answer"],
                title="Answer",
            )
        )
        console.print(f"[yellow]Time taken: {format_time(elapsed_time)}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error processing question: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command()
def daemon(
    model_name: str = "Qwen/Qwen2-7B",
    device: str = "cuda",
    index_path: Optional[Path] = DEFAULT_INDEX_PATH,
):
    """Start DocSense in daemon mode for faster responses."""
    try:
        start_time = time.time()
        ds = get_docsense(model_name=model_name, device=device, index_path=str(index_path))
        load_time = time.time() - start_time
        console.print("[green]DocSense daemon is ready![/green]")
        console.print(f"[yellow]Model load time: {format_time(load_time)}[/yellow]")

        while True:
            question = input("Ask a question (or 'exit' to quit): ")
            if question.lower() == "exit":
                break

            start_time = time.time()
            response = ds.ask(question)
            elapsed_time = time.time() - start_time

            console.print(
                Panel(
                    response["answer"],
                    title="Answer",
                )
            )
            console.print(f"[yellow]Time taken: {format_time(elapsed_time)}[/yellow]\n")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
