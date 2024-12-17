# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
Document processing module for loading and chunking documents.

This module provides functionality for representing documents and splitting them into
manageable chunks while preserving metadata. It includes classes for document
representation and text chunking with configurable overlap.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """
    Represents a document or a chunk of a document.

    A Document object contains the actual text content and associated metadata.
    The metadata can include information like source, timestamps, chunk positions, etc.
    """

    content: str
    metadata: dict

    def __init__(self, content: str, metadata: dict | None = None):
        """
        Initialize a Document object.

        Args:
            content: The text content of the document
            metadata: Optional dictionary containing metadata about the document.
                     If None, an empty dict will be used.
        """
        self.content = content
        self.metadata = metadata or {}


class DocumentChunker:
    """
    Splits documents into smaller chunks with configurable overlap.

    This class provides functionality to split large text documents into smaller,
    overlapping chunks while preserving document metadata. It attempts to split
    at sentence boundaries to maintain context.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk. Default is 1000.
            chunk_overlap: Number of characters to overlap between consecutive chunks
                         to maintain context. Default is 200.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: dict) -> List[Document]:
        """
        Split text into overlapping chunks while preserving metadata.

        The method attempts to split at sentence boundaries to maintain readability
        and context. Each chunk inherits the original document's metadata with
        additional chunk position information.

        Args:
            text: Text content to split into chunks
            metadata: Metadata to attach to each chunk. Will be extended with
                     chunk-specific position information.

        Returns:
            List of Document objects, each containing a chunk of the original text
            and associated metadata.
        """
        if len(text) <= self.chunk_size:
            chunk_metadata = {**metadata, "chunk_start": 0, "chunk_end": len(text)}
            return [Document(content=text, metadata=chunk_metadata)]

        chunks = []
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = min(start + self.chunk_size, len(text))

            # If we're not at the end of the text, try to break at a sentence
            if end < len(text):
                # Look for sentence boundaries (., !, ?)
                last_boundary = end
                for i in range(end - 1, max(start, end - 100), -1):
                    if i < len(text) and text[i] in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
                        last_boundary = i + 1
                        break
                end = last_boundary

            # Create chunk with metadata
            chunk_metadata = {**metadata, "chunk_start": start, "chunk_end": end}
            chunks.append(Document(content=text[start:end].strip(), metadata=chunk_metadata))

            # Move start position for next chunk
            start = max(start + 1, end - self.chunk_overlap)

        return chunks
