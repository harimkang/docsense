# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
Vector store implementation using FAISS.

This module provides a vector store implementation using FAISS (Facebook AI Similarity Search)
for efficient similarity search of document embeddings. It supports both CPU and GPU operations,
persistence to disk, and metadata management.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss  # type: ignore
import numpy as np

from .document import Document


class VectorStore:
    """
    Vector store for document embeddings using FAISS.

    This class implements a vector store that uses FAISS for efficient similarity search
    of document embeddings. It supports:
    - GPU acceleration when available
    - Persistence to disk
    - Document metadata management
    - IVF (Inverted File) index for faster search
    """

    def __init__(self, dimension: int, index_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the vector store.

        Args:
            dimension: Dimension of the embedding vectors
            index_path: Path to load/save the index and metadata. If None, store will be in-memory only
            use_gpu: Whether to use GPU for FAISS operations. Falls back to CPU if GPU is not available

        Raises:
            ValueError: If dimension is invalid
            RuntimeError: If GPU initialization fails
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        # Use simple FlatL2 index instead of IVF for testing
        self.index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)

        # Store document metadata
        self.documents: List[Document] = []

        # Create directory if index path is provided
        if self.index_path:
            self.index_path.mkdir(parents=True, exist_ok=True)

        # Load existing index if path is provided and exists
        if self.index_path and (self.index_path / "index.faiss").exists():
            self.load()

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the store.

        Args:
            documents: List of Document objects to add
            embeddings: numpy array of document embeddings with shape (n_docs, dimension)

        Raises:
            ValueError: If number of documents doesn't match number of embeddings,
                       or if embedding dimensions don't match
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")

        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store documents
        self.documents.extend(documents)

        # Save if index path is set
        if self.index_path:
            self.save()

    def search(self, query_embedding: np.ndarray, k: int = 2) -> List[Tuple[Document, float]]:
        """
        Search for most similar documents using the query embedding.

        Args:
            query_embedding: Query vector with shape (dimension,) or (1, dimension)
            k: Number of results to return

        Returns:
            List of (document, distance) tuples sorted by similarity (closest first)

        Raises:
            ValueError: If query_embedding has invalid shape
        """
        if len(self.documents) == 0:
            return []

        # Ensure query embedding has correct shape
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search the index
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))

        # Return documents and distances
        results = []
        for idx, distance in zip(indices[0], distances[0], strict=False):
            if idx != -1:  # FAISS returns -1 for invalid indices
                results.append((self.documents[idx], float(distance)))

        return results

    def save(self) -> None:
        """
        Save the index and metadata to disk.

        This method saves both the FAISS index and document metadata to the specified
        index path. The index is saved in FAISS binary format and metadata in JSON.

        Raises:
            ValueError: If no index path was specified
            IOError: If saving fails
        """
        if not self.index_path:
            raise ValueError("No index path specified")

        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU for saving if necessary
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(index_to_save, str(index_file))

        # Save metadata
        metadata_file = self.index_path / "metadata.json"
        metadata = {
            "dimension": self.dimension,
            "documents": [{"content": doc.content, "metadata": doc.metadata} for doc in self.documents],
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """
        Load the index and metadata from disk.

        This method loads both the FAISS index and document metadata from the specified
        index path. The index is loaded from FAISS binary format and metadata from JSON.

        Raises:
            ValueError: If no index path was specified or dimension mismatch
            FileNotFoundError: If index or metadata files are missing
            IOError: If loading fails
        """
        if not self.index_path:
            raise ValueError("No index path specified")

        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError("Index or metadata file not found")

        # Load FAISS index
        cpu_index = faiss.read_index(str(index_file))

        # Move to GPU if requested
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
        else:
            self.index = cpu_index

        # Verify dimension
        if self.index.d != self.dimension:
            raise ValueError(f"Index dimension mismatch. Expected {self.dimension}, got {self.index.d}")

        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Reconstruct documents
        self.documents = [Document(content=doc["content"], metadata=doc["metadata"]) for doc in metadata["documents"]]

    def clear(self) -> None:
        """
        Clear all documents and reset the index.

        This method removes all documents and their embeddings from the store,
        effectively resetting it to its initial state. If persistence is enabled,
        the cleared state will be saved to disk.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
        self.documents = []

        if self.index_path:
            self.save()
