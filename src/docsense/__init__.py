# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
DocSense: An intelligent document assistant powered by Qwen.
"""

__version__ = "0.1.0"

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .indexer import DocumentLoader, VectorStore
from .models import EmbeddingModel, LLMModel

DEFAULT_INDEX_PATH = Path(os.path.expanduser("~/.docsense/index"))


class DocSense:
    """Main class for document processing and question answering."""

    _instance = None

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B",
        embedding_model: Optional[str] = None,
        device: str = "cuda",
        index_path: Optional[str] = None,
        use_gpu_faiss: bool = True,
    ):
        """
        Initialize DocSense instance.

        Args:
            model_name: Name of the Qwen model to use
            embedding_model: Name of the embedding model (defaults to model_name if None)
            device: Device to run the model on ('cuda' or 'cpu')
            index_path: Path to store/load the vector index (defaults to ~/.docsense/index)
            use_gpu_faiss: Whether to use GPU for FAISS operations
        """
        self.model_name = model_name
        self.embedding_model = embedding_model or model_name
        self.device = device
        self.index_path = Path(index_path) if index_path else Path(DEFAULT_INDEX_PATH)

        # Create index directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._init_models()

        # Initialize document processing
        self.doc_loader = DocumentLoader()

        # Initialize vector store
        if self._embedding_model:
            self._vector_store = VectorStore(
                dimension=self._embedding_model.get_embedding_dim(),
                index_path=str(self.index_path),
                use_gpu=use_gpu_faiss,
            )
        else:
            self._vector_store = None

        self._response_cache = {}  # Simple cache for responses

    def _init_models(self):
        """Initialize the models."""
        # Initialize embedding model
        self._embedding_model = EmbeddingModel(model_name=self.embedding_model, device=self.device)

        # Initialize LLM
        self._llm = LLMModel(model_name=self.model_name, device=self.device)

    def index_documents(self, doc_path: str) -> None:
        """
        Index documents from the specified path.

        Args:
            doc_path: Path to the documents directory
        """
        try:
            print(f"\nStarting document indexing from: {doc_path}")

            # Load documents as iterator
            document_iterator = self.doc_loader.load_directory(doc_path)

            # Convert to list only when needed for length check and processing
            documents = list(document_iterator)

            print(f"\nFound {len(documents)} documents")
            if not documents:
                print(f"No documents were found in {doc_path}")
                raise ValueError(f"No documents found in {doc_path}")

            print(f"\nProcessing {len(documents)} documents...")

            # Generate embeddings (texts는 메모리 효율을 위해 generator로)
            texts = (doc.content for doc in documents)
            texts_list = list(texts)
            print(f"Generating embeddings for {len(texts_list)} texts...")

            try:
                embeddings = self._embedding_model.encode(texts_list)
                print(f"Generated embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
                raise

            # Add to vector store
            try:
                self._vector_store.add_documents(documents, embeddings)
                print("Successfully added documents to vector store")
            except Exception as e:
                print(f"Error adding to vector store: {str(e)}")
                raise

        except Exception as e:
            print(f"\nError during document indexing: {str(e)}")
            print(f"Type: {type(e).__name__}")
            raise

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a question based on the indexed documents.

        Args:
            question: User question

        Returns:
            Dict containing answer and relevant source information
        """
        # Check cache
        cache_key = question.strip().lower()
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        if not self._vector_store:
            raise RuntimeError("No documents have been indexed yet")

        # Generate question embedding
        question_embedding = self._embedding_model.encode(question)

        # Search for relevant documents
        relevant_docs = self._vector_store.search(question_embedding)

        if not relevant_docs:
            return {"answer": "I couldn't find any relevant information to answer your question.", "sources": []}

        # Prepare context and sources with scores
        context = []
        sources = []
        for doc, score in relevant_docs:
            context.append(doc.content)
            sources.append(
                {
                    "path": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown"),
                    "relevance_score": float(score),  # numpy float를 Python float로 변환
                }
            )

        # Generate answer
        response = self._llm.generate(question, context=context)

        # Cache response
        response = {
            "answer": response["answer"],
            "sources": sources,  # 모든 소스 정보를 포함
            "metadata": {"prompt": response["prompt"], "generation_config": response["generation_config"]},
        }

        return response


# Convenience functions
def get_docsense(**kwargs) -> DocSense:
    return DocSense.get_instance(**kwargs)


def create_index(doc_path: str, **kwargs) -> None:
    ds = get_docsense(**kwargs)
    ds.index_documents(doc_path)


def ask_question(question: str, **kwargs) -> Dict[str, Any]:
    ds = get_docsense(**kwargs)
    return ds.ask(question)
