from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from docsense import DEFAULT_INDEX_PATH, DocSense, ask_question, create_index, get_docsense
from docsense.indexer import Document


@pytest.fixture
def mock_models():
    """Mock models and their initialization."""
    with (
        patch("docsense.EmbeddingModel") as mock_embedding_cls,
        patch("docsense.LLMModel") as mock_llm_cls,
        patch("docsense.VectorStore") as mock_store_cls,
        patch("docsense.DocumentLoader") as mock_loader_cls,
    ):
        # Configure embedding model mock
        embedding_model = Mock()
        embedding_model.get_embedding_dim.return_value = 768
        embedding_model.encode.return_value = np.random.rand(1, 768)
        mock_embedding_cls.return_value = embedding_model

        # Configure LLM mock
        llm_model = Mock()
        llm_model.generate.return_value = {
            "answer": "Test answer",
            "prompt": "Test prompt",
            "generation_config": {"test": "config"},
        }
        mock_llm_cls.return_value = llm_model

        # Configure vector store mock
        store = Mock()
        store.search.return_value = [
            (Document("Test content", {"source": "test.txt"}), 0.9),
        ]
        mock_store_cls.return_value = store

        # Configure document loader mock
        loader = Mock()
        loader.load_directory.return_value = [
            Document("Test content", {"source": "test.txt"}),
        ]
        mock_loader_cls.return_value = loader

        yield {
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "vector_store": store,
            "doc_loader": loader,
        }


def test_docsense_initialization(mock_models):
    """Test DocSense initialization."""
    ds = DocSense()

    assert ds.model_name == "Qwen/Qwen2-7B"
    assert ds.embedding_model == "Qwen/Qwen2-7B"
    assert ds.index_path == Path(DEFAULT_INDEX_PATH)

    # Check model initialization
    mock_models["embedding_model"].get_embedding_dim.assert_called_once()


def test_docsense_custom_initialization(mock_models, tmp_path):
    """Test DocSense initialization with custom parameters."""
    custom_path = tmp_path / "custom_index"
    ds = DocSense(
        model_name="custom-model",
        embedding_model="custom-embedding",
        device="cpu",
        index_path=str(custom_path),
    )

    assert ds.model_name == "custom-model"
    assert ds.embedding_model == "custom-embedding"
    assert ds.device == "cpu"
    assert ds.index_path == custom_path


def test_index_documents(mock_models):
    """Test document indexing."""
    ds = DocSense()
    ds.index_documents("test/path")

    # Check document loading
    mock_models["doc_loader"].load_directory.assert_called_once_with("test/path")

    # Check embedding generation
    mock_models["embedding_model"].encode.assert_called_once()

    # Check vector store addition
    mock_models["vector_store"].add_documents.assert_called_once()


def test_index_documents_empty(mock_models):
    """Test indexing with no documents."""
    mock_models["doc_loader"].load_directory.return_value = []

    ds = DocSense()
    with pytest.raises(ValueError, match="No documents found"):
        ds.index_documents("empty/path")


def test_ask(mock_models):
    """Test question answering."""
    ds = DocSense()
    response = ds.ask("Test question")

    # Check response format
    assert "answer" in response
    assert "sources" in response
    assert "metadata" in response
    assert isinstance(response["sources"], list)
    assert isinstance(response["metadata"], dict)

    # Verify search and generation
    mock_models["embedding_model"].encode.assert_called_once()
    mock_models["vector_store"].search.assert_called_once()
    mock_models["llm_model"].generate.assert_called_once()


def test_ask_no_results(mock_models):
    """Test question answering with no relevant documents."""
    mock_models["vector_store"].search.return_value = []

    ds = DocSense()
    response = ds.ask("Test question")

    assert "couldn't find any relevant information" in response["answer"]
    assert response["sources"] == []


def test_response_caching(mock_models):
    """Test response caching."""
    ds = DocSense()

    # Configure mock responses
    mock_embedding = np.random.rand(1, 768)
    mock_models["embedding_model"].encode.return_value = mock_embedding

    mock_doc = Document("Test content", {"source": "test.txt"})
    mock_models["vector_store"].search.return_value = [(mock_doc, 0.9)]

    mock_llm_response = {"answer": "Test answer", "prompt": "Test prompt", "generation_config": {"test": "config"}}
    mock_models["llm_model"].generate.return_value = mock_llm_response

    # Expected response format
    expected_response = {
        "answer": "Test answer",
        "sources": [{"path": "test.txt", "type": "Unknown", "relevance_score": 0.9}],
        "metadata": {"prompt": "Test prompt", "generation_config": {"test": "config"}},
    }

    # First call
    with patch.object(ds, "_response_cache", {}) as mock_cache:
        response1 = ds.ask("Test question")

        # Store first response in cache
        cache_key = "test question"  # Normalized question
        mock_cache[cache_key] = response1

        # Second call with same question (should use cache)
        response2 = ds.ask("Test question")

        # Check that second call used cache
        assert response1 == expected_response
        assert response2 == expected_response
        assert mock_cache[cache_key] == expected_response
        assert len(mock_cache) == 1


def test_get_docsense():
    """Test get_docsense singleton function."""
    with patch("docsense.DocSense") as mock_cls:
        # Configure mock
        mock_instance = Mock()
        mock_cls.get_instance.return_value = mock_instance

        ds1 = get_docsense()
        ds2 = get_docsense()

        # Should return same instance
        assert ds1 == ds2
        # Should call get_instance twice but create only one instance
        assert mock_cls.get_instance.call_count == 2
        mock_cls.assert_not_called()  # Constructor should not be called directly


def test_create_index():
    """Test create_index convenience function."""
    with patch("docsense.DocSense") as mock_cls:
        mock_instance = Mock()
        mock_cls.get_instance.return_value = mock_instance

        create_index("test/path", model_name="test-model")

        # Should use get_instance
        mock_cls.get_instance.assert_called_once_with(model_name="test-model")
        # Should call index_documents
        mock_instance.index_documents.assert_called_once_with("test/path")


def test_ask_question():
    """Test ask_question convenience function."""
    with patch("docsense.DocSense") as mock_cls:
        mock_instance = Mock()
        mock_cls.get_instance.return_value = mock_instance

        ask_question("Test question", device="cpu")

        # Should use get_instance
        mock_cls.get_instance.assert_called_once_with(device="cpu")
        # Should call ask
        mock_instance.ask.assert_called_once_with("Test question")
