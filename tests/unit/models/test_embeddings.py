from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from docsense.models import EmbeddingModel


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Mock model loading to prevent actual model downloads and initialization."""
    with (
        patch("docsense.models.embeddings.AutoModel") as mock_model_cls,
        patch("docsense.models.embeddings.AutoTokenizer") as mock_tokenizer_cls,
    ):
        # Configure mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.model_max_length = 512

        def create_mock_inputs(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            inputs = Mock()
            inputs.input_ids = torch.ones((batch_size, 10))
            inputs.attention_mask = torch.ones((batch_size, 10))
            inputs.to = Mock(return_value=inputs)

            # Make inputs behave like a dict
            inputs.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
            inputs.keys = lambda: ["input_ids", "attention_mask"]
            inputs.items = lambda: [("input_ids", inputs.input_ids), ("attention_mask", inputs.attention_mask)]
            inputs.__getitem__ = lambda self, key: getattr(self, key)
            return inputs

        tokenizer.side_effect = create_mock_inputs
        mock_tokenizer_cls.from_pretrained.return_value = tokenizer

        # Configure mock model
        model = Mock()
        model.config.hidden_size = 768
        model.to = Mock(return_value=model)

        def create_model_output(**kwargs):
            # Get batch size from input tensors
            batch_size = kwargs["input_ids"].shape[0]
            # Create embeddings for testing (mean will be taken later)
            embeddings = torch.randn(batch_size, 10, 768)

            model_output = Mock()
            model_output.last_hidden_state = embeddings
            return model_output

        model.side_effect = create_model_output
        mock_model_cls.from_pretrained.return_value = model

        yield {
            "tokenizer": tokenizer,
            "model": model,
            "mock_tokenizer": mock_tokenizer_cls,
            "mock_model": mock_model_cls,
        }


def test_embedding_initialization(mock_model_loading):
    """Test embedding model initialization."""
    model = EmbeddingModel(device="cpu")

    assert model.device == "cpu"
    assert model.max_length == 512
    assert model.normalize_embeddings is True

    # Check model loading parameters
    mock_model_loading["mock_model"].from_pretrained.assert_called_once_with(
        "Qwen/Qwen2-7B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )


def test_encode_single(mock_model_loading):
    """Test encoding a single text."""
    model = EmbeddingModel(device="cpu")

    text = "Test document content"
    embeddings = model.encode(text)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32


def test_encode_batch(mock_model_loading):
    """Test encoding a batch of texts."""
    model = EmbeddingModel(device="cpu")

    texts = ["First document", "Second document", "Third document"]
    embeddings = model.encode(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32


def test_encode_with_normalization(mock_model_loading):
    """Test encoding with normalization."""
    model = EmbeddingModel(device="cpu", normalize_embeddings=True)

    text = "Test document"
    embeddings = model.encode(text)

    # Check if embeddings are normalized (L2 norm ≈ 1)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones_like(norms))


def test_encode_without_normalization(mock_model_loading):
    """Test encoding without normalization."""
    model = EmbeddingModel(device="cpu", normalize_embeddings=False)

    text = "Test document"
    embeddings = model.encode(text)

    # Check if embeddings are not normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert not np.allclose(norms, np.ones_like(norms))


def test_get_embedding_dim(mock_model_loading):
    """Test getting embedding dimension."""
    model = EmbeddingModel(device="cpu")
    assert model.get_embedding_dim() == 768


@pytest.mark.parametrize(
    "device,expected_device",
    [
        ("cuda", "cuda" if torch.cuda.is_available() else "cpu"),
        ("cpu", "cpu"),
    ],
)
def test_device_fallback(device, expected_device, mock_model_loading):
    """Test device fallback when CUDA is not available."""
    model = EmbeddingModel(device=device)
    assert model.device == expected_device


def test_long_text_truncation(mock_model_loading):
    """Test handling of long texts."""
    model = EmbeddingModel(device="cpu", max_length=10)

    # Create a very long text
    long_text = "word " * 100
    embeddings = model.encode(long_text)

    # Verify embeddings shape and type
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32

    # Verify tokenizer was called with truncation
    mock_model_loading["tokenizer"].assert_called_with(
        [long_text], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
