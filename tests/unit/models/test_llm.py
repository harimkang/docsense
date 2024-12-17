from unittest.mock import Mock, patch

import pytest
import torch

from docsense.models import LLMModel


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Mock model loading to prevent actual model downloads and initialization."""
    with (
        patch("docsense.models.llm.AutoTokenizer") as mock_tokenizer,
        patch("docsense.models.llm.AutoModelForCausalLM") as mock_model,
    ):
        # Configure mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        mock_tokenizer.from_pretrained.return_value = tokenizer

        # Configure mock model
        model = Mock()
        model.config.hidden_size = 768
        model.eval = Mock(return_value=None)
        model.device = "cpu"
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.from_pretrained.return_value = model

        yield {"tokenizer": tokenizer, "model": model, "mock_tokenizer": mock_tokenizer, "mock_model": mock_model}


def test_create_prompt(mock_model_loading):
    """Test prompt creation with different inputs."""
    model = LLMModel(device="cpu")

    # Test with context
    context = ["Context 1", "Context 2"]
    question = "Test question?"
    prompt = model._create_prompt(question, context)

    assert "Context:" in prompt
    assert "Context 1" in prompt
    assert "Context 2" in prompt
    assert "Question: Test question?" in prompt

    # Test without context
    prompt = model._create_prompt(question)
    assert "Question: Test question?" in prompt
    assert "Context:" not in prompt


def test_post_process_response(mock_model_loading):
    """Test response post-processing."""
    model = LLMModel(device="cpu")

    # Test truncation
    long_response = "a" * 2000
    processed = model._post_process_response(long_response)
    assert len(processed) <= 1100
    assert "..." in processed
    assert processed.split("\n\n")[0].endswith("...")  # Check main content truncation

    # Test multiple warnings
    response = "I think this might be the answer"  # Contains speculative language and no citations
    processed = model._post_process_response(response)
    warning_section = processed.split("\n\n", 1)[1]  # Get warning section
    assert "(Please only state facts directly from the documents.)" in warning_section

    # Test no answer found
    response = "I cannot find the answer in the documents"
    processed = model._post_process_response(response)
    assert processed == response  # Should return as-is

    # Test response with citations
    response = "According to [Document 1], the answer is correct."
    processed = model._post_process_response(response)
    assert "(Please include document citations" not in processed


def test_prepare_context(mock_model_loading):
    """Test context preparation."""
    model = LLMModel(device="cpu")

    contexts = ["First document content.", "Second document content.", "Third document content."]

    formatted = model._prepare_context(contexts)

    # Check document references
    assert "[Document 1]" in formatted
    assert "[Document 2]" in formatted
    assert "[Document 3]" in formatted

    # Check content
    assert "First document content" in formatted

    # Test max length
    long_contexts = ["Long content " * 100] * 5
    formatted = model._prepare_context(long_contexts, max_length=100)
    assert len(formatted) <= 100


@pytest.mark.parametrize(
    "device,expected_device",
    [
        ("cuda", "cuda" if torch.cuda.is_available() else "cpu"),
        ("cpu", "cpu"),
    ],
)
def test_device_fallback(device, expected_device, mock_model_loading):
    """Test device fallback when CUDA is not available."""
    model = LLMModel(device=device)
    assert model.device == expected_device
