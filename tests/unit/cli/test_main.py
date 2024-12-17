from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from docsense.cli.main import app, format_time

runner = CliRunner()


def test_format_time():
    """Test time formatting function"""
    # Less than a minute
    assert format_time(45.5) == "45.50 seconds"

    # More than a minute
    assert format_time(125.5) == "2 minutes 5.50 seconds"


@pytest.fixture
def mock_docsense():
    with (
        patch("docsense.cli.main.create_index") as mock_create,
        patch("docsense.cli.main.ask_question") as mock_ask,
        patch("docsense.cli.main.get_docsense") as mock_get,
    ):
        yield {"create_index": mock_create, "ask_question": mock_ask, "get_docsense": mock_get}


def test_index_command(mock_docsense, tmp_path):
    """Test index command with default parameters"""
    # Arrange
    doc_path = tmp_path / "docs"
    doc_path.mkdir()

    # Act
    result = runner.invoke(app, ["index", str(doc_path)])

    # Assert
    assert result.exit_code == 0
    assert "Creating document index..." in result.stdout
    assert "Using default index path:" in result.stdout
    mock_docsense["create_index"].assert_called_once()


def test_index_command_with_custom_path(mock_docsense, tmp_path):
    """Test index command with custom index path"""
    # Arrange
    doc_path = tmp_path / "docs"
    doc_path.mkdir()
    index_path = tmp_path / "custom_index"

    # Act
    result = runner.invoke(app, ["index", str(doc_path), "--index-path", str(index_path)])

    # Assert
    assert result.exit_code == 0
    assert "Index saved to:" in result.stdout
    mock_docsense["create_index"].assert_called_once_with(
        str(doc_path), model_name="Qwen/Qwen2-7B", device="cuda", index_path=str(index_path)
    )


def test_ask_command(mock_docsense, tmp_path):
    """Test ask command"""
    # Arrange
    mock_docsense["ask_question"].return_value = {
        "answer": "Test answer",
        "sources": [{"path": "test.txt", "type": "text", "relevance_score": 0.9}],
    }
    # Create a dummy index file
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)

    # Act
    result = runner.invoke(app, ["ask", "test question", "--index-path", str(index_path)])

    # Assert
    assert result.exit_code == 0
    mock_docsense["ask_question"].assert_called_once_with(
        "test question",
        model_name="Qwen/Qwen2-7B",
        device="cuda",
        index_path=str(index_path),
    )


def test_ask_command_no_index(mock_docsense, tmp_path):
    """Test ask command when no index exists"""
    # Arrange
    non_existent_path = tmp_path / "non_existent"

    # Act
    result = runner.invoke(app, ["ask", "test question", "--index-path", str(non_existent_path)])

    # Assert
    assert result.exit_code == 1
    assert "No index found" in result.stdout
    mock_docsense["ask_question"].assert_not_called()


@patch("builtins.input", side_effect=["test question", "exit"])
def test_daemon_command(mock_input, mock_docsense):
    """Test daemon command"""
    # Arrange
    mock_ds = Mock()
    mock_ds.ask.return_value = {
        "answer": "Test answer",
        "sources": [{"path": "test.txt", "type": "text", "relevance_score": 0.9}],
    }
    mock_docsense["get_docsense"].return_value = mock_ds

    # Act
    result = runner.invoke(app, ["daemon"])

    # Assert
    assert result.exit_code == 0
    assert "DocSense daemon is ready!" in result.stdout
    assert "Test answer" in result.stdout
    mock_ds.ask.assert_called_once_with("test question")
