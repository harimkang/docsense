from pathlib import Path
from unittest.mock import patch

import pytest

from docsense.indexer import DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing."""
    # Create test directory structure
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create test files with different extensions
    files = {
        "test.txt": "This is a test document.",
        "test.md": "# Test Markdown\nThis is markdown content.",
        "test.rst": "Test RST\n=========\nThis is RST content.",
        "test.py": "def test():\n    print('This is Python code')",
        "test.json": '{"key": "This is JSON content"}',
        "test.yaml": "key: This is YAML content",
    }

    for filename, content in files.items():
        file_path = docs_dir / filename
        file_path.write_text(content)

    return docs_dir


def test_supported_extensions(loader):
    """Test that loader has expected supported extensions."""
    expected_extensions = {
        ".txt",
        ".md",
        ".rst",
        ".rest",
        ".text",
        ".conf",
        ".py",
        ".ini",
        ".cfg",
        ".json",
        ".yaml",
        ".yml",
    }
    assert set(loader.supported_extensions.keys()) == expected_extensions


def test_load_directory_success(loader, sample_files):
    """Test successful loading of documents from directory."""
    # Act
    documents = loader.load_directory(sample_files)

    # Assert
    assert len(documents) == 6  # Number of test files

    # Check document contents and metadata
    for doc in documents:
        assert doc.content  # Content should not be empty
        assert "source" in doc.metadata
        assert "type" in doc.metadata
        assert Path(doc.metadata["source"]).exists()


def test_load_directory_empty(loader, tmp_path):
    """Test loading from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No documents found"):
        loader.load_directory(empty_dir)


def test_load_directory_not_found(loader):
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
        loader.load_directory("/nonexistent/path")


def test_load_directory_not_directory(loader, tmp_path):
    """Test loading from a file path instead of directory."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    with pytest.raises(NotADirectoryError):
        loader.load_directory(file_path)


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "latin1"])
def test_load_text_different_encodings(loader, tmp_path, encoding):
    """Test loading text files with different encodings."""
    content = "This is a test with encoding"
    file_path = tmp_path / "test.txt"

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)

    loaded_content = loader._load_text(file_path)
    assert loaded_content == content


def test_load_text_utf8_with_bom(loader, tmp_path):
    """Test loading UTF-8 with BOM."""
    content = "This is a test with BOM"
    file_path = tmp_path / "test.txt"

    # Write UTF-8 with BOM
    with open(file_path, "wb") as f:
        f.write(b"\xef\xbb\xbf" + content.encode("utf-8"))

    loaded_content = loader._load_text(file_path)
    assert loaded_content == content


def test_load_text_unicode_error_fallback(loader):
    """Test fallback behavior when encountering encoding errors."""
    with patch("builtins.open") as mock_open:
        # First attempt (utf-8) raises UnicodeDecodeError
        mock_open.side_effect = [
            UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte"),
            UnicodeDecodeError("utf-8-sig", b"\xff\xfe", 0, 1, "invalid start byte"),
            UnicodeDecodeError("cp1252", b"\xff\xfe", 0, 1, "invalid start byte"),
            UnicodeDecodeError("latin1", b"\xff\xfe", 0, 1, "invalid start byte"),
        ]

        with pytest.raises(UnicodeDecodeError):
            loader._load_text(Path("test.txt"))


def test_load_directory_with_subdirectories(loader, sample_files):
    """Test loading documents from directory with subdirectories."""
    # Create subdirectory with additional files
    subdir = sample_files / "subdir"
    subdir.mkdir()
    (subdir / "subtest.txt").write_text("Subdirectory test content")

    # Act
    documents = loader.load_directory(sample_files)

    # Assert
    assert len(documents) == 7  # Original 6 files + 1 in subdirectory

    # Verify subdirectory file was loaded
    subdir_docs = [doc for doc in documents if "subdir" in doc.metadata["source"]]
    assert len(subdir_docs) == 1
    assert subdir_docs[0].content == "Subdirectory test content"
