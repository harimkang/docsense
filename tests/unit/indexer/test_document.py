import pytest

from docsense.indexer import Document
from docsense.indexer.document import DocumentChunker


def test_document_creation():
    """Test basic document creation."""
    # Test with content and metadata
    content = "Test content"
    metadata = {"source": "test.txt", "type": "text"}
    doc = Document(content=content, metadata=metadata)

    assert doc.content == content
    assert doc.metadata == metadata

    # Test with content only (metadata should default to empty dict)
    doc = Document(content=content)
    assert doc.content == content
    assert doc.metadata == {}


def test_document_metadata_immutability():
    """Test that default metadata doesn't share state between instances."""
    doc1 = Document(content="Test 1")
    doc2 = Document(content="Test 2")

    doc1.metadata["key"] = "value"
    assert "key" not in doc2.metadata


@pytest.fixture
def chunker():
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


def test_chunker_initialization():
    """Test chunker initialization with different parameters."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 50


def test_chunker_single_chunk(chunker):
    """Test chunking when text is smaller than chunk size."""
    text = "Short text that fits in one chunk."
    metadata = {"source": "test.txt"}

    chunks = chunker.split(text, metadata)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].metadata["source"] == metadata["source"]
    assert "chunk_start" in chunks[0].metadata
    assert "chunk_end" in chunks[0].metadata


def test_chunker_multiple_chunks(chunker):
    """Test chunking of longer text into multiple chunks."""
    # Create text that's longer than chunk_size
    text = " ".join(["chunk"] * 30)  # Will be longer than 100 chars
    metadata = {"source": "test.txt"}

    chunks = chunker.split(text, metadata)

    assert len(chunks) > 1

    # Check overlap between chunks
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i].metadata["chunk_end"]
        chunk2_start = chunks[i + 1].metadata["chunk_start"]
        assert chunk2_start < chunk1_end  # Should overlap

        # Verify metadata inheritance
        assert chunks[i].metadata["source"] == metadata["source"]


def test_chunker_sentence_boundary(chunker):
    """Test that chunker attempts to split at sentence boundaries."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    metadata = {"source": "test.txt"}

    chunks = chunker.split(text, metadata)

    # Check that chunks end with sentence boundaries where possible
    for chunk in chunks:
        if chunk != chunks[-1]:  # All except last chunk
            chunk_text = chunk.content
            assert chunk_text.endswith(("."))


def test_chunker_metadata_preservation():
    """Test that original metadata is preserved in chunks."""
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    text = "Short text for testing metadata preservation."
    metadata = {"source": "test.txt", "type": "text", "author": "Test Author"}

    chunks = chunker.split(text, metadata)

    for chunk in chunks:
        # Original metadata should be preserved
        assert chunk.metadata["source"] == metadata["source"]
        assert chunk.metadata["type"] == metadata["type"]
        assert chunk.metadata["author"] == metadata["author"]
        # Chunk metadata should be added
        assert "chunk_start" in chunk.metadata
        assert "chunk_end" in chunk.metadata


def test_chunker_empty_text():
    """Test chunking of empty text."""
    chunker = DocumentChunker()
    chunks = chunker.split("", {"source": "empty.txt"})

    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].metadata["source"] == "empty.txt"
