import numpy as np
import pytest

from docsense.indexer import Document, VectorStore


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document("First document content", {"source": "doc1.txt"}),
        Document("Second document content", {"source": "doc2.txt"}),
        Document("Third document content", {"source": "doc3.txt"}),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # Create 3 embeddings of dimension 4
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def vector_store(tmp_path):
    """Create a vector store instance for testing."""
    return VectorStore(dimension=4, index_path=str(tmp_path / "index"), use_gpu=False)


def test_vector_store_initialization(tmp_path):
    """Test vector store initialization."""
    store = VectorStore(dimension=4, index_path=str(tmp_path / "index"), use_gpu=False)
    assert store.dimension == 4
    assert store.index_path == tmp_path / "index"
    assert not store.use_gpu


def test_add_documents(vector_store, sample_documents, sample_embeddings):
    """Test adding documents to the store."""
    vector_store.add_documents(sample_documents, sample_embeddings)
    assert len(vector_store.documents) == 3

    # Test document storage
    assert vector_store.documents[0].content == "First document content"
    assert vector_store.documents[0].metadata["source"] == "doc1.txt"


def test_add_documents_dimension_mismatch(vector_store, sample_documents):
    """Test adding documents with wrong embedding dimensions."""
    wrong_embeddings = np.random.rand(3, 5)  # Wrong dimension (5 instead of 4)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        vector_store.add_documents(sample_documents, wrong_embeddings)


def test_add_documents_count_mismatch(vector_store, sample_documents, sample_embeddings):
    """Test adding documents with mismatched counts."""
    wrong_embeddings = np.random.rand(2, 4)  # Only 2 embeddings for 3 documents

    with pytest.raises(ValueError, match="Number of documents must match"):
        vector_store.add_documents(sample_documents, wrong_embeddings)


def test_search(vector_store, sample_documents, sample_embeddings):
    """Test searching for similar documents."""
    # Add documents
    vector_store.add_documents(sample_documents, sample_embeddings)

    # Search with a query that should match the first document
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = vector_store.search(query)

    assert len(results) > 0
    assert results[0][0].content == "First document content"
    assert isinstance(results[0][1], float)  # Check that distance is returned


def test_search_empty_store(vector_store):
    """Test searching in empty store."""
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = vector_store.search(query)
    assert len(results) == 0


def test_save_and_load(vector_store, sample_documents, sample_embeddings, tmp_path):
    """Test saving and loading the index."""
    # Add documents and save
    vector_store.add_documents(sample_documents, sample_embeddings)
    vector_store.save()

    # Create new store and load
    new_store = VectorStore(dimension=4, index_path=str(tmp_path / "index"), use_gpu=False)
    new_store.load()

    # Verify documents were loaded
    assert len(new_store.documents) == 3
    assert new_store.documents[0].content == "First document content"

    # Test search functionality after loading
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = new_store.search(query)
    assert len(results) > 0


def test_save_no_path(vector_store):
    """Test saving without index path."""
    vector_store.index_path = None
    with pytest.raises(ValueError, match="No index path specified"):
        vector_store.save()


def test_load_no_path(vector_store):
    """Test loading without index path."""
    vector_store.index_path = None
    with pytest.raises(ValueError, match="No index path specified"):
        vector_store.load()


def test_load_missing_files(vector_store):
    """Test loading when files don't exist."""
    with pytest.raises(FileNotFoundError):
        vector_store.load()


def test_clear(vector_store, sample_documents, sample_embeddings):
    """Test clearing the store."""
    # Add documents
    vector_store.add_documents(sample_documents, sample_embeddings)
    assert len(vector_store.documents) > 0

    # Clear store
    vector_store.clear()
    assert len(vector_store.documents) == 0

    # Test search after clearing
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = vector_store.search(query)
    assert len(results) == 0
