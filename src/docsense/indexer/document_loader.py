# Copyright (c) 2024 Harim Kang
# SPDX-License-Identifier: MIT

"""
Document loader implementation.

This module provides functionality to load documents from various file formats
and directories. It supports multiple text-based file formats and handles
different character encodings.
"""

from pathlib import Path

from .document import Document


class DocumentLoader:
    """Load documents from various sources."""

    def __init__(self):
        # Supported file extensions and their corresponding loader functions
        self.supported_extensions = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".rst": self._load_text,  # reStructuredText files
            ".rest": self._load_text,  # Alternative RST extension
            ".text": self._load_text,  # Alternative text extension
            ".conf": self._load_text,  # Configuration files
            ".py": self._load_text,  # Python source files
            ".ini": self._load_text,  # INI configuration files
            ".cfg": self._load_text,  # Config files
            ".json": self._load_text,  # JSON files
            ".yaml": self._load_text,  # YAML files
            ".yml": self._load_text,  # YAML files
        }

    def load_directory(self, path: str | Path) -> list[Document]:
        """Load all supported documents from a directory recursively.

        Args:
            path: Directory path to load documents from

        Returns:
            List of Document objects containing file contents and metadata

        Raises:
            FileNotFoundError: If directory does not exist
            NotADirectoryError: If path is not a directory
            ValueError: If no supported documents are found
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Find files for each supported extension
        all_files = []
        for ext in self.supported_extensions.keys():
            pattern = f"**/*{ext}"
            files = list(path.glob(pattern))
            all_files.extend(files)

        # Load files
        documents = []
        for file_path in all_files:
            try:
                content = self.supported_extensions[file_path.suffix.lower()](file_path)
                documents.append(
                    Document(
                        content=content,
                        metadata={
                            "source": str(file_path),
                            "type": file_path.suffix[1:],
                        },
                    )
                )
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")

        if not documents:
            raise ValueError(f"No documents found in {path}")

        return documents

    def _load_text(self, file_path: Path) -> str:
        """Load content from text-based files (txt, md, rst).

        Args:
            file_path: Path to the text file to load

        Returns:
            String containing the file contents

        Raises:
            UnicodeDecodeError: If file cannot be decoded with any supported encoding
            Exception: For other unexpected errors while reading the file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ["utf-8-sig", "cp1252", "latin1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                        return content
                except UnicodeDecodeError:
                    continue
            print(f"Failed to read {file_path} with all attempted encodings")
            raise
        except Exception as e:
            print(f"Unexpected error reading {file_path}: {str(e)}")
            raise
