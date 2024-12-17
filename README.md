# DocSense 📚

[![PyPI version](https://badge.fury.io/py/docsense.svg)](https://badge.fury.io/py/docsense)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/docsense.svg)](https://pypi.org/project/docsense/)
[![Tests](https://github.com/harimkang/docsense/actions/workflows/test.yml/badge.svg)](https://github.com/harimkang/docsense/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/harimkang/docsense/branch/main/graph/badge.svg)](https://codecov.io/gh/harimkang/docsense)

An intelligent document assistant powered by Large Language Models 🤖

DocSense is a powerful tool that helps you interact with your documents using natural language. It currently uses the Qwen language model (with plans to support more models) to understand and answer questions about your documents with high accuracy and context awareness.

## Features ✨

- 🔍 Advanced semantic search using FAISS
- 💡 Intelligent question answering with LLMs (currently Qwen)
- 📝 Support for multiple document formats (txt, md, rst, etc.)
- ⚡ GPU acceleration for faster processing
- 🔄 Batch processing for memory efficiency
- 💾 Persistent vector storage

## Installation 🛠️

### CPU Version

```bash
    pip install docsense
```

### GPU Version (Recommended)

First, install PyTorch with CUDA support:

```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install FAISS with GPU support:

```bash
    conda install -c conda-forge faiss-gpu
```

Finally, install DocSense:

```bash
    pip install docsense
```

## Usage 🚀

### Creating Document Index

Index your documents directory:

```bash
    docsense index /path/to/your/documents
```

### Asking Questions

Ask a question to your documents:

```bash
    docsense ask "What is the meaning of life?"
```

### Interactive Mode

Start an interactive session for multiple questions:

```bash
    docsense daemon
```

### Command Line Options

All commands support the following options:

- `--model-name`: Specify the Qwen model to use (default: "Qwen/Qwen2-7B")
- `--device`: Choose computing device ("cuda" or "cpu", default: "cuda")
- `--index-path`: Set custom path for the vector index

Example with options:

```bash
    docsense index /path/to/your/documents --model-name "Qwen/Qwen2-7B" --device "cuda" --index-path /path/to/your/index
```

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Star History 🌟

[![Star History Chart](https://api.star-history.com/svg?repos=harimkang/docsense&type=Date)](https://star-history.com/#harimkang/docsense&Date)
