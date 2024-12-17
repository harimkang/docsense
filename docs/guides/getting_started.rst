🚀 Getting Started
================

📋 Overview
----------

DocSense is a command-line tool that helps you create a searchable knowledge base from your documents and interact with them using natural language queries.

🎯 Quick Start
------------

1. **Install DocSense** 📥

.. code-block:: bash

    pip install docsense

2. **Index Your Documents** 📚

.. code-block:: bash

    # Index a directory of documents
    docsense index ./docs

    # Index with custom settings
    docsense index ./docs --model-name "Qwen/Qwen2-7B" --device cpu

3. **Ask Questions** 💬

.. code-block:: bash

    # Ask a simple question
    docsense ask "What are the key points in the documentation?"

    # Use different model or device
    docsense ask "Explain the architecture" --device cpu --model-name "custom-model"

    # Run in daemon mode for faster consecutive queries
    docsense daemon

🔧 Command Reference
-----------------

Index Command
~~~~~~~~~~~~

.. code-block:: bash

    # Basic usage
    docsense index <directory>

    # Options
      --model-name TEXT     Model to use for embeddings
      --device TEXT        Computing device (cuda/cpu)
      --index-path PATH    Custom index location
      --help              Show this message and exit

Ask Command
~~~~~~~~~~

.. code-block:: bash

    # Basic usage
    docsense ask <question>

    # Options
      --model-name TEXT    Model to use for answering
      --device TEXT       Computing device (cuda/cpu)
      --index-path PATH   Custom index location
      --help             Show this message and exit

Daemon Command
~~~~~~~~~~~~

.. code-block:: bash

    # Basic usage
    docsense daemon

    # Options
      --model-name TEXT    Model to use
      --device TEXT       Computing device (cuda/cpu)
      --index-path PATH   Custom index location
      --help             Show this message and exit

📝 Example Use Cases
-----------------

1. **Documentation Search** 📖

.. code-block:: bash

    # Index technical docs
    docsense index ./technical-docs

    # Search for specific information
    docsense ask "How do I configure the logging system?"

2. **Interactive Mode** 🔄

.. code-block:: bash

    # Start daemon mode for faster responses
    docsense daemon --device cpu

    # Then ask questions interactively
    > How do I get started?
    > What are the main features?
    > exit  # to quit

🔍 Next Steps
-----------

- Check out the :doc:`configuration` for advanced settings
- See :doc:`examples` for more use cases
- Read the :doc:`../api/docsense` for programmatic usage 