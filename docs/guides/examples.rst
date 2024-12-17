💡 CLI Usage Examples
======================

📚 Basic Usage
---------------

Indexing Documents 📂
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Index a directory
    docsense index ./docs

    # Index with specific model
    docsense index ./docs --model-name "Qwen/Qwen2-7B"

    # Index using CPU
    docsense index ./docs --device cpu

Asking Questions 💭
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Simple query
    docsense ask "What is DocSense?"

    # Query with custom model
    docsense ask "How to use?" --model-name "custom-model"

    # Query from specific index
    docsense ask "Explain the API" --index-path ./my_index

🔄 Interactive Mode
-------------------

Daemon Mode Usage 🚀
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start daemon mode
    docsense daemon

    # With custom settings
    docsense daemon --device cpu --model-name "custom-model"

    # Interactive session example:
    > What is the main feature?
    > How do I configure it?
    > exit  # to quit

🎯 Common Scenarios
-------------------

Documentation Search 📖
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Index documentation
    docsense index ./api-docs

    # Query specific features
    docsense ask "How do I authenticate?"
    docsense ask "What are the API endpoints?"

Multiple Projects 📚
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Index different projects
    docsense index ./project1 --index-path ./index1
    docsense index ./project2 --index-path ./index2

    # Query specific project
    docsense ask "How to setup?" --index-path ./index1

💡 Tips and Tricks
------------------

Device Management 🖥️
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Check GPU availability
    docsense index ./docs  # Uses CUDA if available

    # Force CPU usage
    docsense index ./docs --device cpu
    docsense daemon --device cpu

Index Management 📁
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use custom index location
    export DOCSENSE_INDEX_PATH="./custom_index"
    docsense index ./docs

    # Query from custom index
    docsense ask "How to use?" --index-path ./custom_index

Batch Processing 📊
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Index multiple directories
    for dir in ./*/docs; do
        docsense index "$dir" --index-path "./indices/$(basename $(dirname $dir))"
    done