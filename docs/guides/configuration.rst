⚙️ Configuration Guide
======================

🛠️ Command-Line Options
------------------------

Global Options
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Default
   * - ``--model-name``
     - Name of the Qwen model
     - ``"Qwen/Qwen2-7B"``
   * - ``--device``
     - Computing device (cuda/cpu)
     - ``"cuda"``
   * - ``--index-path``
     - Custom index location
     - ``~/.docsense/index``

💾 Index Command
~~~~~~~~~~~~~~~~

.. code-block:: bash

    docsense index <directory>

    Options:
      --model-name TEXT    Name of the Qwen model [default: Qwen/Qwen2-7B]
      --device TEXT       Computing device (cuda/cpu) [default: cuda]
      --index-path PATH   Custom index location
      --help             Show this message and exit

🔍 Ask Command
~~~~~~~~~~~~~~

.. code-block:: bash

    docsense ask <question>

    Options:
      --model-name TEXT    Name of the Qwen model [default: Qwen/Qwen2-7B]
      --device TEXT       Computing device (cuda/cpu) [default: cuda]
      --index-path PATH   Custom index location
      --help             Show this message and exit

🔄 Daemon Command
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docsense daemon

    Options:
      --model-name TEXT    Name of the Qwen model [default: Qwen/Qwen2-7B]
      --device TEXT       Computing device (cuda/cpu) [default: cuda]
      --index-path PATH   Custom index location
      --help             Show this message and exit

🎛️ Advanced Usage
------------------

Environment Variables 🌍
~~~~~~~~~~~~~~~~~~~~~~~~~

You can set default options using environment variables:

.. code-block:: bash

    # Set default model
    export DOCSENSE_MODEL="custom-model"

    # Set default device
    export DOCSENSE_DEVICE="cpu"

    # Set default index location
    export DOCSENSE_INDEX_PATH="./my_index"

Custom Index Location 📁
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use custom index location
    docsense index ./docs --index-path ./project_index

    # Query from custom index
    docsense ask "How to use?" --index-path ./project_index

🚀 Performance Tips
---------------------

Device Selection 💻
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use CPU for indexing
    docsense index ./docs --device cpu

    # Use GPU for faster processing
    docsense index ./docs --device cuda

Daemon Mode for Multiple Queries ⚡
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start daemon mode for faster consecutive queries
    docsense daemon --device cuda

    # Interactive queries
    > What is DocSense?
    > How do I use it?
    > exit  # to quit

Memory Management 🎯
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Use CPU when GPU memory is limited
    docsense index ./large-docs --device cpu

    # Run daemon mode with CPU
    docsense daemon --device cpu 