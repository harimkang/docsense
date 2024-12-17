🚀 Installation Guide
======================

📋 Prerequisites
-----------------

Before installing DocSense, ensure you have the following:

- Python 3.10 or higher 🐍
- pip (Python package installer) 📦
- CUDA toolkit (optional, for GPU support) 🎮

💿 Installation Methods
-----------------------

🌐From PyPI
~~~~~~~~~~~~

.. code-block:: bash

    pip install docsense

💻 From Source
~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/yourusername/docsense.git
    cd docsense
    pip install -e .

🎯 GPU Support
---------------

To enable GPU acceleration:

1. Install CUDA toolkit (recommended version: 11.8) ⚡
2. Install PyTorch with CUDA support: 🔥

.. code-block:: bash

    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

🔍 Verifying Installation
--------------------------

.. code-block:: python

    import docsense
    print(docsense.__version__)

🚨 Common Issues
-----------------

🔍 CUDA Not Found
~~~~~~~~~~~~~~~~~~

If you encounter CUDA-related errors:

1. Verify CUDA installation
2. Check PyTorch CUDA compatibility
3. Set device to 'cpu' if GPU is not available

💾 Memory Issues
~~~~~~~~~~~~~~~~~

For large document collections:

1. Reduce batch size
2. Use CPU if GPU memory is insufficient
3. Consider splitting document collection