Getting Started
===============

Get up and running with Hyped in no time! Follow these simple steps to install the framework and start defining and executing your data pipelines effortlessly.

Installation
------------

Hyped is available on PyPI and can be installed using pip:

.. code-block:: bash

    pip install hyped

Alternatively, you can install Hyped directly from the source code repository:

.. code-block:: bash

    # Clone the Hyped repository from GitHub
    git clone https://github.com/open-hyped/hyped.git

    # Navigate to the cloned repository
    cd hyped

    # Install the package including optional developer dependencies
    pip install -e .[linting, tests]

Now you're ready to start using Hyped for managing and executing your data pipelines!

Usage
-----

Start by importing the necessary modules and classes:

.. code-block:: python

    import datasets
    from hyped.data.flow import DataFlow
    from hyped.data.flow.processors.tokenizers.transformers import (
        TransformersTokenizer,
        TransformersTokenizerConfig
    )

Next, load your dataset using the datasets library. In this example, we load the IMDb dataset:

.. code-block:: python

    ds = datasets.load_dataset("imdb", split="test")

With the dataset features available we can create a data flow instance:

.. code-block:: python

    flow = DataFlow(features=ds.features)

Now we can add processing steps by calling data processors on the features. In this example we add a tokenizer processor to tokenize the text input feature using a BERT tokenizer:

.. code-block:: python

    tokenizer = TransformersTokenizer(tokenizer="bert-base-uncased")
    tokenized_features = tokenizer.call(text=flow.src_features.text)

Finally, we can apply the data pipeline to your dataset using the `apply` method. Here we also need to specify the which features are to be collected into the output dataset:

.. code-block:: python

    ds, _ = flow.apply(ds, collect=tokenized_features)

For more examples and advanced usage scenarios, check out the `Hyped examples <https://github.com/open-hyped/examples>`_ repository.

Configuration
-------------

Hyped provides various configuration options that allow users to customize the behavior of the framework. Below are some of the key configuration options and how you can use them:

Processor Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Each data processor in Hyped can be configured with specific parameters to tailor its behavior. For example, when using the :class:`TransformersTokenizer`, you can specify the tokenizer model to use, the maximum sequence length, and other tokenizer-specific settings.

.. code-block:: python

    config = TransformersTokenizerConfig(
        tokenizer="bert-base-uncased",
        max_length=128,
        padding=True,
        truncation=True
    )

Multiprocessing and Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyped supports data parallel multiprocessing to utilize multiple CPU cores for faster data processing. You can configure the number of processes to use and other multiprocessing options based on your system's specifications. Additionally, batch processing allows you to process data in batches, which can further improve performance and memory efficiency.

.. code-block:: python

    ds, _ = flow.apply(ds, num_proc=4, batch_size=32)

Data Streaming
~~~~~~~~~~~~~~

Hyped supports streaming data directly from and to disk, enabling efficient processing of large datasets that may not fit into memory. You can stream datasets using lazy processing, where examples are only processed when accessed.

.. code-block:: python

    from hyped.data.io.writers.json import JsonDatasetWriter

    # Load dataset with streaming enabled
    ds = datasets.load_dataset("imdb", split="train", streaming=True)

    # Apply data pipeline (lazy processing for streamed datasets)
    ds, _ = flow.apply(ds)

    # Write processed examples to disk using 4 worker processes
    JsonDatasetWriter("dump/", num_proc=4).consume(ds)

