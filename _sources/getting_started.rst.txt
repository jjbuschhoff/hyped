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
    from hyped.data.pipe import DataPipe
    from hyped.data.processors.tokenizers.hf import (
        HuggingFaceTokenizer,
        HuggingFaceTokenizerConfig
    )

Next, load your dataset using the datasets library. In this example, we load the IMDb dataset:

.. code-block:: python

    ds = datasets.load_dataset("imdb")

Then, define your data pipeline using the `DataPipe` class from Hyped. Add data processors to the pipeline to specify the desired data transformations. For instance, the following code applies a HuggingFace tokenizer to tokenize the text feature of the dataset using the BERT tokenizer:

.. code-block:: python

    pipe = DataPipe([
        HuggingFaceTokenizer(
            HuggingFaceTokenizerConfig(
                tokenizer="bert-base-uncased",
                text="text"
            )
        )
    ])

Finally, apply the data pipeline to your dataset using the apply method:

.. code-block:: python

    ds = pipe.apply(ds)

Now, your dataset has been processed according to the defined pipeline, and you can proceed with further analysis or downstream tasks in your application.

For more examples and advanced usage scenarios, check out the `Hyped examples <https://github.com/open-hyped/examples>`_ repository.

Configuration
-------------

Hyped provides various configuration options that allow users to customize the behavior of the framework. Below are some of the key configuration options and how you can use them:

Processor Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Each data processor in Hyped can be configured with specific parameters to tailor its behavior. For example, when using the HuggingFaceTokenizer, you can specify the tokenizer model to use, the maximum sequence length, and other tokenizer-specific settings.

.. code-block:: python

    tokenizer_config = HuggingFaceTokenizerConfig(
        tokenizer="bert-base-uncased",
        max_length=128,
        padding=True,
        truncation=True
    )

Multiprocessing and Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyped supports data parallel multiprocessing to utilize multiple CPU cores for faster data processing. You can configure the number of processes to use and other multiprocessing options based on your system's specifications. Additionally, batch processing allows you to process data in batches, which can further improve performance and memory efficiency.

.. code-block:: python

    ds = pipe.apply(ds, num_proc=4, batch_size=32)

Data Streaming
~~~~~~~~~~~~~~

Hyped supports streaming data directly from and to disk, enabling efficient processing of large datasets that may not fit into memory. You can stream datasets using lazy processing, where examples are only processed when accessed.

.. code-block:: python

    from hyped.data.io.writers.json import JsonDatasetWriter

    # Load dataset with streaming enabled
    ds = datasets.load_dataset("imdb", split="train", streaming=True)

    # Apply data pipeline (lazy processing for streamed datasets)
    ds = pipe.apply(ds)

    # Write processed examples to disk using 4 worker processes
    JsonDatasetWriter("dump/", num_proc=4).consume(ds)

