"""A comprehensive framework for constructing, managing, and executing complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows, processors,
and augmenters to handle a wide variety of data processing tasks.

Sub-modules:
    - :class:`flow`: Defines the core classes and functions for creating and managing
      data flows as directed acyclic graphs (DAGs). 

    - :class:`processors`: Provides a comprehensive collection of data processors designed
      to handle different data modalities and perform a wide range of data
      transformations. Data processors are the fundamental building blocks in a
      data flow graph, acting as nodes that implement specific, modular data
      transformations.

    - :class:`augmentors`: coming soon

    - :class:`ops`: Provides high-level feature operators for data processors.
      The operator module defines high-level functions for performing common
      operations. These functions delegate the actual processing to specific
      processors. Feature operators are designed to simplify the process of
      adding processors to a data flow by providing high-level functions

Example:
    Define a data flow for processing text data:

    .. code-block:: python

        import datasets
        from hyped.data.flow import DataFlow
        from hyped.data.flow.processors.tokenizers import TransformersTokenizer

        # create the data flow instance
        features = datasets.Features({"text": datasets.Value("string")})
        flow = DataFlow(features=features)

        # Define a processing step to tokenize text
        tokenizer = TokenizerProcessor(model_name="bert-base-uncased")
        tokenized_features = tokenizer.call(text=data_flow.src_features.text)

        # Apply the data flow to a dataset
        dataset = datasets.load(...)
        processed_dataset, _ = flow.apply(dataset)
"""

from . import ops
from .core.flow import DataFlow
