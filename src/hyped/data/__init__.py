"""Data Module.

Usage Example:
    Define and execute a data processing workflow:

    .. code-block: python

        # Import the necessary classes from the module
        from hyped.data.flow import DataFlow, FeatureRef

        # Define the features of the source node
        src_features = datasets.Features({"text": datasets.Value("string")})

        # Initialize a DataFlow instance with the source features
        flow = DataFlow(features=src_features)

        # Define and add processing steps to the data flow
        # For example, tokenize text using a bert tokenizer
        tokenizer = TransformersTokenizer(tokenizer="bert-base-uncased")
        tokenized_features = tokenizer(text=src_features.text)

        # Execute the data flow on a dataset
        processed_dataset = data_flow.apply(dataset)
"""
from hyped.__version__ import __version__, __version_tuple__

from . import ops
from .flow.flow import DataFlow
