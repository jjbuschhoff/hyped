"""Provides high-level feature operators for data processors.

The operator module defines high-level functions for performing common operations.
These functions delegate the actual processing to specific processor classes and
return references to the resulting features represented by `FeatureRef` instances.

Feature operators are designed to simplify the process of adding processors to a data
flow by providing high-level functions for common feature operations. Each operator
encapsulates the logic for performing specific tasks, such as collecting features from
a collection (e.g., dictionary or list). These functions leverage underlying processor
classes, such as `CollectFeatures`, to execute the desired operations.

Functions:
    - `collect`: Collect features from a given collection.

Usage Example:
    Collect features from a dictionary using the `collect` operator:

    .. code-block: python

        # Import the collect operator from the module
        from hyped.data.processors.operator import collect

        # Define the features of the source node
        src_features = datasets.Features({"text": datasets.Value("string")})

        # Initialize a DataFlow instance with the source features
        flow = DataFlow(features=src_features)
        
        # Collect features from the dictionary using the collect operator
        collected_features = collect(
            collection={"out": flow.src_features.text}
        )
"""

from hyped.data.processors.ops.collect import CollectFeatures

from .ref import FeatureRef


def collect(collection: None | dict | list = None, **kwargs) -> FeatureRef:
    """Collect features from a given collection.

    This function provides a high-level operator for collecting features from a given collection,
    such as a dictionary or a list. It delegates the collection process to the `CollectFeatures`
    class and returns a FeatureRef instance representing the collected features.

    Args:
        collection (None | dict | list, optional): The collection from which to collect features.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the collection process.

    Returns:
        FeatureRef: A FeatureRef instance representing the collected features.
    """
    return CollectFeatures().call(collection=collection, **kwargs)
