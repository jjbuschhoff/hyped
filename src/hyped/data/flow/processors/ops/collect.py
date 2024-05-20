"""Module for collecting features from nested structures in data processing.

This module defines a data processor (`CollectFeatures`) that collects features
from a nested structure defined by a `FeatureCollection` object. The nested
structure can be arbitrarily deep, consisting of dictionaries and lists, with
the leaves of the structure being `FeatureRef` objects.

This is particularly useful for defining the output of a data flow.
The desired output features, potentially from different nodes, can
be referenced in a `FeatureCollection`, which the processor can then
collect. This simplifies the management and retrieval of the specified
output features.
"""
from __future__ import annotations

from typing import Any

from datasets.features.features import Features, FeatureType, Sequence
from pydantic import BaseModel, BeforeValidator, Field
from pydantic._internal._model_construction import ModelMetaclass
from typing_extensions import Annotated

from hyped.common.feature_checks import check_feature_equals
from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.refs.inputs import InputRefs
from hyped.data.flow.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


class FeatureCollection(BaseModel):
    """Represents a collection of features.

    This class provides a structure for a collection of features.
    It can be infinitely nested dictionaries or lists. The leaves
    of this nested structure must be FeatureRef instances.
    """

    collection: Annotated[
        (
            dict[str, FeatureCollection | FeatureRef]
            | Annotated[
                list[FeatureCollection | FeatureRef], Field(min_length=1)
            ]
        ),
        BeforeValidator(
            lambda x: (
                {
                    k: (
                        v
                        if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for k, v in x.items()
                }
                if isinstance(x, dict)
                else [
                    (
                        v
                        if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for v in x
                ]
                if isinstance(x, list)
                else x
            )
        ),
    ]
    """
    The nested collection of features
    """

    @property
    def feature(self) -> FeatureType:
        """The feature type of the collection.

        Returns:
            FeatureType: The feature type.
        """
        if isinstance(self.collection, dict):
            return Features(
                {k: v.feature_ for k, v in self.collection.items()}
            )

        if isinstance(self.collection, list):
            # collect all features in specified in the list
            collected_features = (item.feature_ for item in self.collection)
            f = next(collected_features)
            # make sure the feature types match
            for ff in collected_features:
                if not check_feature_equals(f, ff):
                    raise TypeError(
                        "Expected all items of a sequence to be of the "
                        "same feature type, got %s != %s" % (str(f), str(ff))
                    )
            return Sequence(f, length=len(self.collection))

    @property
    def refs(self) -> set[FeatureRef]:
        """Get the feature references within the collection.

        Returns:
            set[FeatureRef]: A set of feature references.
        """
        feature_refs = set()

        for v in (
            self.collection.values()
            if isinstance(self.collection, dict)
            else self.collection
        ):
            if isinstance(v, FeatureCollection):
                feature_refs.update(v.refs)
            elif isinstance(v, FeatureRef):
                feature_refs.add(v)

        return feature_refs


class CollectFeaturesInputRefs(InputRefs):
    """Input references for the CollectFeatures data processor.

    This class represents the input references used by the CollectFeatures
    data processor. It contains a single attribute, 'collection', which is
    an instance of FeatureCollection containing the features to be collected.
    """

    collection: FeatureCollection
    """
    Feature collection instance to be collected and packed into a new feature
    """

    @classmethod
    def type_validator(cls) -> None:
        """Validate the type of input references."""
        pass

    @classmethod
    @property
    def keys(cls) -> set[str]:
        """Get the keys corresponding to the input reference fields.

        Since the input references are dynamic and based on the collection,
        this method returns an empty set.

        Returns:
            set[str]: An empty set.
        """
        return set()

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        """Get the named input references.

        This property returns a dictionary mapping input reference field names
        to their corresponding instances. The field names are the hash values
        of the FeatureRef instances in the collection.

        Returns:
            dict[str, FeatureRef]: A dictionary of named input references.
        """
        return {str(hash(ref)): ref for ref in self.collection.refs}

    @property
    def flow(self) -> object:
        """Get the associated data flow graph.

        Returns:
            object: The associated data flow graph.
        """
        return next(iter(self.collection.refs)).flow_


class CollectFeaturesOutputRefs(OutputRefs):
    """Output feature references for the CollectFeatures data processor.

    This class represents the output feature references produced by the
    CollectFeatures data processor. It contains a single attribute, 'collected',
    which is an instance of FeatureRef representing the collected features as
    defined by the FeatureCollection in the input references.
    """

    collected: Annotated[
        FeatureRef, LambdaOutputFeature(lambda _, i: i.collection.feature)
    ]
    """
    The output reference representing the collected features.
    
    The feature type is inferred from the feature collection defined
    in by the input of the CollectFeatures processor.
    """


class CollectFeaturesConfig(BaseDataProcessorConfig):
    """Empty Configuration class for the CollectFeatures data processor."""


class CollectFeatures(
    BaseDataProcessor[
        CollectFeaturesConfig,
        CollectFeaturesInputRefs,
        CollectFeaturesOutputRefs,
    ]
):
    """Data processor for collecting features into a new (nested) feature.

    This processor collects features from a nested structure defined by a
    FeatureCollection object. It traverses the nested structure and gathers
    the features, maintaining the structure defined by the FeatureCollection.
    """

    def __init__(self) -> None:
        """Initializes the CollectFeatures processor."""
        self.collection: FeatureCollection | None = None
        super(CollectFeatures, self).__init__(config=CollectFeaturesConfig())

    @classmethod
    def from_config(cls, config: CollectFeaturesConfig) -> None:
        """Creates a CollectFeatures processor from a configuration.

        This method creates a new instance of the CollectFeatures processor
        from the provided configuration object.

        Args:
            config (CollectFeaturesConfig): The configuration object for
                the CollectFeatures processor.

        Returns:
            CollectFeatures: An instance of the CollectFeatures processor.
        """
        return cls()

    def call(
        self,
        inputs: None | CollectFeaturesInputRefs = None,
        collection: None | FeatureCollection = None,
        **kwargs,
    ) -> int:
        """Calls the CollectFeatures processor with specified inputs.

        This method calls the CollectFeatures processor with either the
        specified input references or a FeatureCollection object. It checks
        if the processor is already in use, and if so, creates a new instance
        for this call. The processor is then added to the data flow graph,
        and an output reference is returned.

        Args:
            inputs (None | CollectFeaturesInputRefs, optional): The input
                references for the processor. Defaults to None.
            collection (None | FeatureCollection, optional): The feature
                collection defining the structure of the features to be
                collected. Defaults to None.
            **kwargs: Additional keyword arguments to be passed as inputs.

        Returns:
            int: The output of the CollectFeatures processor.

        Raises:
            ValueError: If multiple input options are specified.
        """
        if self.collection is not None:
            # feature collector is already in use
            # create a new one for this call
            return CollectFeatures().call(
                inputs=inputs, collection=collection, **kwargs
            )

        # check inputs
        if (
            sum([inputs is not None, collection is not None, len(kwargs) > 0])
            > 1
        ):
            raise ValueError(
                "Please specify either 'inputs', 'collection' or keyword "
                "arguments, but not multiple."
            )

        if collection is not None:
            # build collection from collection argument
            collection = (
                collection
                if isinstance(collection, FeatureCollection)
                else FeatureCollection(collection=collection)
            )
        elif len(kwargs) > 0:
            # build collection from keyword arguments
            collection = FeatureCollection(collection=kwargs)

        if inputs is None:
            # build input refs from collection
            inputs = CollectFeaturesInputRefs(collection=collection)

        # save collection
        self.collection = inputs.collection
        # call the processor
        return super(CollectFeatures, self).call(inputs=inputs)

    def collect_values(
        self, inputs: Batch, col: FeatureCollection | FeatureRef
    ) -> list[Any]:
        """Recursively collects values from the input batch.

        This method recursively collects values from the input batch according
        to the structure defined by the feature collection. It traverses the
        nested structure and gathers the values, maintaining the structure
        defined by the FeatureCollection.

        Args:
            inputs (Batch): The batch of input samples.
            col (FeatureCollection | FeatureRef): The current feature collection
                or feature reference to collect values from.

        Returns:
            list[Any]: The collected values.
        """
        if isinstance(col, FeatureRef):
            return inputs[str(hash(col))]

        if isinstance(col.collection, dict):
            # collect values from sub-collection and
            # convert from dict of lists to list of dicts
            data = {
                k: self.collect_values(inputs, v)
                for k, v in col.collection.items()
            }
            return [
                dict(zip(data.keys(), values))
                for values in zip(*data.values())
            ]

        if isinstance(col.collection, list):
            # collect values from sub-collection and transpose
            data = (self.collect_values(inputs, v) for v in col.collection)
            return [list(row) for row in zip(*data)]

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> tuple[Batch, list[int]]:
        """Processes a batch of inputs and collects the features.

        This method processes a batch of input samples and collects the
        features according to the structure defined by the feature collection.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            tuple[Batch, list[int]]: The batch of output samples and the
                corresponding indices.
        """
        # collect values
        out = {
            "collected": self.collect_values(
                inputs=inputs, col=self.collection
            )
        }
        # return collected values and index
        return out, index
