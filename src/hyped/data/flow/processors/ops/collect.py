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

import typing

from datasets import Dataset
from datasets.features.features import Features, FeatureType, Sequence, Value
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

T = typing.TypeVar("T", str, int, float)


class Const(BaseModel):
    """Constant Value Wrapper.

    This wrapper is used to internally mark constant values in feature descriptions.
    It allows for the inclusion of constant values within feature collections,
    ensuring that these constants are properly handled and propagated within
    the data flow.
    """

    value: T
    """The value wrapped by the constant wrapper"""

    ftype: Value = None
    """HuggingFace Datasets Feature Type of the constant value."""

    def __init__(self, value: T, ftype: Value = None) -> None:
        """Initialize the Const object.

        Args:
            value (T): The constant value to be wrapped.
            ftype (Value, optional): The feature type of the constant value. If not provided,
                it will be inferred from the value.
        """
        # infer the feature type from the value
        if ftype is None:
            ftype = Dataset.from_dict({"feature": [value]}).features["feature"]
        # initialize model
        super(Const, self).__init__(value=value, ftype=ftype)

    def __str__(self) -> str:
        """String representation of the constant feature value."""
        return "Const(%s)" % repr(self.value)

    def __repr__(self) -> str:
        """String representation of the constant feature value."""
        return str(self)


class FeatureCollection(BaseModel):
    """Represents a collection of features.

    This class represents a collection of features that can be structured
    as nested dictionaries or lists. The leaves of this nested structure
    can be either :class:`FeatureRef` instances or primitive values.

    - :class:`FeatureRef` instances represent references to features within
      the data flow.
    - :class:`Const` instances represent constant feature values, which will
      be propagated to all collected samples
    - Primitive values are converted to :class:`Const` instances
      with inferred feature type.

    This flexibility allows for the inclusion of both dynamic feature references
    and static constant values within the same feature collection.
    """

    collection: Annotated[
        (
            dict[str, FeatureCollection | FeatureRef | Const]
            | Annotated[
                list[FeatureCollection | FeatureRef | Const],
                Field(min_length=1),
            ]
        ),
        BeforeValidator(
            lambda x: (
                {
                    k: (
                        v
                        if isinstance(
                            v, (FeatureRef, FeatureCollection, Const)
                        )
                        else FeatureCollection(collection=v)
                        if isinstance(v, (typing.Mapping, list, tuple))
                        else Const(value=v)
                    )
                    for k, v in x.items()
                }
                if isinstance(x, typing.Mapping)
                else [
                    (
                        v
                        if isinstance(
                            v, (FeatureRef, FeatureCollection, Const)
                        )
                        else FeatureCollection(collection=v)
                        if isinstance(v, (typing.Mapping, list, tuple))
                        else Const(value=v)
                    )
                    for v in x
                ]
                if isinstance(x, (list, tuple))
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
                {
                    k: (
                        v.ftype
                        if isinstance(v, Const)
                        else v.feature_
                        if isinstance(v, FeatureRef)
                        else v.feature
                        if isinstance(v, FeatureCollection)
                        else None  # unexpected type in collection
                    )
                    for k, v in self.collection.items()
                }
            )

        if isinstance(self.collection, list):
            # collect all features in specified in the list
            collected_features = (
                (
                    v.ftype
                    if isinstance(v, Const)
                    else v.feature_
                    if isinstance(v, FeatureRef)
                    else v.feature
                    if isinstance(v, FeatureCollection)
                    else None  # unexpected type in collection
                )
                for v in self.collection
            )
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
    def named_refs(self) -> dict[str, FeatureRef]:
        """Retrieve a dictionary of all feature references within the collection.

        This property method traverses the nested structure of the `FeatureCollection`
        and collects all `FeatureRef` instances. The keys in the returned dictionary
        represent the paths to the references in the nested structure, using dot notation
        for dictionaries and square bracket notation for lists.

        Returns:
            dict[str, FeatureRef]: A dictionary mapping the paths to their respective
            :class:`FeatureRef` instances.
        """
        named_items = (
            self.collection
            if isinstance(self.collection, dict)
            else {"[%i]" % i: v for i, v in enumerate(self.collection)}
        )

        named_refs = {}
        # collect all named references
        # names are the paths to the reference in the nested structure
        for k, v in named_items.items():
            if isinstance(v, FeatureRef):
                # collect feature refs
                named_refs[k] = v

            elif isinstance(v, FeatureCollection):
                # collect refs from sub-collection
                named_refs.update(
                    {
                        (
                            f"{k}.{kk}"
                            if isinstance(v.collection, dict)
                            else f"{k}{kk}"
                        ): ref
                        for kk, ref in v.named_refs.items()
                    }
                )

        return named_refs

    def _collect_values(
        self, inputs: Batch, batch_size: int
    ) -> list[typing.Any]:
        """Recursively collects values from an input batch.

        This method recursively collects values from the input batch according
        to the structure defined by the feature collection. It traverses the
        nested structure and gathers the values, maintaining the structure
        defined by the FeatureCollection.

        Args:
            inputs (Batch): The batch of input samples.
            batch_size (int): The number of samples in the batch.

        Returns:
            list[Any]: The collected values.
        """

        def collect(
            col: FeatureCollection, values: dict[FeatureRef, typing.Any]
        ) -> list[typing.Any]:
            if isinstance(col.collection, dict):
                # collect values from sub-collection and
                # convert from dict of lists to list of dicts
                data = {
                    k: (
                        [v.value] * batch_size
                        if isinstance(v, Const)
                        else values[v]
                        if isinstance(v, FeatureRef)
                        else collect(v, values)
                        if isinstance(v, FeatureCollection)
                        else None  # unexpected type in collection
                    )
                    for k, v in col.collection.items()
                }
                return [
                    dict(zip(data.keys(), values))
                    for values in zip(*data.values())
                ]

            if isinstance(col.collection, list):
                # collect values from sub-collection and transpose
                data = (
                    (
                        [v.value] * batch_size
                        if isinstance(v, Const)
                        else values[v]
                        if isinstance(v, FeatureRef)
                        else collect(v, values)
                        if isinstance(v, FeatureCollection)
                        else None  # unexpected type in collection
                    )
                    for v in col.collection
                )
                return [list(row) for row in zip(*data)]

        # map inputs to refs
        named_refs = self.named_refs
        inputs = {named_refs[n]: v for n, v in inputs.items()}
        # collect values
        return collect(self, inputs)


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
    def required_keys(cls) -> set[str]:
        """Get the required keys.

        Since the input references are dynamic and based on the collection,
        this method returns an empty set.

        Returns:
            set[str]: An empty set.
        """
        return set()

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        """Get the named input references.

        Returns:
            dict[str, FeatureRef]: A dictionary of named input references.
        """
        return self.collection.named_refs

    @property
    def flow(self) -> object:
        """Get the associated data flow graph.

        Returns:
            object: The associated data flow graph.
        """
        if len(self.collection.named_refs) == 0:
            raise NotImplementedError(
                "Constant Processors without any input references are not "
                "supported yet."
            )

        return next(iter(self.collection.named_refs.values())).flow_


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
    `FeatureCollection` object. It traverses the nested structure and gathers
    the features, maintaining the structure defined by the `FeatureCollection`.

    - :class:`FeatureRef` instances represent references to dynamic features
      within the data flow.
    - :class:`Const` instances represent constant feature values, which will
      be propagated to all collected samples
    - Primitive values are converted to :class:`Const` instances with inferred
      feature type.

    This allows the processor to combine both dynamic feature references and static
    constant values into a single nested feature collection.

    Example:
        The following example collects a set of features and induces a constant feature:

        .. code-block:: python

            # assume ref is a feature reference to an existing feature
            ref = ...
            # collect features
            features = CollectFeatures().call(
                a=[ref, ref],
                b={"x": ref}
                c="constant feature",
                d=Const(132, Value("int32"))  # manually specify the feature type
            )
            # features.a == [value, value]
            # features.b == {"x": value}
            # features.c == "constant feature"
            # features.d == 132
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
    ) -> CollectFeaturesOutputRefs:
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
            CollectFeaturesOutputRefs: The output of the CollectFeatures processor.

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
        out = Batch(
            collected=self.collection._collect_values(
                inputs=inputs, batch_size=len(index)
            )
        )
        # return collected values and index
        return out, index
