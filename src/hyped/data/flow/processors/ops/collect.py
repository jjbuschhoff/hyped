"""Module for collecting features from nested structures in data processing.

This module defines a data processor (:class:`CollectFeatures`) that collects features
from a nested structure defined by a :class:`NestedCollection` object. The nested
structure can be arbitrarily deep, consisting of dictionaries and lists, with
the leaves of the structure being :class:`FeatureRef` objects.

This is particularly useful for defining the output of a data flow.
The desired output features, potentially from different nodes, can
be referenced in a `NestedCollection`, which the processor can then
collect. This simplifies the management and retrieval of the specified
output features.

The high-level function for utilizing the :class:`CollectFeatures` processor is
the :class:`hyped.data.flow.ops.collect` function. This function serves as the entry point
for collecting features within the data processing pipeline. By providing a convenient
interface, it allows users to easily integrate feature collection into their data flow
graph.
"""

from functools import cache
from typing import Any, Hashable

from datasets.features.features import Features, FeatureType, Sequence
from typing_extensions import Annotated

from hyped.common.container import NestedContainer
from hyped.common.feature_checks import check_feature_equals
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import InputRefs, InputRefsModel
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class CollectFeaturesConfig(BaseDataProcessorConfig):
    """Configuration class for the CollectFeatures data processor."""


def _path_to_str(path: tuple[Hashable | int]) -> str:
    """Convert a path tuple to a dot-separated string.

    Args:
        path (tuple[Hashable | int]): The path tuple.

    Returns:
        str: The dot-separated string representation of the path.
    """
    return ".".join(map(str, path))


class CollectFeaturesInputRefsModel(InputRefsModel):
    """Input references class for the :class:`CollectFeatures` data processor."""

    collection: NestedContainer[FeatureRef]
    """The nested collection of feature references to collect."""

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        """Get named references from the input collection.

        Returns:
            dict[str, FeatureRef]: A dictionary of named feature references.
        """
        return {
            _path_to_str(key): ref
            for key, ref in self.collection.flatten().items()
        }

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


def _infer_feature_type(
    container: NestedContainer[FeatureRef],
) -> Features:
    """Infer the feature type from a nested container of feature references.

    Args:
        container (NestedContainer[FeatureRef]): The nested container of feature references.

    Returns:
        Features: The inferred feature type.
    """
    if isinstance(container.data, dict):
        return Features(
            {k: _infer_feature_type(v) for k, v in container.data.items()}
        )

    if isinstance(container.data, list):
        assert len(container.data) > 0
        # get the feature types of the list items
        item_types = map(_infer_feature_type, container.data)
        f = next(item_types)
        # make sure all feature types align
        for ff in item_types:
            if not check_feature_equals(f, ff):
                raise TypeError(
                    "Expected all items of a sequence to be of the "
                    "same feature type, got %s != %s" % (str(f), str(ff))
                )
        # build sequence feature
        return Sequence(f, length=len(container.data))

    # return the feature type of the referenced feature
    assert isinstance(container.data, FeatureRef)
    return container.data.feature_


class CollectFeaturesOutputRefs(OutputRefs):
    """Output references class for the :class:`CollectFeatures` data processor."""

    collected: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, inputs: _infer_feature_type(inputs.collection)
        ),
    ]
    """Reference to the collected feature."""


class CollectFeatures(
    BaseDataProcessor[
        CollectFeaturesConfig,
        None,  # unused
        CollectFeaturesOutputRefs,
    ]
):
    """Data processor for collecting features into a new (nested) feature.

    This processor collects features from a nested structure defined by a
    `NestedCollection` object. It traverses the nested structure and gathers
    the features, maintaining the structure defined by the collection.
    """

    def call(
        self, collection: NestedContainer[FeatureRef]
    ) -> CollectFeaturesOutputRefs:
        """Collect features from a nested container and return output references.

        This method takes a nested container of feature references and returns
        the collected feature references wrapped in an output references object.

        Args:
            collection (NestedContainer[FeatureRef]): A nested container holding
                feature references to be collected.

        Returns:
            CollectFeaturesOutputRefs: An object containing the references to the
            collected features, preserving the structure defined by the input
            collection.
        """
        inputs = CollectFeaturesInputRefsModel(collection=collection)
        # compute output features and add the processor to the data flow
        out_features = self._out_refs_type.build_features(self.config, inputs)
        node_id = inputs.flow.add_processor_node(self, inputs, out_features)
        # return the output feature refs
        return self._out_refs_type(inputs.flow, node_id, out_features)

    @cache
    def _lookup(self, io: IOContext) -> NestedContainer[str]:
        """Generate lookup mapping for the collected features.

        Args:
            io (IOContext): The IO context.

        Returns:
            NestedContainer[str]: The container of lookup strings.
        """

        def build_nested_lookup(
            feature: FeatureType, path: tuple = tuple()
        ) -> NestedContainer[tuple[Hashable | int, ...]]:
            if _path_to_str(path) in io.inputs:
                # trivial case of the recursion
                return NestedContainer[tuple[Hashable | int, ...]](data=path)

            # parse feature dictionary
            if isinstance(feature, (Features, dict)):
                return NestedContainer[tuple[Hashable | int, ...]](
                    data={
                        k: build_nested_lookup(v, path + (k,))
                        for k, v in feature.items()
                    }
                )
            # parse sequence feature
            if isinstance(feature, Sequence):
                assert feature.length >= 0
                return NestedContainer[tuple[Hashable | int, ...]](
                    data=[
                        build_nested_lookup(feature.feature, path + (i,))
                        for i in range(feature.length)
                    ]
                )

            # not a nested feature
            return NestedContainer[tuple[Hashable | int, ...]](data=path)

        # build the lookup container
        container = build_nested_lookup(io.outputs["collected"])
        container = container.map(lambda path, _: _path_to_str(path), str)
        # make sure the lookup contains all inputs
        assert set(container.flatten().values()) == set(io.inputs.keys())

        return container

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process batches of inputs.

        Args:
            inputs (Batch): The input batch.
            index (list[int]): The index of the batch.
            rank (int): The rank of the batch.
            io (IOContext): The execution context.

        Returns:
            Batch: The processed batch.
        """
        # convert dict of lists to list of dicts
        keys = inputs.keys()
        samples = [dict(zip(keys, values)) for values in zip(*inputs.values())]
        # collect values from each sample
        return {
            "collected": [
                self._lookup(io).map(lambda _, key: sample[key], Any).unpack()
                for sample in samples
            ]
        }
