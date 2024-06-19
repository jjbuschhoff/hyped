"""Provides classes for constant nodes in a data flow graph.

This module defines the classes and configurations for constant nodes,
which introduce constant values into the data flow. These nodes are
source nodes that generate fixed values as their output. The module
includes configurations, output references, and the constant node
implementation itself.

To add constants to the data flow, use the :code:`DataFlow.const` function. 
This function provides a high-level interface to create and add constant
nodes to the data flow graph.

Classes:
    - ConstConfig: Configuration class for constant nodes.
    - ConstOutputRefs: Output references class for constant nodes.
    - Const: Implementation of the constant node.

Example:
    Define and use a constant node in a data flow:

    .. code-block:: python

        from hyped.data.flow import DataFlow

        flow = DataFlow(...)
        const_ref = flow.const(value=42)
"""
from __future__ import annotations

from typing import Annotated, Any

from datasets import Dataset
from datasets.features.features import FeatureType
from pydantic import model_validator

from hyped.common.feature_checks import raise_object_matches_feature
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

from .base import BaseNode, BaseNodeConfig


class ConstConfig(BaseNodeConfig):
    """Configuration class for constant nodes."""

    value: Any
    """The constant value to be introduced into the data flow."""

    feature: None | FeatureType = None
    """The type of the feature.
    
    If not provided, the feature type is inferred from the value.
    """

    @model_validator(mode="after")
    def _validate_feature_type(self) -> ConstConfig:
        """Validates and infers the feature type after the configuration is initialized.

        If the feature type is not provided, it infers the feature type from the value.
        If the feature type is provided, it ensures the value matches the feature type.

        Returns:
            ConstConfig: The validated and potentially modified configuration object.
        """
        if self.feature is None:
            # infer feature type from value
            ds = Dataset.from_dict({"x": [self.value]})
            self.feature = ds.features["x"]

        else:
            # make sure feature type aligns with the value
            raise_object_matches_feature(self.value, self.feature)

        return self


class ConstOutputRefs(OutputRefs):
    """Output references for constant nodes."""

    value: Annotated[FeatureRef, LambdaOutputFeature(lambda c, _: c.feature)]
    """The output feature reference for the constant value."""


class Const(BaseNode[ConstConfig, None, ConstOutputRefs]):
    """Constant node class.

    This type of node introduces a constant value into the data flow graph.
    """

    def get_const_batch(self, batch_size: int) -> list[Any]:
        """Returns a batch of the constant value.

        Args:
            batch_size (int): The size of the batch to be generated.

        Returns:
            list[Any]: A list containing the constant value repeated
            :code:`batch_size` times.
        """
        return {"value": [self.config.value] * batch_size}

    def to(self, flow: object) -> ConstOutputRefs:
        """Adds the constant node to the data flow graph.

        This method adds the constant node to the data flow graph
        and returns the output feature reference.

        Args:
            flow (DataFlowGraph): The data flow graph object to which the node is added.

        Returns:
            ConstOutputRefs: The output feature reference for the constant node.
        """
        # add node to flow
        out_features = self._out_refs_type.build_features(self.config, None)
        node_id = flow.add_processor_node(self, None, out_features)
        # return output feature reference
        return self._out_refs_type(flow, node_id, out_features)
