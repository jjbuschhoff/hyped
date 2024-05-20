"""Provides a NoOp data processor that passes input features directly to output features.

The `NoOp` data processor is a simple operation that passes input features directly
to output features without any transformation. It serves as a placeholder or a baseline
in data processing pipelines.
"""
from __future__ import annotations

from typing import Annotated, Any

from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


class NoOpInputRefs(InputRefs):
    """A collection of input references for the NoOp data processor."""

    x: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]
    """
    The input reference representing the input feature.
    The input feature can be of any type.
    """


class NoOpOutputRefs(OutputRefs):
    """A collection of output feature references for the NoOp data processor."""

    y: Annotated[FeatureRef, LambdaOutputFeature(lambda _, i: i.x.feature_)]
    """
    The output reference representing the output feature. The output feature
    type is copied from the feature type of the input.
    """


class NoOpConfig(BaseDataProcessorConfig):
    """Empty Configuration class for the NoOp data processor."""


class NoOp(BaseDataProcessor[NoOpConfig, NoOpInputRefs, NoOpOutputRefs]):
    """NoOp data processor class.

    This class defines the NoOp data processor, which simply passes the input
    feature through to the output feature without any processing.
    """

    def __init__(self) -> None:
        """Initialize the NoOp data processor."""
        super(NoOp, self).__init__(config=NoOpConfig())

    @classmethod
    def from_config(cls, config: NoOpConfig) -> NoOp:
        """Creates a NoOp data processor instance from the provided configuration.

        Args:
            config (NoOpConfig): The configuration object for the NoOp processor.

        Returns:
            NoOp: An instance of the NoOp data processor.
        """
        return cls()

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> tuple[Batch, list[int]]:
        """Processes a batch of inputs and returns the corresponding batch of outputs.

        This method implements the processing logic for the NoOp data processor,
        which simply passes the input feature through to the output feature without
        any processing.

        Args:
            inputs (Batch): The batch of input samples.
            index (int): The index associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            Batch: The batch of output samples.
        """
        return {"y": inputs["x"]}, index
