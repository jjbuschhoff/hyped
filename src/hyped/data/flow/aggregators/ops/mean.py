"""Provides a mean data aggregator for computing the mean of input features.

The :class:`MeanAggregator` data aggregator calculates the mean of specified input features
over batches of data. It supports a variety of numeric and boolean input types and can be
configured with an initial starting value for the mean calculation. This aggregator is
useful for tasks where an average of certain features is required.
"""

from typing import Annotated

from datasets import Value
from pydantic import Field

from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.core.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class MeanAggregatorInputRefs(InputRefs):
    """A collection of input references for the :class:`MeanAggregator`.

    This class defines the expected input feature for the :class:`MeanAggregator`.
    The input feature :code:`x` must be one of the specified numeric or boolean types.
    """

    x: Annotated[
        FeatureRef,
        CheckFeatureEquals(
            [
                Value("bool"),
                Value("float16"),
                Value("float32"),
                Value("float64"),
                Value("int8"),
                Value("int16"),
                Value("int32"),
                Value("int64"),
                Value("uint8"),
                Value("uint16"),
                Value("uint32"),
                Value("uint64"),
            ]
        ),
    ]
    """
    The input feature reference for the aggregation. It must be of one of the specified types:

    .. code-block:: python

        Value("bool"),
        Value("float16"), Value("float32"), Value("float64"),
        Value("int8"), Value("int16"), Value("int32"), Value("int64"),
        Value("uint8"), Value("uint16"), Value("uint32"), Value("uint64")
    """


class MeanAggregatorOutputRefs(OutputRefs):
    """A collection of output references for the :class:`MeanAggregator`.

    This class defines the expected output feature for the :class:`MeanAggregator`.
    The output feature :code:`value` will be of type :code:`float64`.
    """

    value: Annotated[FeatureRef, OutputFeature(Value("float64"))]
    """
    The output feature reference representing the aggregated mean value.
    This value is always of type :code:`float64`.
    """


class MeanAggregatorConfig(BaseDataAggregatorConfig):
    """Configuration for the :class:`MeanAggregator`.

    This class defines the configuration options for the :class:`MeanAggregator`,
    including the starting value for the mean calculation.
    """

    start: float = 0
    """The initial value to start the mean calculation. Defaults to 0."""

    start_count: float = Field(default=0, ge=0)
    """The initial count to start the mean calculation. Defaults to 0."""


class MeanAggregator(
    BaseDataAggregator[
        MeanAggregatorConfig, MeanAggregatorInputRefs, MeanAggregatorOutputRefs
    ]
):
    """A data aggregator that computes the mean of input features.

    This class implements a data aggregator that calculates the mean of the specified
    input feature :code:`x` over batches of data.
    """

    def initialize(self, io: IOContext) -> tuple[float, float]:
        """Initializes the aggregation with the starting value and a count of 0.

        Args:
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, float]: A tuple containing the starting value and a count of 0.
        """
        return {"value": self.config.start}, self.config.start_count

    async def extract(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> tuple[float, int]:
        """Extracts the sum of the input feature :code:`x` and the count of items in the batch.

        Args:
            inputs (Batch): The batch of input data.
            index (list[int]): The indices of the current batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, int]: The sum of the input feature :code:`x` and the count of items in the batch.
        """
        return sum(inputs["x"]), len(index)

    async def update(
        self, val: float, ctx: tuple[float, int], state: float, io: IOContext
    ) -> tuple[float, None]:
        """Updates the running mean with the extracted value and count.

        Args:
            val (float): The current running mean.
            ctx (tuple[float, int]): The extracted sum and count from the current batch.
            state (float): The current count of items.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, float]: The updated running mean and the new count of items.
        """
        ext_val, ext_count = ctx
        return {
            "value": (val["value"] * state + ext_val) / (state + ext_count)
        }, (state + ext_count)
