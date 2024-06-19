"""Provides a sum data aggregator for computing the sum of input features.

The :class:`SumAggregator` data aggregator calculates the sum of specified input features
over batches of data. It supports a variety of numeric and boolean input types and can be 
configured with an initial starting value for the summation. This aggregator is useful for
tasks where a cumulative sum of certain features is required.
"""
from typing import Annotated

from datasets import Value

from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.core.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class SumAggregatorInputRefs(InputRefs):
    """A collection of input references for the SumAggregator.

    This class defines the expected input feature for the SumAggregator.
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


class SumAggregatorOutputRefs(OutputRefs):
    """A collection of output references for the :class:`SumAggregator`.

    This class defines the expected output feature for the :class:`SumAggregator`.
    The output feature :code:`value` will be of type :code:`float64`.
    """

    value: Annotated[FeatureRef, OutputFeature(Value("float64"))]
    """
    The output feature reference representing the aggregated sum value.
    This value is always of type :code:`float64`.
    """


class SumAggregatorConfig(BaseDataAggregatorConfig):
    """Configuration for the :class:`SumAggregator`.

    This class defines the configuration options for the :class:`SumAggregator`,
    including the starting value for the summation.
    """

    start: float = 0
    """The initial value to start the mean calculation. Defaults to 0."""


class SumAggregator(
    BaseDataAggregator[
        SumAggregatorConfig, SumAggregatorInputRefs, SumAggregatorOutputRefs
    ]
):
    """A data aggregator that computes the sum of input features.

    This class implements a data aggregator that calculates the sum of the specified
    input feature :code:`x` over batches of data.
    """

    def initialize(self, io: IOContext) -> tuple[float, None]:
        """Initializes the aggregation with the starting value from the configuration.

        Args:
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, None]: A tuple containing the starting value and None for the state.
        """
        return {"value": self.config.start}, None

    async def extract(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> float:
        """Extracts the sum of the input feature :code:`x` from the batch of data.

        Args:
            inputs (Batch): The batch of input data.
            index (list[int]): The indices of the current batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            float: The sum of the input feature :code:`x` for the current batch.
        """
        return sum(inputs["x"])

    async def update(
        self, val: float, ctx: float, state: None, io: IOContext
    ) -> tuple[float, None]:
        """Updates the running total with the extracted value.

        Args:
            val (float): The current running total.
            ctx (float): The extracted sum from the current batch.
            state (None): The context, which is not used in this aggregator.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[float, None]: The updated running total and None for the state.
        """
        return {"value": val["value"] + ctx}, None
