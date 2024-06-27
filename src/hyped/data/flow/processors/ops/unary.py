"""Module containing processor implementations of all standard unary operators.

This module defines various processors for standard unary operations, such as
negation and logical not. Each processor applies a specific unary operation to
an input feature and produces an output feature based on the operation's result.

The module is structured around a few key components:

- :class:`UnaryOpInputRefs`: Defines the input references for unary operations. 
  Each unary operation takes one input feature, `a`.

- :class:`UnaryOpOutputRefs`: Defines the output references for unary operations. 
  The result of the operation is stored in the `result` attribute.

- :class:`UnaryOpConfig`: Configuration class for unary operations. 
  It specifies the operation to be applied through the `op` attribute.

- :class:`UnaryOp`: Base class for all unary operation processors. 
  It provides the `batch_process` method to apply the operation to batches of data.

Specific subclasses of :class:`UnaryOp` implement various types of operations:

- :class:`UnaryMathOp`: Base class for mathematical unary operations, such as negation.
- :class:`UnaryLogicalOp`: Base class for logical unary operations, such as logical not.
"""
import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Value

from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.core.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef


class UnaryOpInputRefs(InputRefs):
    """Defines input references for unary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The input feature. Can be any value type."""


class BaseUnaryOpOutputRefs(OutputRefs, ABC):
    """Defines output references for unary operations."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """The result of the unary operation. Placeholder type."""


class BaseUnaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for unary operations."""


C = TypeVar("C", bound=BaseUnaryOpConfig)
I = TypeVar("I", bound=UnaryOpInputRefs)
O = TypeVar("O", bound=BaseUnaryOpOutputRefs)


class BaseUnaryOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for unary operations."""

    @abstractmethod
    def op(self, val: Any) -> Any:
        """The unary operation to apply."""
        ...

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the unary operation.

        Args:
            inputs (Batch): The input batch containing feature 'a'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the unary operation.
        """
        return {"result": [self.op(a) for a in inputs["a"]]}


class MathUnaryOpOutputRefs(BaseUnaryOpOutputRefs):
    """Defines output references for mathematical unary operations."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(lambda config, inputs: inputs.a.feature_),
    ]
    """The result of the mathematical unary operation."""


class NegConfig(BaseUnaryOpConfig):
    """Configuration class for the negation operation."""


class Neg(BaseUnaryOp[NegConfig, UnaryOpInputRefs, MathUnaryOpOutputRefs]):
    """Processor for the negation operation."""

    op = operator.neg


class AbsConfig(BaseUnaryOpConfig):
    """Configuration class for the absolute operation."""


class Abs(BaseUnaryOp[AbsConfig, UnaryOpInputRefs, MathUnaryOpOutputRefs]):
    """Processor for the absolute operation."""

    op = operator.abs


class InvertConfig(BaseUnaryOpConfig):
    """Configuration class for the bitwise inversion operation."""


class Invert(
    BaseUnaryOp[InvertConfig, UnaryOpInputRefs, MathUnaryOpOutputRefs]
):
    """Processor for the bitwise inversion operation."""

    op = operator.invert
