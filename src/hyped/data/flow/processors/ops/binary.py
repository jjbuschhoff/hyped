"""Module containing processor implementations of all standard binary operators.

This module defines various processors for standard binary operations, such as
comparisons, logical operations, and mathematical operations. Each processor
applies a specific binary operation to two input features and produces an output
feature based on the operation's result.

The module is structured around a few key components:

- :class:`BinaryOpInputRefs`: Defines the input references for binary operations. 
  Each binary operation takes two input features, `a` and `b`, which can be of any value type.

- :class:`BinaryOpOutputRefs`: Defines the output references for binary operations. 
  The result of the operation is stored in the `result` attribute.

- :class:`BinaryOpConfig`: Configuration class for binary operations. 
  It specifies the operation to be applied through the `op` attribute.

- :class:`BinaryOp`: Base class for all binary operation processors. 
  It provides the `batch_process` method to apply the operation to batches of data.

Specific subclasses of :class:`BinaryOp` implement various types of operations:

- :class:`Comparator`: Base class for comparison operations, such as equality and inequality checks.
- :class:`LogicalOp`: Base class for logical operations, such as AND, OR, and XOR.
- :class:`ClosedOp`: Base class for mathematical operations that require type inference for the result, 
  such as addition, subtraction, multiplication, and division.

Each specific operation, like :class:`Equals`, :class:`NotEquals`, :class:`LessThan`, :class:`LogicalAnd`, :class:`Add`, etc., 
is implemented as a subclass of the appropriate base class (:class:`Comparator`, :class:`LogicalOp`, :class:`ClosedOp`), 
with a corresponding configuration class that sets the operation to be applied.
"""
import operator
from abc import ABC
from typing import Annotated, Any, Callable, TypeVar

import numpy as np
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

INTS = {"int8", "int16", "int32", "int64"}
FLOATS = {"float16", "float32", "float64"}


class BinaryOpInputRefs(InputRefs):
    """Defines input references for binary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The first input feature. Can be any value type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The second input feature. Can be any value type."""


class BinaryOpOutputRefs(OutputRefs, ABC):
    """Defines output references for binary operations."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """The result of the binary operation. Placeholder type."""


class BinaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for binary operations."""

    op: Callable[[Any, Any], Any]
    """The binary operation to be applied."""


C = TypeVar("C", bound=BinaryOpConfig)
I = TypeVar("I", bound=BinaryOpInputRefs)
O = TypeVar("O", bound=BinaryOpOutputRefs)


class BinaryOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for binary operations."""

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the binary operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the binary operation.
        """
        return {
            "result": [
                self.config.op(a, b) for a, b in zip(inputs["a"], inputs["b"])
            ]
        }


class BoolOutputRefs(BinaryOpOutputRefs):
    """Defines output references for binary operations with boolean output."""

    result: Annotated[FeatureRef, OutputFeature(Value("bool"))]
    """The result of the binary operation. Represents a boolean feature type."""


class ComparatorConfig(BinaryOpConfig):
    """Configuration class for comparator operations."""

    op: Callable[[Any, Any], bool]
    """The comparator operation to be applied. Takes any value type inputs and outputs a boolean feature."""


class Comparator(
    BinaryOp[ComparatorConfig, BinaryOpInputRefs, BoolOutputRefs]
):
    """Base class for comparator operations.

    Comparators are characterized by their ability to take inputs of any value type
    and output a boolean feature.
    """


class EqualsConfig(ComparatorConfig):
    """Configuration class for the equality operation."""

    op: Callable[[Any, Any], bool] = operator.eq


class Equals(Comparator):
    """Processor for the equality operation."""

    CONFIG_TYPE = EqualsConfig


class NotEqualsConfig(ComparatorConfig):
    """Configuration class for the inequality operation."""

    op: Callable[[Any, Any], bool] = operator.ne


class NotEquals(Comparator):
    """Processor for the inequality operation."""

    CONFIG_TYPE = NotEqualsConfig


class LessThanConfig(ComparatorConfig):
    """Configuration class for the less-than operation."""

    op: Callable[[Any, Any], bool] = operator.lt


class LessThan(Comparator):
    """Processor for the less-than operation."""

    CONFIG_TYPE = LessThanConfig


class LessThanOrEqualConfig(ComparatorConfig):
    """Configuration class for the less-than-or-equal operation."""

    op: Callable[[Any, Any], bool] = operator.le


class LessThanOrEqual(Comparator):
    """Processor for the less-than-or-equal operation."""

    CONFIG_TYPE = LessThanOrEqualConfig


class GreaterThanConfig(ComparatorConfig):
    """Configuration class for the greater-than operation."""

    op: Callable[[Any, Any], bool] = operator.gt


class GreaterThan(Comparator):
    """Processor for the greater-than operation."""

    CONFIG_TYPE = GreaterThanConfig


class GreaterThanOrEqualConfig(ComparatorConfig):
    """Configuration class for the greater-than-or-equal operation."""

    op: Callable[[Any, Any], bool] = operator.ge


class GreaterThanOrEqual(Comparator):
    """Processor for the greater-than-or-equal operation."""

    CONFIG_TYPE = GreaterThanOrEqualConfig


class LogicalOpInputRefs(BinaryOpInputRefs):
    """Defines input references for logical operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The first input feature. Must be of boolean type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The second input feature. Must be of boolean type."""


class LogicalOpConfig(BinaryOpConfig):
    """Configuration class for logical operations."""

    op: Callable[[bool, bool], bool]
    """The logical operation to be applied. Takes two boolean inputs and returns a boolean output."""


class LogicalOp(BinaryOp[BinaryOpConfig, LogicalOpInputRefs, BoolOutputRefs]):
    """Base class for logical operations.

    Logical operators take boolean inputs and return a boolean output.
    """


class LogicalAndConfig(LogicalOpConfig):
    """Configuration class for the logical AND operation."""

    op: Callable[[bool, bool], bool] = operator.and_


class LogicalAnd(LogicalOp):
    """Processor for the logical AND operation."""

    CONFIG_TYPE = LogicalAndConfig


class LogicalOrConfig(LogicalOpConfig):
    """Configuration class for the logical OR operation."""

    op: Callable[[bool, bool], bool] = operator.or_


class LogicalOr(LogicalOp):
    """Processor for the logical OR operation."""

    CONFIG_TYPE = LogicalOrConfig


class LogicalXOrConfig(LogicalOpConfig):
    """Configuration class for the logical XOR operation."""

    op: Callable[[bool, bool], bool] = operator.xor


class LogicalXOr(LogicalOp):
    """Processor for the logical XOR operation."""

    CONFIG_TYPE = LogicalXOrConfig


class MathInputRefs(BinaryOpInputRefs):
    """Defines input references for mathematical operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureEquals(list(map(Value, list(INTS) + list(FLOATS)))),
    ]
    """The first input feature. Must be an integer or float."""

    b: Annotated[
        FeatureRef,
        CheckFeatureEquals(list(map(Value, list(INTS) + list(FLOATS)))),
    ]
    """The second input feature. Must be an integer or float."""


class ClosedOpConfig(BinaryOpConfig):
    """Configuration class for closed mathematical operations."""

    op: Callable[[int | float, int | float], int | float]
    """The closed mathematical operation to be applied.
    
    Closed mathematical operations are operations where applying the operation to two operands 
    always results in a value within the same data type domain as the operands, preserving 
    closure within the set of values.
    """


def closed_op_infer_dtype(
    config: ClosedOpConfig, inputs: MathInputRefs
) -> Value:
    """Infers the output data type for closed operations based on input types.

    For closed operations, the inferred output data type is determined by the input data types.
    If both inputs are integers, the output data type will be an integer. If both inputs are floats,
    the output data type will be a float. For a mixture of integers and floats, the output data
    type will be float.

    Args:
        config (ClosedOpConfig): The configuration for the closed operation.
        inputs (MathInputRefs): The input references.

    Returns:
        Value: The inferred data type for the output.
    """
    a_dtype = inputs.a.feature_.dtype
    b_dtype = inputs.b.feature_.dtype

    if (a_dtype in INTS) and (b_dtype in INTS):
        return Value(max(a_dtype, b_dtype))

    if (a_dtype in FLOATS) and (b_dtype in FLOATS):
        return Value(max(a_dtype, b_dtype))

    # type mismatch, one is int and one is float, keep the float
    return inputs.a.feature_ if a_dtype in FLOATS else inputs.b.feature_


class ClosedOpOutputRefs(BinaryOpOutputRefs):
    """Defines output references for closed mathematical operations."""

    result: Annotated[FeatureRef, LambdaOutputFeature(closed_op_infer_dtype)]
    """The result of the closed mathematical operation."""


class ClosedOp(BinaryOp[ClosedOpConfig, MathInputRefs, ClosedOpOutputRefs]):
    """Base class for closed mathematical operations.

    Closed operations are characterized by preserving closure within the set of values they operate on.
    Such operators include addition or subtraction, but not division, as division may result in values
    outside the set of integers or floats when dividing certain numbers. For example, while adding two
    integers always results in another integer, dividing one integer by another may produce a non-integer
    result.
    """


class AddConfig(ClosedOpConfig):
    """Configuration class for the addition operation."""

    op: Callable[[int | float, int | float], int | float] = operator.add


class Add(ClosedOp):
    """Processor for the addition operation."""

    CONFIG_TYPE = AddConfig


class SubConfig(ClosedOpConfig):
    """Configuration class for the subtraction operation."""

    op: Callable[[int | float, int | float], int | float] = operator.sub


class Sub(ClosedOp):
    """Processor for the subtraction operation."""

    CONFIG_TYPE = SubConfig


class MulConfig(ClosedOpConfig):
    """Configuration class for the multiplication operation."""

    op: Callable[[int | float, int | float], int | float] = operator.mul


class Mul(ClosedOp):
    """Processor for the multiplication operation."""

    CONFIG_TYPE = MulConfig


class PowConfig(ClosedOpConfig):
    """Configuration class for the power operation."""

    op: Callable[[int | float, int | float], int | float] = operator.pow


class Pow(ClosedOp):
    """Processor for the power operation."""

    CONFIG_TYPE = PowConfig


class ModConfig(ClosedOpConfig):
    """Configuration class for the modulus operation."""

    op: Callable[[int | float, int | float], int | float] = operator.mod


class Mod(ClosedOp):
    """Processor for the modulus operation."""

    CONFIG_TYPE = ModConfig


class FloorDivConfig(BinaryOpConfig):
    """Configuration class for the floor division operation."""

    op: Callable[[int | float, int | float], int] = operator.floordiv


class FloorDivOutputRefs(BinaryOpOutputRefs):
    """Defines output references for the floor division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("int32"))]
    """The result of the floor division operation."""


class FloorDiv(BinaryOp[FloorDivConfig, MathInputRefs, FloorDivOutputRefs]):
    """Processor for the floor division operation."""


class TrueDivConfig(BinaryOpConfig):
    """Configuration class for the true division operation."""

    op: Callable[[int | float, int | float], float] = operator.truediv


class TrueDivOutputRefs(BinaryOpOutputRefs):
    """Defines output references for the true division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("float32"))]
    """The result of the true division operation."""


class TrueDiv(BinaryOp[TrueDivConfig, MathInputRefs, TrueDivOutputRefs]):
    """Processor for the true division operation."""
