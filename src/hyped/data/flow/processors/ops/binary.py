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
from abc import ABC, abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Value

from hyped.common.feature_checks import FLOAT_TYPES, INT_TYPES
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


class BinaryOpInputRefs(InputRefs):
    """Defines input references for binary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The first input feature. Can be any value type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The second input feature. Can be any value type."""


class BaseBinaryOpOutputRefs(OutputRefs, ABC):
    """Defines output references for binary operations."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """The result of the binary operation. Placeholder type."""


class BaseBinaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for binary operations."""


C = TypeVar("C", bound=BaseBinaryOpConfig)
I = TypeVar("I", bound=BinaryOpInputRefs)
O = TypeVar("O", bound=BaseBinaryOpOutputRefs)


class BaseBinaryOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for binary operations."""

    @abstractmethod
    def op(self, a: Any, b: Any) -> Any:
        """The binary operation to apply."""

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
            "result": [self.op(a, b) for a, b in zip(inputs["a"], inputs["b"])]
        }


class BoolOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for binary operations with boolean output."""

    result: Annotated[FeatureRef, OutputFeature(Value("bool"))]
    """The result of the binary operation. Represents a boolean feature type."""


class BaseComparatorConfig(BaseBinaryOpConfig):
    """Configuration class for comparator operations."""


C = TypeVar("C", bound=BaseComparatorConfig)


class BaseComparator(BaseBinaryOp[C, BinaryOpInputRefs, BoolOutputRefs]):
    """Base class for comparator operations.

    Comparators are characterized by their ability to take inputs of any value type
    and output a boolean feature.
    """

    @abstractmethod
    def op(self, a: Any, b: Any) -> bool:
        """The comparator operation to be applied.

        Takes any value type inputs and outputs a boolean feature.
        """


class EqualsConfig(BaseComparatorConfig):
    """Configuration class for the equality operation."""


class Equals(BaseComparator[EqualsConfig]):
    """Processor for the equality operation."""

    op = operator.eq


class NotEqualsConfig(BaseComparatorConfig):
    """Configuration class for the inequality operation."""


class NotEquals(BaseComparator[NotEqualsConfig]):
    """Processor for the inequality operation."""

    op = operator.ne


class LessThanConfig(BaseComparatorConfig):
    """Configuration class for the less-than operation."""


class LessThan(BaseComparator[LessThanConfig]):
    """Processor for the less-than operation."""

    op = operator.lt


class LessThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the less-than-or-equal operation."""


class LessThanOrEqual(BaseComparator[LessThanOrEqualConfig]):
    """Processor for the less-than-or-equal operation."""

    op = operator.le


class GreaterThanConfig(BaseComparatorConfig):
    """Configuration class for the greater-than operation."""


class GreaterThan(BaseComparator[GreaterThanConfig]):
    """Processor for the greater-than operation."""

    op = operator.gt


class GreaterThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the greater-than-or-equal operation."""


class GreaterThanOrEqual(BaseComparator[GreaterThanOrEqualConfig]):
    """Processor for the greater-than-or-equal operation."""

    op = operator.ge


class LogicalOpInputRefs(BinaryOpInputRefs):
    """Defines input references for logical operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The first input feature. Must be of boolean type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The second input feature. Must be of boolean type."""


class BaseLogicalOpConfig(BaseBinaryOpConfig):
    """Configuration class for logical operations."""


C = TypeVar("C", bound=BaseLogicalOpConfig)


class BaseLogicalOp(BaseBinaryOp[C, LogicalOpInputRefs, BoolOutputRefs]):
    """Base class for logical operations.

    Logical operators take boolean inputs and return a boolean output.
    """

    @abstractmethod
    def op(self, a: bool, b: bool) -> bool:
        """The logical operation to be applied.

        Takes two boolean inputs and returns a boolean output.
        """


class LogicalAndConfig(BaseLogicalOpConfig):
    """Configuration class for the logical AND operation."""


class LogicalAnd(BaseLogicalOp[LogicalAndConfig]):
    """Processor for the logical AND operation."""

    op = operator.and_


class LogicalOrConfig(BaseLogicalOpConfig):
    """Configuration class for the logical OR operation."""


class LogicalOr(BaseLogicalOp[LogicalOrConfig]):
    """Processor for the logical OR operation."""

    op = operator.or_


class LogicalXOrConfig(BaseLogicalOpConfig):
    """Configuration class for the logical XOR operation."""


class LogicalXOr(BaseLogicalOp[LogicalXOrConfig]):
    """Processor for the logical XOR operation."""

    op = operator.xor


class MathInputRefs(BinaryOpInputRefs):
    """Defines input references for mathematical operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureEquals(INT_TYPES + FLOAT_TYPES),
    ]
    """The first input feature. Must be an integer or float."""

    b: Annotated[
        FeatureRef,
        CheckFeatureEquals(INT_TYPES + FLOAT_TYPES),
    ]
    """The second input feature. Must be an integer or float."""


class BaseClosedOpConfig(BaseBinaryOpConfig):
    """Configuration class for closed mathematical operations."""


def closed_op_infer_dtype(
    config: BaseClosedOpConfig, inputs: MathInputRefs
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
    a_dtype = inputs.a.feature_
    b_dtype = inputs.b.feature_

    if (a_dtype in INT_TYPES) and (b_dtype in INT_TYPES):
        return Value(max(a_dtype.dtype, b_dtype.dtype))

    if (a_dtype in FLOAT_TYPES) and (b_dtype in FLOAT_TYPES):
        return Value(max(a_dtype.dtype, b_dtype.dtype))

    # type mismatch, one is int and one is float, keep the float
    return inputs.a.feature_ if a_dtype in FLOAT_TYPES else inputs.b.feature_


class ClosedOpOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for closed mathematical operations."""

    result: Annotated[FeatureRef, LambdaOutputFeature(closed_op_infer_dtype)]
    """The result of the closed mathematical operation."""


C = TypeVar("C", bound=BaseClosedOpConfig)


class BaseClosedOp(BaseBinaryOp[C, MathInputRefs, ClosedOpOutputRefs]):
    """Base class for closed mathematical operations.

    Closed operations are characterized by preserving closure within the set of values they operate on.
    Such operators include addition or subtraction, but not division, as division may result in values
    outside the set of integers or floats when dividing certain numbers. For example, while adding two
    integers always results in another integer, dividing one integer by another may produce a non-integer
    result.
    """

    @abstractmethod
    def op(self, a: int | float, b: int | float) -> int | float:
        """The closed mathematical operation to be applied.

        Closed mathematical operations are operations where applying the operation to two operands
        always results in a value within the same data type domain as the operands, preserving
        closure within the set of values.
        """


class AddConfig(BaseClosedOpConfig):
    """Configuration class for the addition operation."""


class Add(BaseClosedOp[AddConfig]):
    """Processor for the addition operation."""

    op = operator.add


class SubConfig(BaseClosedOpConfig):
    """Configuration class for the subtraction operation."""


class Sub(BaseClosedOp[SubConfig]):
    """Processor for the subtraction operation."""

    op = operator.sub


class MulConfig(BaseClosedOpConfig):
    """Configuration class for the multiplication operation."""


class Mul(BaseClosedOp[MulConfig]):
    """Processor for the multiplication operation."""

    op = operator.mul


class PowConfig(BaseClosedOpConfig):
    """Configuration class for the power operation."""


class Pow(BaseClosedOp[PowConfig]):
    """Processor for the power operation."""

    op = operator.pow


class ModConfig(BaseClosedOpConfig):
    """Configuration class for the modulus operation."""


class Mod(BaseClosedOp[ModConfig]):
    """Processor for the modulus operation."""

    op = operator.mod


class FloorDivConfig(BaseBinaryOpConfig):
    """Configuration class for the floor division operation."""


class FloorDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the floor division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("int32"))]
    """The result of the floor division operation."""


class FloorDiv(
    BaseBinaryOp[FloorDivConfig, MathInputRefs, FloorDivOutputRefs]
):
    """Processor for the floor division operation."""

    op = operator.floordiv


class TrueDivConfig(BaseBinaryOpConfig):
    """Configuration class for the true division operation."""


class TrueDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the true division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("float32"))]
    """The result of the true division operation."""


class TrueDiv(BaseBinaryOp[TrueDivConfig, MathInputRefs, TrueDivOutputRefs]):
    """Processor for the true division operation."""

    op = operator.truediv
