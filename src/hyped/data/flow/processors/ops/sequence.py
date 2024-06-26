"""Module containing processor implementations for sequence operators."""
import operator
from abc import ABC
from collections import deque
from functools import partial
from itertools import starmap
from typing import Any, Callable, ClassVar, TypeVar

import numpy as np
from datasets import Sequence, Value
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    INDEX_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import (
    AnyFeatureType,
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef

from .binary import (
    BinaryOp,
    BinaryOpConfig,
    BinaryOpInputRefs,
    BinaryOpOutputRefs,
)
from .unary import UnaryOp, UnaryOpConfig, UnaryOpInputRefs, UnaryOpOutputRefs


class SequenceLengthInputRefs(UnaryOpInputRefs):
    """Input references for the Sequence Length operation."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The sequence feature reference to get the length of."""


class SequenceLengthOutputRefs(UnaryOpOutputRefs):
    """Output references for the Sequence Length operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the length of the sequence."""


class SequenceLengthConfig(UnaryOpConfig):
    """Configuration class for the Sequence Length operation."""

    op: Callable[[list[Any]], int] = len
    """The operation to get the length of the sequence."""


class SequenceLength(
    UnaryOp[
        SequenceLengthConfig, SequenceLengthInputRefs, SequenceLengthOutputRefs
    ]
):
    """Sequence Length Data Processor.

    This class defines the operation to get the length of a sequence feature.
    """

    CONFIG_TYPE = SequenceLengthConfig


class SequenceConcatConfig(BinaryOpConfig):
    """Configuration class for the Concat operation."""

    op: Callable[[list[Any], list[Any]], list[Any]] = operator.concat
    """The concatenate operation."""


class SequenceConcatInputRefs(BinaryOpInputRefs):
    """Input references for the Concat operation."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The first sequence feature reference."""

    b: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The second sequence feature reference."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization check to ensure sequence features align for concatenation.

        Args:
            __context: The context for the model initialization.

        Raises:
            RuntimeError: If the sequence features do not match.
        """
        # sequence features must align in
        # order to concatenate sequences
        if not check_feature_equals(
            get_sequence_feature(self.a.feature_),
            get_sequence_feature(self.b.feature_),
        ):
            raise RuntimeError(
                f"The sequence features of 'a' and 'b' must match to perform concatenation, "
                f"got `{get_sequence_feature(self.a.feature_)}` "
                f"!= `{get_sequence_feature(self.b.feature_)}`"
            )


def infer_concat_output_dtype(
    config: SequenceConcatConfig, inputs: SequenceConcatInputRefs
) -> Sequence:
    """Infer the output data type for the Concat operation.

    Args:
        config (ConcatConfig): The configuration for the Concat operation.
        inputs (ConcatInputRefs): The input references for the Concat operation.

    Returns:
        Sequence: The output sequence feature with inferred length.
    """
    # get the lengths of the input sequences
    a_length = get_sequence_length(inputs.a.feature_)
    b_length = get_sequence_length(inputs.b.feature_)
    # compute the length of the output sequence
    length = -1 if -1 in (a_length, b_length) else a_length + b_length
    # build the output sequence feature assuming that
    # the input sequence features match
    return Sequence(
        feature=get_sequence_feature(inputs.a.feature_), length=length
    )


class SequenceConcatOutputRefs(BinaryOpOutputRefs):
    """Output references for the Concat operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(infer_concat_output_dtype)
    ]
    """The feature reference to the result of the concatenation operation."""


class SequenceConcat(
    BinaryOp[
        SequenceConcatConfig, SequenceConcatInputRefs, SequenceConcatOutputRefs
    ]
):
    """Sequence Concatente Data Processor.

    This class defines the concatenation operation for sequence features.
    """


class SequenceGetItemInputRefs(InputRefs):
    """Input references for the GetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The sequence feature reference."""

    # either an index or a sequence of indices
    index: Annotated[
        FeatureRef,
        CheckFeatureEquals(INDEX_TYPES) | CheckFeatureIsSequence(INDEX_TYPES),
    ]
    """The index or sequence of indices to get items from the sequence."""


class SequenceGetItemOutputRefs(OutputRefs):
    """Output references for the GetItem operation."""

    gathered: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: (
                Sequence(
                    feature=get_sequence_feature(i.sequence.feature_),
                    length=get_sequence_length(i.index.feature_),
                )
                if check_feature_is_sequence(i.index.feature_)
                else get_sequence_feature(i.sequence.feature_)
            )
        ),
    ]
    """The feature reference to the gathered items from the sequence."""


class SequenceGetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the GetItem operation."""


class SequenceGetItem(
    BaseDataProcessor[
        SequenceGetItemConfig,
        SequenceGetItemInputRefs,
        SequenceGetItemOutputRefs,
    ]
):
    """Sequence GetItem Data Processor.

    This class defines the operation to get items from a sequence feature based
    on the given index or indices.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process a batch of data for the GetItem operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the binary operation.
        """
        op = (
            (lambda s, idx: list(map(s.__getitem__, idx)))
            if check_feature_is_sequence(io.inputs["index"])
            else (lambda s, i: s[i])
        )

        return {
            "gathered": list(
                starmap(op, zip(inputs["sequence"], inputs["index"]))
            )
        }


class SequenceSetItemInputRefs(InputRefs):
    """Input references for the SetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The sequence feature reference."""

    # either an index or a sequence of indices
    index: Annotated[
        FeatureRef,
        CheckFeatureEquals(INDEX_TYPES) | CheckFeatureIsSequence(INDEX_TYPES),
    ]
    """The index or sequence of indices to set items in the sequence."""

    value: Annotated[FeatureRef, AnyFeatureType()]
    """The value or sequence of values to set at the specified indices in the sequence."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization check to ensure values align for setting in the sequence.

        Args:
            __context: The context for the model initialization.

        Raises:
            TypeError: If the values do not match the required type or length.
        """
        if check_feature_is_sequence(self.index.feature_):
            # make sure the values match the other inputs, i.e.
            #  - values must be a sequence
            #  - values sequence must have the same length as the index
            #  - values must be of the same type as the values in the sequence
            # TODO: support broadcasting of values, i.e. values can be a single value
            #       even if the index is a sequence of indices, np.put implements the
            #       broadcasting logic anyways
            if (
                (not check_feature_is_sequence(self.value.feature_))
                or (
                    (get_sequence_length(self.value.feature_) != -1)
                    and (get_sequence_length(self.index.feature_) != -1)
                    and (
                        get_sequence_length(self.value.feature_)
                        != get_sequence_length(self.index.feature_)
                    )
                )
                or (
                    not check_feature_equals(
                        get_sequence_feature(self.value.feature_),
                        get_sequence_feature(self.sequence.feature_),
                    )
                )
            ):
                raise TypeError(
                    "Values must match the sequence type and length."
                )

        else:
            # values must be a single value of the correct type
            if not check_feature_equals(
                self.value.feature_,
                get_sequence_feature(self.sequence.feature_),
            ):
                raise TypeError("Value must match the sequence type.")


class SequenceSetItemOutputRefs(OutputRefs):
    """Output references for the SetItem operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(lambda _, i: i.sequence.feature_)
    ]
    """The feature reference to the result of the SetItem operation."""


class SequenceSetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the SetItem operation."""

    ignore_length_mismatch: bool = False
    """Whether to ignore length mismatch between index and value sequences."""


class SequenceSetItem(
    BaseDataProcessor[
        SequenceSetItemConfig,
        SequenceSetItemInputRefs,
        SequenceSetItemOutputRefs,
    ]
):
    """Sequence SetItem Data Processor.

    This class defines the operation to set items in a sequence feature based
    on the given index or indices.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process a batch of data for the SetItem operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the binary operation.
        """
        op = (
            (lambda s, idx: list(map(s.__getitem__, idx)))
            if check_feature_is_sequence(io.inputs["index"])
            else (lambda s, i: s[i])
        )

        if (
            not self.config.ignore_length_mismatch
            and check_feature_is_sequence(io.inputs["index"])
            and (
                (get_sequence_length(io.inputs["index"]) == -1)
                or (get_sequence_length(io.inputs["value"]) == -1)
            )
        ):
            # make sure the length of the value and index list match
            if any(
                len(vals) != len(idx)
                for vals, idx in zip(inputs["value"], inputs["index"])
            ):
                raise RuntimeError()  # TODO: write error message

        # convert sequences to numpy arrays and create all put operations
        sequences = list(
            map(partial(np.asarray, dtype=object), inputs["sequence"])
        )
        operations = starmap(
            np.put, zip(sequences, inputs["index"], inputs["value"])
        )
        # efficiently exhaust operations iterator, basically run all operations
        deque(operations, maxlen=0)

        # convert arrays with new values back to python lists
        return {"result": list(map(np.ndarray.tolist, sequences))}


class SequenceValueOpInputRefs(InputRefs):
    """Input references for sequence value operations."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The reference to the sequence feature."""

    value: Annotated[FeatureRef, AnyFeatureType()]
    """The reference to the value feature to be checked against the sequence values."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization check to ensure value matches the feature type of the sequence values.

        Args:
            __context: The context for the model initialization.

        Raises:
            TypeError: If the value feature does not match the feature type of the sequence values.
        """
        # make sure the value matches the feature type of the sequence values
        if not check_feature_is_sequence(
            self.sequence.feature_, self.value.feature_
        ):
            raise TypeError(
                "Value feature type does not match the sequence feature type."
            )


C = TypeVar("C", bound=BaseDataProcessorConfig)
I = TypeVar("I", bound=SequenceValueOpInputRefs)
O = TypeVar("O", bound=OutputRefs)


class SequenceValueOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for sequence value operations.

    This class provides a template for processing operations that involve a sequence and a value.

    Attributes:
        _OUTPUT_KEY (str): The key for the output feature in the result batch.
    """

    _OUTPUT_KEY: ClassVar[str]

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the sequence-value operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the sequence-value operation.
        """
        return {
            type(self)._OUTPUT_KEY: [
                self.config.op(a, b)
                for a, b in zip(inputs["sequence"], inputs["value"])
            ]
        }


class SequenceContainsOutputRefs(BinaryOpOutputRefs):
    """Output references for the :code:`contains` operation."""

    contains: Annotated[FeatureRef, OutputFeature(Value("bool"))]
    """The feature reference to the result of the :code:`contains` operation."""


class SequenceContainsConfig(BinaryOpConfig):
    """Configuration class for the :code:`contains` operation."""

    op: Callable[[list[Any], Any], bool] = operator.contains
    """The :code:`contains` operation."""


class SequenceContains(
    SequenceValueOp[
        SequenceContainsConfig,
        SequenceValueOpInputRefs,
        SequenceContainsOutputRefs,
    ]
):
    """Sequence Contains Data Drocessor.

    This class defines the operation that checks if a value is contained within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "contains"


class SequenceCountOfOutputRefs(BinaryOpOutputRefs):
    """Output references for the :code:`countOf` operation."""

    count: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the result of the :code:`countOf` operation."""


class SequenceCountOfConfig(BinaryOpConfig):
    """Configuration class for the :code:`countOf operation."""

    op: Callable[[list[Any], Any], int] = operator.countOf
    """The :code:`countOf` operation."""


class SequenceCountOf(
    SequenceValueOp[
        SequenceCountOfConfig,
        SequenceValueOpInputRefs,
        SequenceCountOfOutputRefs,
    ]
):
    """:code:`countOf` Data Processor.

    This class defines the operation that counts occurrences of a value within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "count"


class SequenceIndexOfOutputRefs(BinaryOpOutputRefs):
    """Output references for the :code:`indexOf` operation."""

    index: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the result of the :code:`indexOf` operation."""


class SequenceIndexOfConfig(BinaryOpConfig):
    """Configuration class for the :code:`indexOf` operation."""

    op: Callable[[list[Any], Any], int] = operator.indexOf
    """The :code:`indexOf` operation."""


class SequenceIndexOf(
    SequenceValueOp[
        SequenceIndexOfConfig,
        SequenceValueOpInputRefs,
        SequenceIndexOfOutputRefs,
    ]
):
    """:code:`indexOf` Data Processor.

    This class defines the operation that finds the index of a value within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "index"
