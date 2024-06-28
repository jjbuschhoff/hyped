"""Provides high-level feature operators for data processors.

The operator module defines high-level functions for performing common operations.
These functions delegate the actual processing to specific processor classes and
return references to the resulting features represented by `FeatureRef` instances.

Feature operators are designed to simplify the process of adding processors to a data
flow by providing high-level functions for common feature operations. Each operator
encapsulates the logic for performing specific tasks, such as collecting features from
a collection (e.g., dictionary or list). These functions leverage underlying processor
classes, such as `CollectFeatures`, to execute the desired operations.

Functions:
    - :class:`collect`: Collect features from a given collection.

Usage Example:
    Collect features from a dictionary using the :class:`collect` operator:

    .. code-block:: python

        # Import the collect operator from the module
        from hyped.data.processors.operator import collect

        # Define the features of the source node
        src_features = datasets.Features({"text": datasets.Value("string")})

        # Initialize a DataFlow instance with the source features
        flow = DataFlow(features=src_features)
        
        # Collect features from the dictionary using the collect operator
        collected_features = collect(
            collection={
                "out": [
                    flow.src_features.text,
                    flow.src_features.text
                ]
            }
        )

        collected_features.out  # work with the collected features

"""

from functools import wraps
from typing import Any, Callable

from datasets import Value

from hyped.common.feature_checks import (
    STRING_LIKE_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_length,
)

from .aggregators.ops.mean import MeanAggregator
from .aggregators.ops.sum import SumAggregator
from .core.nodes.const import Const
from .core.refs.ref import FeatureRef
from .processors.ops import binary, sequence, unary
from .processors.ops.collect import CollectFeatures, NestedContainer


def _check_args(*args: FeatureRef | Any) -> tuple[FeatureRef]:
    """Ensure at least one argument is a FeatureRef and convert constants to feature references.

    This function performs two main tasks:
    1. It checks that at least one of the arguments is a FeatureRef instance.
    2. It converts any non-FeatureRef arguments into FeatureRef instances by using the flow
       associated with the first FeatureRef in the sequence.

    Args:
        *args (FeatureRef | Any): A variable number of arguments, which can be either
            FeatureRef instances or constant values.

    Returns:
        tuple[FeatureRef]: A tuple where all constant values have been converted into
            FeatureRef instances, and the original FeatureRef instances are unchanged.

    Raises:
        RuntimeError: If all inputs are constants (i.e., there are no FeatureRef instances).
    """
    if not any(isinstance(a, FeatureRef) for a in args):
        raise RuntimeError(
            "All inputs are constants. At least one input must "
            "be a FeatureRef instance."
        )

    # get the flow from the argument sequence
    flow = next(iter(arg for arg in args if isinstance(arg, FeatureRef))).flow_

    # add all constants in the argument sequence to the flow
    return tuple(
        (
            arg
            if isinstance(arg, FeatureRef)
            else Const(value=arg).to(flow).value
        )
        for arg in args
    )


def _handle_constant_inputs_for_binary_op(
    binary_op: Callable[[FeatureRef, FeatureRef], FeatureRef]
) -> Callable[[FeatureRef | Any, FeatureRef | Any], FeatureRef]:
    """Decorator to handle constant inputs for binary operations on feature references.

    This decorator allows binary operations to be applied to a mix of feature references
    and constant values. If both inputs are constants, it raises an error. If one of the
    inputs is a constant, it is converted into a feature reference before the binary
    operation is applied.

    Args:
        binary_op (Callable[[FeatureRef, FeatureRef], FeatureRef]): The binary operation
            function to be decorated.

    Returns:
        Callable[[FeatureRef | Any, FeatureRef | Any], FeatureRef]: The wrapped binary operation
        function that can handle constant inputs.

    Raises:
        ValueError: If both inputs are constants.
    """

    @wraps(binary_op)
    def wrapped_binary_op(
        a: FeatureRef | Any, b: FeatureRef | Any
    ) -> FeatureRef:
        # add constant arguments to the data flow
        a, b = _check_args(a, b)
        # apply binary operation on feature refs
        return binary_op(a, b)

    return wrapped_binary_op


def collect(
    collection: None | dict | list = None, flow: None | object = None, **kwargs
) -> FeatureRef:
    """Collects features into a feature collection.

    This function collects features into a feature collection, which can then be used as input
    to other nodes in the data flow graph. It accepts either a collection (dict or list) or keyword
    arguments representing feature values. If both collection and kwargs are provided, an error is raised.

    If any non-reference values are present in the collection, they are added as constants to the data flow graph.

    Args:
        collection (None | dict | list, optional): A collection (dict or list) containing features or
            feature values. Defaults to None.
        flow (None | object, optional): The data flow object. If not provided, the flow is inferred from
            the feature references in the collection. Defaults to None.
        **kwargs: Keyword arguments representing feature values.

    Returns:
        FeatureRef: A feature reference to the collected features.

    Raises:
        ValueError: If both collection and keyword arguments are provided.
        RuntimeError: If the flow cannot be inferred from the constant collection and no flow is provided explicitly.
    """
    if (collection is not None) and len(kwargs) > 0:
        raise ValueError(
            "Both `collection` and keyword arguments provided. "
            "Please provide only one."
        )

    # create a nested container from the inputs
    # this collection might contain constants of any type
    container = NestedContainer[FeatureRef | Any](
        data=collection if collection is not None else kwargs
    )

    if flow is None:
        # get the flow referenced in the collection in case it
        # contains any feature reference
        vals = container.flatten().values()
        vals = [v for v in vals if isinstance(v, FeatureRef)]

        if len(vals) == 0:
            raise RuntimeError(
                "Could not infer flow from constant collection, please "
                "specify the flow explicitly by setting the `flow` argument."
            )

        # get the flow from the first valid feature reference
        # in the nested collection
        flow = next(iter(vals)).flow_

    def _add_const(p: tuple[str, int], v: FeatureRef | Any) -> FeatureRef:
        return (
            v if isinstance(v, FeatureRef) else Const(value=v).to(flow).value
        )

    # add all constants in the collection to the flow
    container = container.map(_add_const, FeatureRef)

    return CollectFeatures().call(collection=container).collected


def sum_(a: FeatureRef) -> FeatureRef:
    """Calculate the sum of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        FeatureRef: A reference to the result of the sum operation.
    """
    return SumAggregator().call(x=a).value


def mean(a: FeatureRef) -> FeatureRef:
    """Calculate the mean of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        FeatureRef: A reference to the result of the mean operation.
    """
    return MeanAggregator().call(x=a).value


@_handle_constant_inputs_for_binary_op
def add(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Add two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the addition.
    """
    return binary.Add().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def sub(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Subtract one feature from another.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the subtraction.
    """
    return binary.Sub().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def mul(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Multiply two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the multiplication.
    """
    return binary.Mul().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def truediv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Divide one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the division.
    """
    return binary.TrueDiv().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def floordiv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform integer division of one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the integer division.
    """
    return binary.FloorDiv().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def pow(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Raise one feature to the power of another.

    Args:
        a (FeatureRef): The base feature.
        b (FeatureRef): The exponent feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the exponentiation.
    """
    return binary.Pow().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def mod(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Calculate the modulo of one features by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the modulo operation.
    """
    return binary.Mod().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def eq(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two features are equal.

    Args:
        a (FeatureRef): The first feature
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the equality comparison.
    """
    return binary.Equals().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def ne(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two feature references are not equal.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the inequality comparison.
    """
    return binary.NotEquals().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def lt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the less-than comparison.
    """
    return binary.LessThan().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def le(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the less-than-or-equal-to comparison.
    """
    return binary.LessThanOrEqual().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def gt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the greater-than comparison.
    """
    return binary.GreaterThan().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def ge(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the greater-than-or-equal-to comparison.
    """
    return binary.GreaterThanOrEqual().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def and_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical AND operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the and operation.
    """
    return binary.LogicalAnd().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def or_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical OR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the or operation.
    """
    return binary.LogicalOr().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def xor_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical XOR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the xor operation.
    """
    return binary.LogicalXOr().call(a=a, b=b).result


def neg(a: FeatureRef) -> FeatureRef:
    """Perform a negation operation on a feature.

    Args:
        a (FeatureRef): The feature to negate.

    Returns:
        FeatureRef: A FeatureRef instance representing the negated value of the input feature.
    """
    return unary.Neg().call(a=a).result


def abs_(a: FeatureRef) -> FeatureRef:
    """Compute the absolute value of a feature.

    Args:
        a (FeatureRef): The feature to compute the absolute value.

    Returns:
        FeatureRef: A FeatureRef instance representing the absolute value of the input feature.
    """
    return unary.Abs().call(a=a).result


def invert(a: FeatureRef) -> FeatureRef:
    """Perform a bitwise inversion operation on a feature.

    Args:
        a (FeatureRef): The feature to invert bitwise.

    Returns:
        FeatureRef: A FeatureRef instance representing the bitwise inverted value of the input feature.
    """
    return unary.Invert().call(a=a).result


def len_(a: FeatureRef) -> FeatureRef | int:
    """Determine the length of a feature.

    This function calculates the length of a feature if it is a sequence or a
    string-like type according to the following logic:

    - For sequences, it returns the length as an integer if the length is fixed, or
      as a FeatureRef if the length is dynamic.
    - For string-like features, the length is always assumed to be dynamic, and thus
      results in a FeatureRef instance.

    Args:
        a (FeatureRef): The feature for which to determine the length.

    Returns:
        FeatureRef | int: The length of the sequence as an integer if fixed, or as a FeatureRef
        if dynamic.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    if check_feature_is_sequence(a.feature_):
        # return constant in case length if fixed and a
        # feature reference to the length feature otherwise
        length = get_sequence_length(a.feature_)
        return (
            length
            if length != -1
            else sequence.SequenceLength().call(a=a).result
        )

    elif check_feature_equals(a.feature_, STRING_LIKE_TYPES):
        # implement length operation for string-like features
        raise NotImplementedError()

    else:
        # unexpected feature type
        raise TypeError(
            f"Unexpected feature type for length operation, "
            "got `{a.feature_}`."
        )


@_handle_constant_inputs_for_binary_op
def concat(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Concatenate two features.

    Args:
        a (FeatureRef): The first feature to concatenate.
        b (FeatureRef): The second feature to concatenate.

    Returns:
        FeatureRef: The concatenated feature if the features are sequences.

    Raises:
        TypeError: If the features are of unexpected types.
    """
    if check_feature_is_sequence(a.feature_):
        # return concatenated sequence feature
        return sequence.SequenceConcat().call(a=a, b=b).result

    elif check_feature_equals(a.feature_, STRING_LIKE_TYPES):
        # implement string concat operation
        raise NotImplementedError()

    else:
        # unexpected feature type for concat operation
        raise TypeError(
            f"Unexpected feature type for concat operation, "
            "got `{a.feature_}` and `{b.feature_}`."
        )


def get_item(seq: FeatureRef | Any, index: FeatureRef | Any) -> FeatureRef:
    """Retrieve an item from a sequence feature at a specified index.

    Args:
        seq (FeatureRef | Any): The sequence feature or constant value to retrieve the item from.
        index (FeatureRef | Any): The index at which to retrieve the item. Can also be a sequence of indices.

    Returns:
        FeatureRef: The feature representing the item at the specified index.
    """
    # check arguments
    seq, index = _check_args(seq, index)
    # add the getitem processor
    return sequence.SequenceGetItem().call(sequence=seq, index=index).gathered


def set_item(
    seq: FeatureRef | list[Any],
    index: FeatureRef | int | list[int] | slice,
    value: FeatureRef | Any | list[Any],
) -> FeatureRef:
    """Set an item in a sequence feature at a specified index.

    Args:
        seq (FeatureRef): The sequence feature to modify.
        index (FeatureRef | int | list[int] | slice): The index at which
            to set the item. Can also be a sequence of indices or slice.
        value (FeatureRef | Any | list[Any]): The value to set at the
            specified index. Can be a sequence of values in case the
            index is a sequence as well.

    Returns:
        FeatureRef: The feature representing the modified sequence.
    """
    if isinstance(index, slice):
        # TODO: support slices as index
        raise NotImplemented()
    # check arguments
    seq, index, value = _check_args(seq, index, value)
    # add the setitem processor
    return (
        sequence.SequenceSetItem()
        .call(sequence=seq, index=index, value=value)
        .result
    )


def contains(obj: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Check if a sequence or string-like feature contains a specified value.

    Args:
        obj (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to check for.

    Returns:
        FeatureRef: A FeatureRef instance representing whether the value is contained in the feature.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    obj, value = _check_args(obj, value)

    if check_feature_is_sequence(obj.feature_):
        return (
            sequence.SequenceContains()
            .call(sequence=obj, value=value)
            .contains
        )

    elif check_feature_equals(obj.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError(
            f"Unexpected feature type for contains operation, "
            "got `{obj.feature_}`."
        )


def count_of(obj: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Count the occurrences of a value in a sequence or string-like feature.

    Args:
        obj (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to count occurrences of.

    Returns:
        FeatureRef: A FeatureRef instance representing the count of occurrences.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    obj, value = _check_args(obj, value)

    if check_feature_is_sequence(obj.feature_):
        return sequence.SequenceCountOf().call(sequence=obj, value=value).count

    elif check_feature_equals(obj.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError(
            f"Unexpected feature type for countOf operation, "
            "got `{obj.feature_}`."
        )


def index_of(obj: FeatureRef | Any, value: FeatureRef | Any) -> FeatureRef:
    """Find the index of a value in a sequence or string-like feature.

    Args:
        obj (FeatureRef | Any): The sequence or string-like feature to check.
        value (FeatureRef | Any): The value to find the index of.

    Returns:
        FeatureRef: A FeatureRef instance representing the index of the value.

    Raises:
        TypeError: If the feature is of an unexpected type.
    """
    obj, value = _check_args(obj, value)

    if check_feature_is_sequence(obj.feature_):
        return sequence.SequenceIndexOf().call(sequence=obj, value=value).index

    elif check_feature_equals(obj.feature_, STRING_LIKE_TYPES):
        # implement contains operation for string-like features
        raise NotImplementedError()

    else:
        raise TypeError(
            f"Unexpected feature type for indexOf operation, "
            "got `{obj.feature_}`."
        )


def zip_(*sequences: FeatureRef) -> FeatureRef:
    """Zip multiple sequences together.

    Args:
        *sequences (FeatureRef): Sequences to zip together.

    Returns:
        FeatureRef: A FeatureRef instance representing the zipped sequences.
    """
    # zip collected sequences
    return (
        sequence.SequenceZip().call(sequences=collect(list(sequences))).result
    )
