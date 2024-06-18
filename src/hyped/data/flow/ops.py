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

from .aggregators.ops.mean import MeanAggregator
from .aggregators.ops.sum import SumAggregator
from .core.nodes.const import Const
from .core.refs.ref import AggregationRef, FeatureRef
from .processors.ops import binary
from .processors.ops.collect import CollectFeatures, NestedContainer


def _handle_constant_inputs_for_binary_op(
    binary_op: Callable[[FeatureRef, FeatureRef], FeatureRef]
) -> Callable[[FeatureRef, FeatureRef], FeatureRef]:
    """Decorator to handle constant inputs for binary operations on feature references.

    This decorator allows binary operations to be applied to a mix of feature references
    and constant values. If both inputs are constants, it raises an error. If one of the
    inputs is a constant, it is converted into a feature reference before the binary
    operation is applied.

    Args:
        binary_op (Callable[[FeatureRef, FeatureRef], FeatureRef]): The binary operation
            function to be decorated.

    Returns:
        Callable[[FeatureRef, FeatureRef], FeatureRef]: The wrapped binary operation
        function that can handle constant inputs.

    Raises:
        ValueError: If both inputs are constants.
    """

    @wraps(binary_op)
    def wrapped_binary_op(
        a: FeatureRef | Any, b: FeatureRef | Any
    ) -> FeatureRef:
        if not isinstance(a, FeatureRef) and not isinstance(b, FeatureRef):
            raise RuntimeError(
                "Both inputs to the binary operation are constants. "
                "At least one input must be a FeatureRef."
            )

        # get the flow from the valid feature reference
        flow = (a if isinstance(a, FeatureRef) else b).flow_

        # collect constant value a
        if not isinstance(a, FeatureRef):
            a = Const(value=a).to(flow).value

        # collect constant value b
        if not isinstance(b, FeatureRef):
            b = Const(value=b).to(flow).value

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
        raise ValueError()  # TODO: only one allowed

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


def sum_(a: FeatureRef) -> AggregationRef:
    """Calculate the sum of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        AggregationRef: A reference to the result of the sum operation.
    """
    return SumAggregator().call(x=a)


def mean(a: FeatureRef) -> AggregationRef:
    """Calculate the mean of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        AggregationRef: A reference to the result of the mean operation.
    """
    return MeanAggregator().call(x=a)


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
