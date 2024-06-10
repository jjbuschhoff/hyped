"""Provides a class for referencing specific features within a data flow graph.

This module defines the :class:`FeatureRef` class, which represents references to
features within a data processing flow. These references are used to specify and
retrieve nested features within the data flow graph.

The :class:`FeatureRef` class supports both attribute-style and index-style access to
sub-features, enabling convenient navigation and manipulation of complex data structures
within the data flow.
"""

from __future__ import annotations

import json
from typing import TypeAlias

from datasets.features.features import Features, FeatureType, Sequence, Value
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing_extensions import Annotated

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.aggregators.ref import DataAggregationRef

FeaturePointer: TypeAlias = tuple[int, FeatureKey, object]


class FeatureRef(BaseModel):
    r"""A reference to a specific feature within a data flow graph.

    FeatureRef objects represent references to features within a data processing flow.
    These objects are used when defining a data flow but are not instantiated manually.
    Instead instances are provided by the data flow system.

    The pointer to the feature is fully defined by the
    (:class:`flow\_`, :class:`node_id\_`, :class:`key\_`)-tuple. In addition the
    feature reference still keeps track of the feature type referenced by the pointer.

    The class supports dynamic access to sub-features, enabling the specification and
    retrieval of nested features using both attribute-style and index-style access.

    Example:
        Assuming :code:`ref` is an instance of :class:`FeatureRef`:

        .. code-block:: python

            # attribute and index style reference
            sub_ref_attr = ref.some_feature
            sub_ref_index = ref['some_feature']
            # reference sequence items
            item_ref = ref[0]
            subseq_ref = ref[:4]

        All of these will return a new :class:`FeatureRef` instance pointing to the
        sub-feature within the data flow.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key_: FeatureKey
    """
    The key identifying the specific feature within the data flow.

    The key is used to locate and access the feature within the outputs
    of a node in the data flow.
    """
    node_id_: int
    """
    The identifier of the node within the data flow graph.

    This attribute represents the identifier of the node within the data flow
    graph to which the referenced feature belongs.
    """

    flow_: object
    """
    The data flow graph to which the feature reference belongs.

    This attribute represents the data flow graph to which the feature reference belongs.
    It provides context for the feature reference within the overall data processing flow.
    """

    feature_: Annotated[
        Features | FeatureType,
        # custom serialization
        PlainSerializer(
            lambda f: json.dumps(
                Features({"feature": f}).to_dict()["feature"]
            ),
            return_type=str,
            when_used="unless-none",
        ),
        # custom deserialization
        BeforeValidator(
            lambda v: (
                Features(v)
                if isinstance(v, dict)
                else Sequence(v[0])
                if isinstance(v, list) and len(v) == 1
                else v
                if isinstance(v, FeatureType)
                else Features.from_dict({"feature": json.loads(v)})["feature"]
            )
        ),
    ]
    """
    The type of the feature referenced by this instance.
    """

    @property
    def ptr(self) -> FeaturePointer:
        r"""Retrieve the pointer to the referenced feature.

        This property returns a pointer-tuple
        (:class:`node_id\_`, :class:`key\_`, :class:`flow\_`)

        Returns:
            tuple[int, FeatureKey, object]: A ptr-tuple containing the node ID,
            key, and flow.
        """
        return (self.node_id_, self.key_, self.flow_)

    def __hash__(self) -> str:
        r"""Compute the hash value of the FeatureRef instance.

        Note that the hash value of a FeatureRef instance is independent
        of the feature type, it only considers the pointer
        (:class:`node_id\_`, :class:`key\_`, :class:`flow\_`)
        of the feature.

        Returns:
            str: The hash value of the FeatureRef instance, computed
                based on its attributes.
        """
        return hash(self.ptr)

    def __getattr__(self, key: str) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via attribute-style access.

        Args:
            key (str): The name of the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.
        """
        if key.startswith("_"):
            return object.__getitem__(self, key)

        return self.__getitem__(key)

    def __getitem__(self, key: str | int | slice | FeatureKey) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via index-style access.

        Args:
            key (str | int | slice | FeatureKey): The index or key of the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.
        """
        key = key if isinstance(key, tuple) else (key,)
        key = tuple.__new__(FeatureKey, key)
        return FeatureRef(
            key_=self.key_ + key,
            feature_=key.index_features(self.feature_),
            node_id_=self.node_id_,
            flow_=self.flow_,
        )

    def __add__(self, other: FeatureRef) -> FeatureRef:
        """Perform addition with another feature.

        Args:
            other (FeatureRef): Reference to the other feature reference to add.

        Returns:
            FeatureRef: Reference to the result of the addition.
        """
        from hyped.data.flow.ops import add

        return add(self, other)

    def __sub__(self, other: FeatureRef) -> FeatureRef:
        """Perform subtraction with another feature.

        Args:
            other (FeatureRef): Reference to the other feature reference to subtract.

        Returns:
            FeatureRef: Reference to the result of the subtraction.
        """
        from hyped.data.flow.ops import sub

        return sub(self, other)

    def __mul__(self, other: FeatureRef) -> FeatureRef:
        """Perform multiplication with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to multiply.

        Returns:
            FeatureRef: Reference to the result of the multiplication.
        """
        from hyped.data.flow.ops import mul

        return mul(self, other)

    def __truediv__(self, other: FeatureRef) -> FeatureRef:
        """Perform division with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to divide.

        Returns:
            FeatureRef: Reference to the result of the division.
        """
        from hyped.data.flow.ops import truediv

        return truediv(self, other)

    def __floordiv__(self, other: FeatureRef) -> FeatureRef:
        """Perform floor division with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to floor divide.

        Returns:
            FeatureRef: Reference to the result of the floor division.
        """
        from hyped.data.flow.ops import floordiv

        return floordiv(self, other)

    def __pow__(self, other: FeatureRef) -> FeatureRef:
        """Perform exponentiation with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to use as the exponent.

        Returns:
            FeatureRef: Reference to the result of the exponentiation.
        """
        from hyped.data.flow.ops import pow

        return pow(self, other)

    def __mod__(self, other: FeatureRef) -> FeatureRef:
        """Perform modulo operation with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to use as the divisor.

        Returns:
            FeatureRef: Reference to the result of the modulo operation.
        """
        from hyped.data.flow.ops import mod

        return mod(self, other)

    def __eq__(self, other: FeatureRef) -> FeatureRef:
        """Check equality with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the equality comparison.
        """
        from hyped.data.flow.ops import eq

        return eq(self, other)

    def __ne__(self, other: FeatureRef) -> FeatureRef:
        """Check inequality with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the inequality comparison.
        """
        from hyped.data.flow.ops import ne

        return ne(self, other)

    def __lt__(self, other: FeatureRef) -> FeatureRef:
        """Check if less than another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the less-than comparison.
        """
        from hyped.data.flow.ops import lt

        return lt(self, other)

    def __le__(self, other: FeatureRef) -> FeatureRef:
        """Check if less than or equal to another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the less-than-or-equal-to comparison.
        """
        from hyped.data.flow.ops import le

        return le(self, other)

    def __gt__(self, other: FeatureRef) -> FeatureRef:
        """Check if greater than another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the greater-than comparison.
        """
        from hyped.data.flow.ops import gt

        return gt(self, other)

    def __ge__(self, other: FeatureRef) -> FeatureRef:
        """Check if greater than or equal to another feature.

        Args:
            other (FeatureRef): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the greater-than-or-equal-to comparison.
        """
        from hyped.data.flow.ops import ge

        return ge(self, other)

    def __and__(self, other: FeatureRef) -> FeatureRef:
        """Perform logical AND with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to use in the AND operation.

        Returns:
            FeatureRef: Reference to the result of the AND operation.
        """
        from hyped.data.flow.ops import and_

        return and_(self, other)

    def __or__(self, other: FeatureRef) -> FeatureRef:
        """Perform logical OR with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to use in the OR operation.

        Returns:
            FeatureRef: Reference to the result of the OR operation.
        """
        from hyped.data.flow.ops import or_

        return or_(self, other)

    def __xor__(self, other: FeatureRef) -> FeatureRef:
        """Perform logical XOR with another feature.

        Args:
            other (FeatureRef): Reference to the other feature to use in the XOR operation.

        Returns:
            FeatureRef: Reference to the result of the XOR operation.
        """
        from hyped.data.flow.ops import xor_

        return xor_(self, other)

    def sum_(self) -> DataAggregationRef:
        """Calculate the sum of the referenced feature.

        Returns:
            DataAggregationRef: Reference to the aggregated value.
        """
        from hyped.data.flow.ops import sum_

        return sum_(self)

    def mean_(self) -> DataAggregationRef:
        """Calculate the mean of the referenced feature.

        Returns:
            DataAggregationRef: Reference to the aggregated value.
        """
        from hyped.data.flow.ops import mean

        return mean(self)


NONE_REF = FeatureRef(
    key_="__NONE__", node_id_=-1, flow_=None, feature_=Value("int32")
)
"""A special instance of :class:`FeatureRef` used to mark :code:`None` features.

This special :class:`FeatureRef` instance serves multiple purposes:

 - It is used by output feature references to mark conditional output features.
 - It is used by input feature references to mark optional input features.
"""
