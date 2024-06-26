"""Provides a class for referencing specific features within a data flow graph.

This module defines the :class:`FeatureRef` class, which represents references to
features within a data processing flow. These references are used to specify and
retrieve nested features within the data flow graph.
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

from datasets.features.features import Features, FeatureType, Sequence, Value
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    STRING_LIKE_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
)
from hyped.common.feature_key import FeatureKey

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

    node_id_: str
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

    def _update(self, other: FeatureRef) -> FeatureRef:
        # update reference
        self.node_id_ = other.node_id_
        self.key_ = other.key_
        self.feature_ = other.feature_

        return self

    def __getattr__(self, key: str) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via attribute-style access.

        Args:
            key (str): The name of the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.
        """
        if key.startswith("_"):
            return object.__getattribute__(self, key)

        return self.__getitem__(key)

    def __getitem__(
        self, key: str | int | slice | FeatureKey | FeatureRef
    ) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via index-style access.

        Args:
            key (str | int | slice | FeatureKey | FeatureRef): The index or key of
                the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.

        Raises:
            TypeError: If the feature type is not a sequence but the index is a feature reference.
        """
        if isinstance(key, FeatureRef):
            # make sure the feature is a sequence
            if not check_feature_is_sequence(self.feature_):
                raise TypeError(
                    f"'{self.feature_}' object is not subscriptable."
                )

            from hyped.data.flow.ops import get_item

            # index sequence with the given key
            return get_item(self, key)

        # if the key is constant, i.e. not a feature reference,
        # we do the indexing explicitly by changing the pointer
        # of the feature reference
        key = key if isinstance(key, tuple) else (key,)
        key = tuple.__new__(FeatureKey, key)
        return FeatureRef(
            key_=self.key_ + key,
            feature_=key.index_features(self.feature_),
            node_id_=self.node_id_,
            flow_=self.flow_,
        )

    def __setitem__(
        self,
        key: str | FeatureRef | int | list[int] | slice,
        value: FeatureRef | Any,
    ) -> FeatureRef:
        """Set an item in the feature collection or sequence.

        This method sets a specified key or index in the feature collection or sequence to the given value.
        If the feature is a collection (like a dictionary), it updates the collection with the new key-value pair.
        Otherwise, it uses the set_item operation to set the value at the specified index.

        Args:
            key (str | FeatureRef | int | list[int] | slice): The key or index where the value should be set.
            value (FeatureRef | Any): The value to set at the specified key or index.

        Returns:
            FeatureRef: A reference to the updated feature.

        Raises:
            TypeError: If the feature type is neither a collection nor a sequence.
        """
        out: FeatureRef
        if isinstance(self.feature_, (Features, dict)):
            from hyped.data.flow.ops import collect

            # collect all the features in the current collection and
            # additionally the requested feature
            out = collect(
                {k: self[k] for k in self.feature_.keys()} | {key: value}
            )
            # update reference to the output reference
            return self._update(out)

        elif check_feature_is_sequence(self.feature_):
            from hyped.data.flow.ops import set_item

            # set the item in the sequence
            out = set_item(self, key, value)
            # update reference to the output reference
            return self._update(out)

        else:
            # setitem not supported
            raise TypeError(
                f"'{self.feature_}' object does not support item assignment."
            )

    def __add__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform addition with another feature.

        Performs a concatenation of the inputs in case of sequences or strings.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to add.

        Returns:
            FeatureRef: Reference to the result of the addition.
        """
        if check_feature_equals(
            self.feature_, STRING_LIKE_TYPES
        ) or check_feature_is_sequence(self.feature_):
            from hyped.data.flow.ops import concat

            return concat(self, other)

        from hyped.data.flow.ops import add

        return add(self, other)

    def __sub__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform subtraction with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to subtract.

        Returns:
            FeatureRef: Reference to the result of the subtraction.
        """
        from hyped.data.flow.ops import sub

        return sub(self, other)

    def __mul__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform multiplication with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to multiply.

        Returns:
            FeatureRef: Reference to the result of the multiplication.
        """
        from hyped.data.flow.ops import mul

        return mul(self, other)

    def __truediv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to divide.

        Returns:
            FeatureRef: Reference to the result of the division.
        """
        from hyped.data.flow.ops import truediv

        return truediv(self, other)

    def __floordiv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform floor division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to floor divide.

        Returns:
            FeatureRef: Reference to the result of the floor division.
        """
        from hyped.data.flow.ops import floordiv

        return floordiv(self, other)

    def __pow__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform exponentiation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the exponent.

        Returns:
            FeatureRef: Reference to the result of the exponentiation.
        """
        from hyped.data.flow.ops import pow

        return pow(self, other)

    def __mod__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform modulo operation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the divisor.

        Returns:
            FeatureRef: Reference to the result of the modulo operation.
        """
        from hyped.data.flow.ops import mod

        return mod(self, other)

    def __and__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform logical AND with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the AND operation.

        Returns:
            FeatureRef: Reference to the result of the AND operation.
        """
        from hyped.data.flow.ops import and_

        return and_(self, other)

    def __or__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform logical OR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the OR operation.

        Returns:
            FeatureRef: Reference to the result of the OR operation.
        """
        from hyped.data.flow.ops import or_

        return or_(self, other)

    def __xor__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform logical XOR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the XOR operation.

        Returns:
            FeatureRef: Reference to the result of the XOR operation.
        """
        from hyped.data.flow.ops import xor_

        return xor_(self, other)

    def __radd__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected addition with another feature.

        Performs a concatenation of the inputs in case of sequences or strings.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to add.

        Returns:
            FeatureRef: Reference to the result of the addition.
        """
        if check_feature_equals(
            self.feature_, STRING_LIKE_TYPES
        ) or check_feature_is_sequence(self.feature_):
            from hyped.data.flow.ops import concat

            return concat(other, self)

        from hyped.data.flow.ops import add

        return add(other, self)

    def __rsub__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected subtraction with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to subtract.

        Returns:
            FeatureRef: Reference to the result of the subtraction.
        """
        from hyped.data.flow.ops import sub

        return sub(other, self)

    def __rmul__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected multiplication with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to multiply.

        Returns:
            FeatureRef: Reference to the result of the multiplication.
        """
        from hyped.data.flow.ops import mul

        return mul(other, self)

    def __rtruediv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to divide.

        Returns:
            FeatureRef: Reference to the result of the division.
        """
        from hyped.data.flow.ops import truediv

        return truediv(other, self)

    def __rfloordiv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected floor division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to floor divide.

        Returns:
            FeatureRef: Reference to the result of the floor division.
        """
        from hyped.data.flow.ops import floordiv

        return floordiv(other, self)

    def __rpow__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected exponentiation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the exponent.

        Returns:
            FeatureRef: Reference to the result of the exponentiation.
        """
        from hyped.data.flow.ops import pow

        return pow(other, self)

    def __rmod__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected modulo operation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the divisor.

        Returns:
            FeatureRef: Reference to the result of the modulo operation.
        """
        from hyped.data.flow.ops import mod

        return mod(other, self)

    def __rand__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected logical AND with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the AND operation.

        Returns:
            FeatureRef: Reference to the result of the AND operation.
        """
        from hyped.data.flow.ops import and_

        return and_(other, self)

    def __ror__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected logical OR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the OR operation.

        Returns:
            FeatureRef: Reference to the result of the OR operation.
        """
        from hyped.data.flow.ops import or_

        return or_(other, self)

    def __rxor__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform reflected logical XOR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the XOR operation.

        Returns:
            FeatureRef: Reference to the result of the XOR operation.
        """
        from hyped.data.flow.ops import xor_

        return xor_(other, self)

    def __iadd__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace addition with another feature.

        Performs a concatenation of the inputs in case of sequences or strings.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to add.

        Returns:
            FeatureRef: Reference to the result of the addition.
        """
        out: FeatureRef
        if check_feature_equals(
            self.feature_, STRING_LIKE_TYPES
        ) or check_feature_is_sequence(self.feature_):
            from hyped.data.flow.ops import concat

            return concat(self, other)
        else:
            from hyped.data.flow.ops import add

            out = add(self, other)

        return self._update(out)

    def __isub__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace subtraction with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature reference to subtract.

        Returns:
            FeatureRef: Reference to the result of the subtraction.
        """
        from hyped.data.flow.ops import sub

        return self._update(sub(self, other))

    def __imul__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace multiplication with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to multiply.

        Returns:
            FeatureRef: Reference to the result of the multiplication.
        """
        from hyped.data.flow.ops import mul

        return self._update(mul(self, other))

    def __itruediv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to divide.

        Returns:
            FeatureRef: Reference to the result of the division.
        """
        from hyped.data.flow.ops import truediv

        return self._update(truediv(self, other))

    def __ifloordiv__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace floor division with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to floor divide.

        Returns:
            FeatureRef: Reference to the result of the floor division.
        """
        from hyped.data.flow.ops import floordiv

        return self._update(floordiv(self, other))

    def __ipow__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace exponentiation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the exponent.

        Returns:
            FeatureRef: Reference to the result of the exponentiation.
        """
        from hyped.data.flow.ops import pow

        return self._update(pow(self, other))

    def __imod__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace modulo operation with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use as the divisor.

        Returns:
            FeatureRef: Reference to the result of the modulo operation.
        """
        from hyped.data.flow.ops import mod

        return self._update(mod(self, other))

    def __iand__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace logical AND with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the AND operation.

        Returns:
            FeatureRef: Reference to the result of the AND operation.
        """
        from hyped.data.flow.ops import and_

        return self._update(and_(self, other))

    def __ior__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace logical OR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the OR operation.

        Returns:
            FeatureRef: Reference to the result of the OR operation.
        """
        from hyped.data.flow.ops import or_

        return self._update(or_(self, other))

    def __ixor__(self, other: FeatureRef | Any) -> FeatureRef:
        """Perform inplace logical XOR with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to use in the XOR operation.

        Returns:
            FeatureRef: Reference to the result of the XOR operation.
        """
        from hyped.data.flow.ops import xor_

        return self._update(xor_(self, other))

    def __eq__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check equality with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the equality comparison.
        """
        from hyped.data.flow.ops import eq

        return eq(self, other)

    def __ne__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check inequality with another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the inequality comparison.
        """
        from hyped.data.flow.ops import ne

        return ne(self, other)

    def __lt__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check if less than another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the less-than comparison.
        """
        from hyped.data.flow.ops import lt

        return lt(self, other)

    def __le__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check if less than or equal to another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the less-than-or-equal-to comparison.
        """
        from hyped.data.flow.ops import le

        return le(self, other)

    def __gt__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check if greater than another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the greater-than comparison.
        """
        from hyped.data.flow.ops import gt

        return gt(self, other)

    def __ge__(self, other: FeatureRef | Any) -> FeatureRef:
        """Check if greater than or equal to another feature.

        Args:
            other (FeatureRef | Any): Reference to the other feature to compare with.

        Returns:
            FeatureRef: Reference to the result of the greater-than-or-equal-to comparison.
        """
        from hyped.data.flow.ops import ge

        return ge(self, other)

    def __neg__(self) -> FeatureRef:
        """Perform unary negation on the feature.

        Returns:
            FeatureRef: Reference to the result of the unary negation operation.
        """
        from hyped.data.flow.ops import neg

        return neg(self)

    def __abs__(self) -> FeatureRef:
        """Compute the absolute value of the feature.

        Returns:
            FeatureRef: Reference to the result of the absolute value computation.
        """
        from hyped.data.flow.ops import abs_

        return abs_(self)

    def __invert__(self) -> FeatureRef:
        """Perform bitwise inversion on the feature.

        Returns:
            FeatureRef: Reference to the result of the bitwise inversion operation.
        """
        from hyped.data.flow.ops import invert

        return invert(self)

    def __len__(self) -> FeatureRef | int:
        """Compute the length of the feature.

        Returns an integer in case the length value of the feature ref is constant.

        Returns:
            FeatureRef | int: The length of the sequence as an integer if fixed,
                or as a FeatureRef if dynamic.

        Raises:
            NotImplementedError: If the feature is a string-like type.
            TypeError: If the feature is of an unexpected type.
        """
        from hyped.data.flow.ops import len_

        return len_(self)

    def __contains__(self, value: FeatureRef | Any) -> FeatureRef:
        """Check if the feature contains a given value.

        Args:
            value (FeatureRef | Any): The value to check for containment in the feature.

        Returns:
            FeatureRef: Reference to the result of the contains operation.
        """
        from hyped.data.flow.ops import contains

        return contains(self, value)

    def sum_(self) -> FeatureRef:
        """Calculate the sum of the referenced feature.

        Returns:
            FeatureRef: Reference to the aggregated value.
        """
        from hyped.data.flow.ops import sum_

        return sum_(self)

    def mean_(self) -> FeatureRef:
        """Calculate the mean of the referenced feature.

        Returns:
            FeatureRef: Reference to the aggregated value.
        """
        from hyped.data.flow.ops import mean

        return mean(self)


NONE_REF = FeatureRef(
    key_="__NONE__",
    node_id_="__NONE_NODE_ID__",
    flow_=None,
    feature_=Value("int32"),
)
"""A special instance of :class:`FeatureRef` used to mark :code:`None` features.

This special :class:`FeatureRef` instance serves multiple purposes:

 - It is used by output feature references to mark conditional output features.
 - It is used by input feature references to mark optional input features.
"""
