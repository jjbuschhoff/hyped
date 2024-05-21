"""Provides classes and utilities for handling input references and validators.

This module defines classes and utilities for handling input references and validators
used by data processors. It includes validators for checking feature types as well as
a collection class for managing input references and associated validators.

Classes:
    - :class:`FeatureValidator`: Validator for checking the type of a FeatureRef instance.
    - :class:`CheckFeatureEquals`: Validator for checking if a FeatureRef matches a specific feature type.
    - :class:`CheckFeatureIsSequence`: Validator for checking if a FeatureRef refers to a sequence.
    - :class:`InputRefs`: A collection of input references used by data processors.

Usage Example:
    Define a custom collection of input references with specified validators:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.processors.inputs import InputRefs, CheckFeatureEquals, CheckFeatureIsSequence
        from hyped.data.ref import FeatureRef
        from datasets.features.features import Value
        from typing_extensions import Annotated
        
        # Define a custom collection of input references with validators
        class CustomInputRefs(InputRefs):
            # String value input feature with validation
            x: Annotated[
                FeatureRef, CheckFeatureEquals(Value("string"))
            ]
            # Integer sequence input feature with validation
            y: Annotated[
                FeatureRef, CheckFeatureIsSequence(Value("int"), length=4)
            ]

    In this example, :class:`CustomInputRefs` extends :class:`InputRefs` to define a collection of input
    references with specified validators for feature type checking.
"""
from typing import Callable

from datasets.features.features import FeatureType
from pydantic import AfterValidator

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_equals,
    raise_feature_is_sequence,
)
from hyped.common.pydantic import BaseModelWithTypeValidation

from .ref import FeatureRef


class FeatureValidator(AfterValidator):
    """Validator for checking the type of a FeatureRef instance.

    This validator checks whether the provided FeatureRef instance
    conforms to a specific feature type by invoking the given validation
    function.

    Parameters:
        f (Callable[[FeatureRef, FeatureType], None]): The validation function
            to be applied to the FeatureRef.

    Raises:
        TypeError: If the provided reference is not a FeatureRef instance
        TypeError: If the feature does not conform to the expected feature type.
    """

    def __init__(self, f: Callable[[FeatureRef, FeatureType], None]) -> None:
        """Initialize the FeatureValidator instance.

        Parameters:
            f (Callable[[FeatureRef, FeatureType], None]): The validation function
                to be applied to the FeatureRef.
        """

        def check(ref: FeatureRef) -> FeatureRef:
            """Check if the provided reference conforms to the expected feature type.

            Parameters:
                ref (FeatureRef): The FeatureRef instance to be validated.

            Returns:
                FeatureRef: The validated FeatureRef instance.

            Raises:
                TypeError: If the provided reference is not a FeatureRef instance.
                TypeError: If the feature does not conform to the expected feature type.
            """
            if not isinstance(ref, FeatureRef):
                raise TypeError("Expected a FeatureRef instance.")

            try:
                f(ref, ref.feature_)
            except TypeError as e:
                raise TypeError(
                    "Feature does not conform to the expected type."
                ) from e

            return ref

        super(FeatureValidator, self).__init__(check)


class CheckFeatureEquals(FeatureValidator):
    """Validator for checking if a FeatureRef matches a specific feature type.

    This validator checks whether the feature type of the provided FeatureRef
    matches the specified feature type or a list of feature types.

    Parameters:
        feature_type (FeatureType | list[FeatureType]): The feature type or
            list of feature types to match.

    Raises:
        TypeError: If the provided reference does not match the expected
            feature type or types.
    """

    def __init__(self, feature_type: FeatureType | list[FeatureType]) -> None:
        """Initialize the CheckFeatureEquals validator.

        Args:
            feature_type (FeatureType | list[FeatureType]): The feature type or
                list of feature types to match.

        Raises:
            TypeError: If the provided reference does not match the expected
                feature type or types.
        """

        def check(ref: FeatureRef, feature: FeatureType) -> None:
            raise_feature_equals(ref.key_, feature, feature_type)

        super(CheckFeatureEquals, self).__init__(check)


class CheckFeatureIsSequence(FeatureValidator):
    """Validator for checking if a FeatureRef refers to a sequence.

    This validator checks whether the feature type of the provided
    FeatureRef is a sequence, and optionally checks if the sequence
    has the expected length.

    Parameters:
        value_type (None | FeatureType | list[FeatureType]): The expected
            type or types of the elements in the sequence.
        length (int, optional): The expected length of the sequence.
            Defaults to -1, indicating no length check.

    Raises:
        TypeError: If the provided reference does not refer to a sequence or
            if its length does not match the expected length.
    """

    def __init__(
        self,
        value_type: None | FeatureType | list[FeatureType],
        length: int = -1,
    ) -> None:
        """Initialize the CheckFeatureIsSequence validator.

        Args:
            value_type (None | FeatureType | list[FeatureType]): The expected
                type or types of the elements in the sequence.
            length (int, optional): The expected length of the sequence.
                Defaults to -1, indicating no length check.

        Raises:
            TypeError: If the provided reference does not refer to a sequence or
                if its length does not match the expected length.
        """

        def check(ref: FeatureRef, feature: FeatureType) -> None:
            raise_feature_is_sequence(ref.key_, feature, value_type)

            if -1 != length != get_sequence_length(feature):
                raise TypeError(
                    "Expected `%s` to be a sequence of length %i, got %i"
                    % (str(ref.key_), length, get_sequence_length(feature))
                )
            return ref

        super(CheckFeatureIsSequence, self).__init__(check)


class InputRefs(BaseModelWithTypeValidation):
    """A collection of input references used by data processors.

    This class represents a collection of input references used by
    data processors. It ensures that all input references adhere
    to specified feature types using pydantic validators.

    Raises:
        TypeError: If any input reference does not conform to the
            specified feature type validation.
    """

    @classmethod
    def type_validator(cls) -> None:
        """Validate the type of input references.

        This method validates that all input reference fields are instances of
        FeatureRef and are annotated with FeatureValidator instances.

        Raises:
            TypeError: If any input reference does not conform to the specified
                feature type validation.
        """
        for name, field in cls.model_fields.items():
            # each field should be a feature ref with
            # an feature validator annotation
            if not (
                issubclass(field.annotation, FeatureRef)
                and len(field.metadata) == 1
                and isinstance(field.metadata[0], FeatureValidator)
            ):
                raise TypeError(name)

    @classmethod
    @property
    def keys(cls) -> set[str]:
        """Get the keys of the input reference fields.

        Returns:
            set[str]: A set of keys corresponding to the input reference fields.
        """
        return set(cls.model_fields.keys())

    @property
    def refs(self) -> set[FeatureRef]:
        """Get the input reference instances.

        Returns:
            set[FeatureRef]: A set of input reference instances.
        """
        return set(self.named_refs.values())

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        """Get the named input reference instances.

        Returns:
            dict[str, FeatureRef]: A dictionary mapping input reference field names
            to their corresponding instances.
        """
        return {key: getattr(self, key) for key in self.model_fields.keys()}

    @property
    def flow(self) -> object:
        """Get the associated data flow graph.

        Returns:
            DataFlowGraph: The data flow graph associated with the input references.
        """
        # assumes that all feature refs refer to the same flow
        # this is checked later when a processor is added to the flow
        return getattr(self, next(iter(self.model_fields.keys()))).flow_
