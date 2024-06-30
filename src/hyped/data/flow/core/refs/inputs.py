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
        from hyped.data.ref import FeatureRef, NONE_REF
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
                FeatureRef, CheckFeatureIsSequence(Value("int32"), length=4)
            ]
            # optional input argument
            z: Annotated[
                FeatureRef, CheckFeatureEquals(Value("int32"))
            ] = NONE_REF

    In this example, :class:`CustomInputRefs` extends :class:`InputRefs` to define a collection of input
    references with specified validators for feature type checking.
"""
from __future__ import annotations

from itertools import chain
from typing import (
    Annotated,
    Any,
    Callable,
    NotRequired,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic
from datasets.features.features import Features, FeatureType

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_equals,
    raise_feature_is_sequence,
)

from .ref import FeatureRef


class FeatureValidationError(Exception):
    """Feature Validation Error.

    Raised by feature validators and catched by the InputRefsValidator.
    """


class FeatureValidator(object):
    """Validator for checking the type of a :class:`FeatureRef` instance.

    This validator checks whether the provided :class:`FeatureRef` instance
    conforms to a specific feature type by invoking the given validation
    function.

    Args:
        f (Callable[[FeatureRef, FeatureType], None]): The validation function
            to be applied to the :class:`FeatureRef`.

    Raises:
        TypeError: If the provided reference is not a :class:`FeatureRef` instance
        FeatureValidationError: If the feature does not conform to the expected feature type.
    """

    def __init__(self, f: Callable[[FeatureRef, FeatureType], None]) -> None:
        """Initialize the FeatureValidator instance.

        Args:
            f (Callable[[FeatureRef, FeatureType], None]): The validation function
                to be applied to the FeatureRef.
        """
        self.f = f

    def __or__(self, other: FeatureValidator) -> FeatureValidator:
        """Combine this validator with another validator using logical OR.

        This method creates a new validator that checks if a :class:`FeatureRef`
        instance conforms to either of the two validators. If the feature does not
        conform to either validator, it raises a TypeError with details from both
        errors.

        Args:
            other (FeatureValidator): The other validator to combine with.

        Returns:
            FeatureValidator: A new FeatureValidator instance that validates
                against either of the combined validators.
        """

        def check_either(ref: FeatureRef, feature: FeatureType) -> None:
            try:
                # check if feature conforms to first validator
                return self.f(ref, feature)

            except FeatureValidationError as e1:
                try:
                    # fallback to second validator
                    return other.f(ref, feature)

                except FeatureValidationError as e2:
                    # TODO: e1 should also be included in traceback
                    raise FeatureValidationError(
                        f"Feature does not conform to any of the expected types: `{str(e1)}` and `{str(e2)}` "
                    ) from e2

        return FeatureValidator(check_either)


class CheckFeatureEquals(FeatureValidator):
    """Validator for checking if a FeatureRef matches a specific feature type.

    This validator checks whether the feature type of the provided FeatureRef
    matches the specified feature type or a list of feature types.

    Args:
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
            try:
                raise_feature_equals(ref.key_, feature, feature_type)
            except TypeError as e:
                raise FeatureValidationError(*e.args) from e

        super(CheckFeatureEquals, self).__init__(check)


class CheckFeatureIsSequence(FeatureValidator):
    """Validator for checking if a FeatureRef refers to a sequence.

    This validator checks whether the feature type of the provided
    FeatureRef is a sequence, and optionally checks if the sequence
    has the expected length.

    Args:
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
        value_type: None | FeatureType | list[FeatureType] = None,
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
            try:
                raise_feature_is_sequence(ref.key_, feature, value_type)
            except TypeError as e:
                raise FeatureValidationError(*e.args) from e

            if -1 != length != get_sequence_length(feature):
                raise FeatureValidationError(
                    "Expected `%s` to be a sequence of length %i, got %i"
                    % (str(ref.key_), length, get_sequence_length(feature))
                )
            return ref

        super(CheckFeatureIsSequence, self).__init__(check)


class AnyFeatureType(FeatureValidator):
    """Validator that allows any feature type.

    This validator is a permissive validator that does not enforce any specific
    feature type. It accepts any feature type without raising an error.
    """

    def __init__(self) -> None:
        """Initialize the AnyFeatureType validator."""
        super(AnyFeatureType, self).__init__(lambda r, f: None)


class InputRefs(TypedDict):
    """A collection of input references used by data processors.

    This class represents a collection of input references used by
    data processors. It ensures that all input references adhere to
    specified feature types using validators.

    It also supports optional input arguments, which can be specified
    by the `typing.NotRequired` annotation. Optional input arguments
    are not required to be present in the input data. Keep that in
    mind when implementing a custom processor.
    """


class InputRefsContainer(pydantic.BaseModel):
    """Input Reference Container.

    This container class provides helper functionality to access
    the input references, their names, the associated data flow
    graph, and the dataset features.
    """

    named_refs: dict[str, FeatureRef]
    """A dictionary mapping input reference field names to their
    corresponding instances."""

    flow: object
    """The data flow graph associated with the input references."""

    @property
    def refs(self) -> list[FeatureRef]:
        """Get the input reference instances.

        Returns:
            list[FeatureRef]: A list of unique input reference instances.
        """
        # get all unique references, i.e. references with unique pointers
        # as equality operator is overloaded
        unique = {ref.ptr: ref for ref in self.named_refs.values()}
        return list(unique.values())

    @property
    def features_(self) -> Features:
        """Get the dataset features for the input references.

        This property returns a :code:`Features` object that represents the
        features of the dataset as defined by the input references in the
        :class:`InputRefs` instance. Each key in the :code:`Features` object
        corresponds to the name of an input reference, and the associated value
        is the feature type of that input reference.

        Returns:
            Features: A dictionary-like object containing the feature types
            of the input references.
        """
        return Features(
            {key: ref.feature_ for key, ref in self.named_refs.items()}
        )


class InputRefsValidator(object):
    """Input Reference Validator.

    This class validates that all input reference fields are instances of
    FeatureRef and are annotated with FeatureValidator instances.

    It further provides functionality to validate input references
    according to their annotated validators.

    Raises:
        TypeError: If any input reference does not conform to the specified
            feature type validation.
    """

    def _validate_type_hint(self, type_hint: Any) -> bool:
        """Check if a type hint is an Annotated type with a specific origin and metadata.

        Args:
            type_hint: The type hint to check.

        Returns:
            bool: True if the type hint is an Annotated type with the specified
            origin and metadata, False otherwise.
        """
        if get_origin(type_hint) is not Annotated:
            return False

        if type_hint.__origin__ is not FeatureRef:
            return False

        return all(
            isinstance(meta, FeatureValidator)
            for meta in type_hint.__metadata__
        )

    def __init__(self, refs_type: type[InputRefs | None]) -> None:
        """Initialize the InputRefsValidator with a given reference type.

        Args:
            refs_type (type[InputRefs | None]): The type of input references to be validated.
        """
        # check if the provided reference type is valid
        self.refs_type = refs_type
        self.no_refs_type = refs_type is type(None)
        # abort processing of the input reference type
        if self.no_refs_type:
            return

        hints = get_type_hints(refs_type, include_extras=True)
        # separate type hints into required and optionals
        required = {
            key: hint
            for key, hint in hints.items()
            if get_origin(hint) is not NotRequired
        }
        optional = {
            key: get_args(hint)[0]
            for key, hint in hints.items()
            if get_origin(hint) is NotRequired
        }
        # check type hints of refs type
        for key, hint in chain(required.items(), optional.items()):
            if not self._validate_type_hint(hint):
                raise TypeError(key)  # TODO: write error message

        self.required_keys = set(required.keys())
        self.optional_keys = set(optional.keys())
        # get all validators for each type
        self.validators = {
            name: hint.__metadata__
            for name, hint in chain(required.items(), optional.items())
        }

    def validate(self, **refs: FeatureRef) -> InputRefsContainer:
        """Validate input references.

        This function validates the given input references according to the
        input references type specified in the constructor. It makes sure
        all required arguments are present and executes the validators.

        Args:
            **refs (FeatureRef): The feature references to validate.

        Returns:
            container (InputRefsContainer): A container wrapping the validates
            feature references.
        """
        if self.no_refs_type:
            raise TypeError("No reference type to validate.")

        if len(refs) == 0:
            raise RuntimeError("No references provided")

        # check all required keys are present in the reference dict
        missing = self.required_keys - set(refs.keys())
        if len(missing) > 0:
            raise TypeError(
                f"`{self.refs_type.__name__}` missing {len(missing)} required "
                f"keyword argument: {', '.join(map(repr, missing))}."
            )
        # check for unexpected keyword arguments
        unexpected = set(refs.keys()) - self.required_keys - self.optional_keys
        if len(unexpected) > 0:
            raise TypeError(
                f"`{self.refs_type.__name__}` got {len(unexpected)} unexpected "
                f"keyword arguments: {' '.join(map(repr, unexpected))}."
            )

        # run all validators
        for key, validators in self.validators.items():
            # if the key is not present in the input then
            # it must be an optional argument
            if key in refs:
                continue

            try:
                # run all validators
                for validator in validators:
                    validator.f(refs[key])
            except FeatureValidationError as e:
                raise FeatureValidationError(
                    f"Error in feature validation: {repr(key)}."
                ) from e

        # get the flow from the given references
        flow = next(iter(refs.values())).flow_

        # build the container
        return InputRefsContainer(named_refs=refs, flow=flow)
