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

from typing import (
    Any,
    Callable,
    ClassVar,
    ForwardRef,
    NotRequired,
    TypedDict,
    get_args,
    get_origin,
)

import pydantic
from datasets.features.features import Features, FeatureType
from pydantic import AfterValidator, ConfigDict

from hyped.common.feature_checks import (
    get_sequence_length,
    raise_feature_equals,
    raise_feature_is_sequence,
)
from hyped.common.pydantic import BaseModelWithTypeValidation

from .ref import NONE_REF, FeatureRef


class ValidationWrapper(object):
    """Wrapper class for feature validation functions.

    This class wraps a validation function and provides a call interface
    that checks if a :class:`FeatureRef` instance conforms to the expected
    feature type.
    """

    def __init__(self, f: Callable[[FeatureRef, FeatureType], None]) -> None:
        """Initialize the ValidationWrapper instance.

        Args:
            f (Callable[[FeatureRef, FeatureType], None]): The validation function
                to be wrapped.
        """
        self.unwrapped = f

    def __call__(self, ref: FeatureRef) -> FeatureRef:
        """Check if the provided reference conforms to the expected feature type.

        Args:
            ref (FeatureRef): The FeatureRef instance to be validated.

        Returns:
            FeatureRef: The validated FeatureRef instance.

        Raises:
            TypeError: If the feature does not conform to the expected feature type.
        """
        if ref is NONE_REF:
            return ref

        try:
            self.unwrapped(ref, ref.feature_)
        except TypeError as e:
            raise TypeError(
                "Feature does not conform to the expected type."
            ) from e

        return ref


class FeatureValidator(AfterValidator):
    """Validator for checking the type of a :class:`FeatureRef` instance.

    This validator checks whether the provided :class:`FeatureRef` instance
    conforms to a specific feature type by invoking the given validation
    function.

    Args:
        f (Callable[[FeatureRef, FeatureType], None]): The validation function
            to be applied to the :class:`FeatureRef`.

    Raises:
        TypeError: If the provided reference is not a :class:`FeatureRef` instance
        TypeError: If the feature does not conform to the expected feature type.
    """

    func: ValidationWrapper
    """The wrapped validation function to be applied to the :class:`FeatureRef`"""

    def __init__(self, f: Callable[[FeatureRef, FeatureType], None]) -> None:
        """Initialize the FeatureValidator instance.

        Args:
            f (Callable[[FeatureRef, FeatureType], None]): The validation function
                to be applied to the FeatureRef.
        """
        validator = ValidationWrapper(f)
        super(FeatureValidator, self).__init__(validator)

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

        def unwrapped_check_either(
            ref: FeatureRef, feature: FeatureType
        ) -> None:
            try:
                # check if feature conforms to first validator
                return self.func.unwrapped(ref, feature)

            except TypeError as e1:
                try:
                    # fallback to second validator
                    return other.func.unwrapped(ref, feature)

                except TypeError as e2:
                    # TODO: e1 should also be included in traceback
                    raise TypeError(
                        f"Feature does not conform to any of the expected types: `{str(e1)}` and `{str(e2)}` "
                    ) from e2

        return FeatureValidator(unwrapped_check_either)


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
            raise_feature_equals(ref.key_, feature, feature_type)

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
            raise_feature_is_sequence(ref.key_, feature, value_type)

            if -1 != length != get_sequence_length(feature):
                raise TypeError(
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


class InputRefsModel(pydantic.BaseModel):
    """Model to handle input references for data processing.

    This class is responsible for managing and validating input references
    which are instances of `FeatureRef`. It provides properties to access
    these references, their names, the associated data flow graph, and the
    dataset features.

    Raises:
        TypeError: If any input reference does not conform to the
        specified feature type validation.
    """

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
    def named_refs(self) -> dict[str, FeatureRef]:
        """Get the named input reference instances.

        Returns:
            dict[str, FeatureRef]: A dictionary mapping input reference field names
            to their corresponding instances.
        """
        named_refs = {
            key: getattr(self, key) for key in self.model_fields.keys()
        }
        named_refs = {
            key: ref for key, ref in named_refs.items() if ref is not NONE_REF
        }
        return named_refs

    @property
    def flow(self) -> object:
        """Get the associated data flow graph.

        Returns:
            DataFlowGraph: The data flow graph associated with the input references.
        """
        # assumes that all feature refs refer to the same flow
        # this is checked later when a processor is added to the flow
        return next(iter(self.refs)).flow_

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
    """Validate the type of input references.

    This class validates that all input reference fields are instances of
    FeatureRef and are annotated with FeatureValidator instances.

    It further provides functionality to validate a input references
    according to their annotated validators.

    Raises:
        TypeError: If any input reference does not conform to the specified
            feature type validation.
    """

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

        # evaluate forward references
        annotations = {
            key: val
            if not isinstance(val, ForwardRef)
            else val._evaluate(globals(), locals(), frozenset())
            for key, val in refs_type.__annotations__.items()
        }
        # get required and optional keys
        req_keys = (
            key
            for key, val in annotations.items()
            if get_origin(val) is not NotRequired
        )
        opt_keys = (
            key
            for key, val in annotations.items()
            if get_origin(val) is NotRequired
        )
        # build pydantic field annotations from refs type
        req_fields = {
            key: (annotations[key], pydantic.Field()) for key in req_keys
        }
        opt_fields = {
            key: (
                # unpack not required annotation and set
                # default value to none reference
                get_args(annotations[key])[0],
                pydantic.Field(default=NONE_REF),
            )
            for key in opt_keys
        }
        # build a pydantic model type
        self.model_type = pydantic.create_model(
            "InputRefsValidationModel",
            __base__=InputRefsModel,
            **req_fields,
            **opt_fields,
        )

        # validate model type
        for name, field in self.model_type.model_fields.items():
            # each field should be a feature ref with
            # an feature validator annotation
            if not (
                issubclass(field.annotation, FeatureRef)
                and len(field.metadata) == 1
                and isinstance(field.metadata[0], FeatureValidator)
            ):
                raise TypeError(name)

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling.

        Returns:
            dict[str, Any]: The state of the object.
        """
        return {"refs_type": self.refs_type}

    def __setstate__(self, state: dict[str, Any]):
        """Set the state from unpickling.

        Args:
            state (dict[str, Any]): The state of the object.
        """
        self.__init__(state["refs_type"])
        return self

    @property
    def required_keys(self) -> set[str]:
        """Get the required keys.

        Returns:
            set[str]: A set of keys corresponding to the required
            input reference fields.
        """
        if self.no_refs_type:
            return set()

        return set(
            k
            for k, f in self.model_type.model_fields.items()
            if f.is_required()
        )

    def validate(self, refs: InputRefs) -> InputRefsModel:
        """Validate the input references.

        Args:
            refs (InputRefs): The input references to be validated.

        Returns:
            InputRefsModel: The validated input references as a pydantic model.

        Raises:
            TypeError: If there is no reference type to validate.
        """
        if self.no_refs_type:
            raise TypeError("No reference type to validate.")

        return self.model_type(**refs)
