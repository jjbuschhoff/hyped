"""Provides classes for managing output features and references.

The Outputs module defines classes for managing output features and references used
by data processors. It includes classes for defining output features with predefined
or dynamically generated types, as well as a collection class for managing output feature
references.

Classes:
    - :class:`LambdaOutputFeature`: Represents a lambda function for generating an output feature type.
    - :class:`OutputFeature`: Represents an output feature with a predefined feature type.
    - :class:`OutputRefs`: A collection of output feature references.

Usage Example:
    Define a collection of output feature references with specified output types:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.ref import FeatureRef
        from hyped.data.processors.outputs import OutputRefs, OutputFeature
        from datasets.features.features import Value
        from typing_extensions import Annotated

        # Define a collection of output feature references with specified output types
        class CustomOutputRefs(OutputRefs):
            # Define an output feature with a predefined feature type
            output_feature: Annotated[FeatureRef, OutputFeature(Value("string"))]

    In this example, :class:`CustomOutputRefs` extends :class:`OutputRefs` to define a collection
    of output feature references with specified output types.
"""
from typing import Any, Callable, ClassVar

from datasets.features.features import Features, FeatureType

from hyped.base.config import BaseConfig
from hyped.common.pydantic import BaseModelWithTypeValidation

from .inputs import InputRefs
from .ref import NONE_REF, FeatureRef


class LambdaOutputFeature(object):
    """Represents a lambda function for generating an output feature type.

    This class encapsulates a lambda function that generates an output
    feature type based on the provided data processor configuration and
    input references. If the lambda function returns None, it indicates
    that the feature isn't populated and cannot be used. Accessing such
    a feature will raise an AttributeError.
    """

    def __init__(
        self, f: Callable[[BaseConfig, None | InputRefs], None | FeatureType]
    ) -> None:
        """Initialize the LambdaOutputFeature instance.

        Args:
            f (Callable[[BaseConfig, InputRefs], FeatureType]):
                The lambda function for generating the output feature type.
                Receives the configuration of the processor or augmenter and
                the input refs instance corresponding to the call. If the
                lambda function returns None, it indicates that the feature
                isn't populated and cannot be used.
        """
        self.build_feature_type = f


class ConditionalOutputFeature(LambdaOutputFeature):
    """Represents a conditionally generated output feature type.

    This class extends LambdaOutputFeature to include a condition that
    determines whether the output feature type should be generated.
    """

    def __init__(
        self,
        feature_type: FeatureType,
        cond: Callable[[BaseConfig, None | InputRefs], bool],
    ) -> None:
        """Initialize the ConditionalOutputFeature instance.

        Args:
            feature_type (FeatureType):
                The feature type to be generated if the condition is met.
            cond (Callable[[BaseConfig, InputRefs], bool]):
                A lambda function that determines whether the feature type
                should be generated based on the configuration and input refs.
        """
        super(ConditionalOutputFeature, self).__init__(
            lambda c, i: feature_type if cond(c, i) else None
        )


class OutputFeature(LambdaOutputFeature):
    """Represents an output feature with a predefined feature type.

    This class defines an output feature with a predefined feature
    type. It inherits from LambdaOutputFeature and initializes the
    lambda function to return the specified feature type.
    """

    def __init__(self, feature_type: FeatureType) -> None:
        """Initialize the OutputFeature instance.

        Args:
            feature_type (FeatureType): The predefined feature type for the output feature.
        """
        super(OutputFeature, self).__init__(lambda _, __: feature_type)


class OutputRefs(FeatureRef, BaseModelWithTypeValidation):
    """A collection of output feature references.

    This class represents a collection of output feature references that
    represent the outputs of a data processor. It inherits the FeatureRef
    type, providing access to specific features within the output data flow.
    """

    _feature_generators: ClassVar[dict[str, LambdaOutputFeature]]
    _feature_names: ClassVar[set[str]]

    @classmethod
    def type_validator(cls) -> None:
        """Validate the type of output references.

        This method validates that all output reference fields are instances of
        FeatureRef and are annotated with LambdaOutputFeature instances.

        Raises:
            TypeError: If any output reference does not conform to the specified output feature type validation.
        """
        cls._feature_generators = {}
        cls._feature_names = set()
        # ignore all fields from the feature ref base type
        ignore_fields = FeatureRef.model_fields.keys()

        for name, field in cls.model_fields.items():
            if name in ignore_fields:
                continue
            # each field should be a feature ref with
            # an output feature type annotation
            if not (
                issubclass(field.annotation, FeatureRef)
                and len(field.metadata) == 1
                and isinstance(field.metadata[0], LambdaOutputFeature)
            ):
                raise TypeError(
                    f"Field '{name}' must be a FeatureRef annotated with a 'LambdaOutputFeature'."
                )
            # add field to feature names and extract generator
            cls._feature_names.add(name)
            cls._feature_generators[name] = field.metadata[0]

    @classmethod
    def build_features(cls, config: BaseConfig, inputs: InputRefs) -> Features:
        """Build output features based on the given configuration and input references.

        This method constructs the output features from the feature annotations of the
        class. Conditional Output Features (i.e. annotations that evaluate to None) are
        filtered out.

        Args:
            config (BaseConfig): The configuration object..
            inputs (InputRefs): The input references that define the input features
                to be processed.

        Returns:
            Features: The constructed features.
        """
        features = {
            key: gen.build_feature_type(config, inputs)
            for key, gen in cls._feature_generators.items()
        }

        return Features({k: f for k, f in features.items() if f is not None})

    def __init__(
        self,
        flow: object,
        node_id: str,
        features: Features,
    ) -> None:
        """Initialize the OutputRefs instance.

        Args:
            flow (DataFlowGraph): The data flow graph.
            node_id (str): The node id of the node generating the ouput.
            features (Features): The output features, typically build by the :class:`build_features` method.
        """
        super(OutputRefs, self).__init__(
            key_=tuple(),
            feature_=features,
            node_id_=node_id,
            flow_=flow,
            **{
                key: FeatureRef(
                    key_=key,
                    feature_=features[key],
                    node_id_=node_id,
                    flow_=flow,
                )
                if key in features
                else NONE_REF
                for key in type(self)._feature_names
            },
        )

    def __getattribute__(self, name: str) -> Any:
        """Retrieve the attribute with the specified name.

        If the requested attribute is a conditional feature reference and
        the condition is not met, an AttributeError is raised.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The attribute value.

        Raises:
            AttributeError: If the attribute is a conditional feature reference
                            and the condition is not met.
        """
        obj = super(OutputRefs, self).__getattribute__(name)
        # check if the requested attribute is a conditional
        # feature ref with the condition not met
        if (
            isinstance(obj, FeatureRef)
            and (name in type(self)._feature_names)
            and obj is NONE_REF
        ):
            # forward the request to getattr
            raise AttributeError()

        # return the attribute
        return obj

    def __getattr__(self, name: str) -> Any:
        """Handle the case where an attribute is not found.

        This method is called when an attribute is not found in the usual places
        (i.e., it is not an instance attribute nor is it found in the class tree
        for self). It raises an AttributeError if the attribute does not exist or
        if the attribute is a conditional feature reference and the condition is
        not met.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The attribute value.

        Raises:
            AttributeError: If the attribute does not exist or is a conditional
                            feature reference and the condition is not met.
        """
        if name in type(self)._feature_names:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}': "
                "A condition output feature with the name exists but "
                "its output condition is not met."
            )

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    @property
    def refs(self) -> set[FeatureRef]:
        """The set of all output feature reference instances.

        Returns:
            set[FeatureRef]: A set of FeatureRef instances.
        """
        ignore_fields = FeatureRef.model_fields.keys()
        return set(
            [
                getattr(self, key)
                for key in self.model_fields.keys()
                if key not in ignore_fields
            ]
        )
