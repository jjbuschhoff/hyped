"""Provides classes for managing output features and references.

The Outputs module defines classes for managing output features and references used
by data processors. It includes classes for defining output features with predefined
or dynamically generated types, as well as a collection class for managing output feature
references.

Classes:
    - `LambdaOutputFeature`: Represents a lambda function for generating an output feature type.
    - `OutputFeature`: Represents an output feature with a predefined feature type.
    - `OutputRefs`: A collection of output feature references.

Usage Example:
    Define a collection of output feature references with specified output types:

    .. code-block: python

        # Import necessary classes from the module
        from hyped.data.ref import FeatureRef
        from hyped.data.processors.outputs import OutputRefs, OutputFeature
        from datasets.features.features import Value
        from typing_extensions import Annotated

        # Define a collection of output feature references with specified output types
        class CustomOutputRefs(OutputRefs):
            # Define an output feature with a predefined feature type
            output_feature: Annotated[FeatureRef, OutputFeature(Value("string"))]

    In this example, `CustomOutputRefs` extends `OutputRefs` to define a collection of output
    feature references with specified output types.
"""
from typing import Callable, ClassVar

from datasets.features.features import Features, FeatureType

from hyped.base.config import BaseConfig
from hyped.common.pydantic import BaseModelWithTypeValidation

from .inputs import InputRefs
from .ref import FeatureRef


class LambdaOutputFeature(object):
    """Represents a lambda function for generating an output feature type.

    This class encapsulates a lambda function that generates an output
    feature type based on the provided data processor configuration and
    input references.

    Attributes:
        build_feature_type (Callable[[BaseConfig, InputRefs], FeatureType]):
            The lambda function for generating the output feature type.
    """

    def __init__(
        self, f: Callable[[BaseConfig, InputRefs], FeatureType]
    ) -> None:
        """Initialize the LambdaOutputFeature instance.

        Args:
            f (Callable[[BaseConfig, InputRefs], FeatureType]):
                The lambda function for generating the output feature type.
                Receives the configuration of the processor or augmenter and
                the input refs instance corresponding to the call.
        """
        self.build_feature_type = f


class OutputFeature(LambdaOutputFeature):
    """Represents an output feature with a predefined feature type.

    This class defines an output feature with a predefined feature
    type. It inherits from LambdaOutputFeature and initializes the
    lambda function to return the specified feature type.

    Parameters:
        feature_type (FeatureType): The predefined feature type for the output feature.
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
    __getattr__ = None  #  Disabling dynamic FeatureRef getattr function

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
                raise TypeError(name)

            # add field to feature names and extract generator
            cls._feature_names.add(name)
            cls._feature_generators[name] = field.metadata[0]

    def __init__(
        self,
        config: BaseConfig,
        inputs: InputRefs,
        node_id: int,
    ) -> None:
        """Initialize the OutputRefs instance.

        Parameters:
            config (BaseConfig): The configuration of the data processor or augmenter.
            inputs (InputRefs): The input references used by the data processor.
            node_id (int): The identifier of the data processor node.
        """
        features = Features(
            {
                key: gen.build_feature_type(config, inputs)
                for key, gen in type(self)._feature_generators.items()
            }
        )

        flow = inputs.flow
        super(OutputRefs, self).__init__(
            key_=tuple(),
            feature_=features,
            node_id_=node_id,
            flow_=flow,
            **{
                key: FeatureRef(
                    key_=key, feature_=feature, node_id_=node_id, flow_=flow
                )
                for key, feature in features.items()
            },
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
