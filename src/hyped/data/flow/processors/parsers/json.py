"""This module contains the implementation of a JSON parser data processor.

The processor is designed to parse JSON strings into structured feature types
using Pydantic for deserialization and validation.
"""
import json
import warnings
from typing import Annotated

from datasets.features.features import Features, FeatureType, Sequence, Value
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from pydantic_core import ValidationError

from hyped.common.pydantic import pydantic_model_from_features
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.core.refs.outputs import (
    ConditionalOutputFeature,
    LambdaOutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef


class JsonParserInputRefs(InputRefs):
    """Inputs for the JsonParser."""

    json_str: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]
    """
    The input JSON string feature.
    """


class JsonParserOutputRefs(OutputRefs):
    """Outputs for the JsonParser."""

    parsed: Annotated[FeatureRef, LambdaOutputFeature(lambda c, _: c.scheme)]
    """The output parsed feature."""
    error: Annotated[
        FeatureRef,
        ConditionalOutputFeature(
            Value("string"), lambda c, _: c.catch_validation_errors
        ),
    ]
    """Feature that is true if the parsing resulted in an error."""


class JsonParserConfig(BaseDataProcessorConfig):
    """Configuration class for the JsonParser."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scheme: Annotated[
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
                else Sequence(v)
                if isinstance(v, list)
                else v
                if isinstance(v, FeatureType)
                else Features.from_dict({"feature": json.loads(v)})["feature"]
            )
        ),
    ]
    """
    The scheme defining the structure of the parsed JSON.
    """
    catch_validation_errors: bool = False
    """Catch validation errors. This creates an additional output feature `errors`
    that indicates if an error was thrown.
    """


class JsonParser(
    BaseDataProcessor[
        JsonParserConfig, JsonParserInputRefs, JsonParserOutputRefs
    ]
):
    """The JSON parser data processor.

    This processor is designed to take a JSON string as input and parse it into
    structured data based on a predefined schema. The schema can be defined using
    either a :code:`Features` object, a single :code:`FeatureType`, or a
    :code:`Sequence` of :code:`FeatureType` instances.

    The parsed data is then validated and transformed into the desired format using
    Pydantic models, ensuring that the data conforms to the specified schema. This
    processor can handle batch processing, where multiple JSON strings are parsed and
    validated in a single operation, improving efficiency and performance.
    """

    def __init__(
        self, config: None | JsonParserConfig = None, **kwargs
    ) -> None:
        """Initialize the JsonParser with the given configuration.

        Args:
            config (JsonParserConfig): Configuration for the JSON parser.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(JsonParser, self).__init__(config, **kwargs)
        self._feature_model = self._build_feature_model()

    def _build_feature_model(self) -> BaseModel:
        """Build the Pydantic model for the features.

        Returns:
            BaseModel: Pydantic model for the features.
        """
        return pydantic_model_from_features(
            features={"parsed": self.config.scheme}
        )

    def __getstate__(self):
        """Prepare the state for serialization.

        Returns:
            dict: State dictionary without the _feature_model.
        """
        d = self.__dict__.copy()
        d.pop("_feature_model")
        return d

    def __setstate__(self, d):
        """Restore the state after deserialization.

        Args:
            d (dict): State dictionary.
        """
        self.__dict__ = d
        self._feature_model = self._build_feature_model()

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Processes a single input sample synchronously and returns the corresponding output sample.

        This method parses a JSON string contained within the input sample and validates it
        against a predefined model. If the configuration is set to catch validation errors,
        it will handle any validation exceptions and return a default model with an error message.
        Otherwise, it will directly parse and validate the JSON string.

        Args:
            inputs (Sample): The input sample containing the JSON string to be processed.
            index (int): The index associated with the input sample.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Sample: The processed output sample. If :code:`config.catch_validation_errors` is True, the output
                    includes the parsed data or a default model and an error message. If not,
                    the output only includes the parsed data.

        Raises:
            ValidationError: If validation of the JSON string fails and :code:`config.catch_validation_errors` is False.
        """
        json_string = f"""{{"parsed": {inputs["json_str"]}}}"""
        if self.config.catch_validation_errors:
            # try parsing json, return default model (Nones) + failed otherwise
            try:
                parsed = self._feature_model.model_validate_json(json_string)
                error = None

            except ValidationError as e:
                parsed = self._feature_model()
                error = str(e)

            return Sample(
                parsed=parsed.model_dump()["parsed"],
                error=error,
            )

        else:
            # parse the json string and return
            parsed = self._feature_model.model_validate_json(json_string)
            return Sample(
                parsed=parsed.model_dump()["parsed"],
            )
