"""This module contains the implementation of a JSON parser data processor.

The processor is designed to parse JSON strings into structured feature types
using Pydantic for deserialization and validation.
"""
import json
from typing import Annotated

from datasets.features.features import Features, FeatureType, Sequence, Value
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer

from hyped.common.pydantic import pydantic_model_from_features
from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


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


class JsonParser(
    BaseDataProcessor[
        JsonParserConfig, JsonParserInputRefs, JsonParserOutputRefs
    ]
):
    """The JSON parser data processor.

    This processor is designed to take a JSON string as input and parse it into
    structured data based on a predefined schema. The schema can be defined using
    either a `Features` object, a single `FeatureType`, or a `Sequence` of `FeatureType`s.

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
            features={"parsed": Sequence(self.config.scheme)}
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

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> tuple[Batch, list[int]]:
        """Process a batch of JSON strings.

        Args:
            inputs (Batch): Batch of input data.
            index (list[int]): List of indices for the current batch.
            rank (int): Rank of the current batch in the processing order.

        Returns:
            tuple[Batch, list[int]]: Tuple of the parsed batch and the list of indices.
        """
        json_strings = inputs["json_str"]
        batch_json_string = '{"parsed": [%s]}' % ",".join(json_strings)
        # load batch in one validation step
        parsed_batch = self._feature_model.model_validate_json(
            batch_json_string
        )
        # return parsed object
        return parsed_batch.model_dump(), index
