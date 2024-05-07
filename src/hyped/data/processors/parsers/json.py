"""Json Parser."""
from typing import Any

from datasets import Features, Sequence, Value
from datasets.features.features import FeatureType
from pydantic import BaseModel

from hyped.common.feature_checks import raise_feature_equals
from hyped.common.feature_key import FeatureKey
from hyped.common.pydantic import pydantic_model_from_features
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class JsonParserConfig(BaseDataProcessorConfig):
    """Json Parser Config.

    Parse a json string using pydantic as parsing backend.

    This processor is much faster then the `RepairJsonParser`
    but it assumes that all json strings valid.

    Attributes:
        json (FeatureKey): feature to parse
        scheme (FeatureType): feature scheme of the parsed object
    """

    json_str: FeatureKey
    scheme: FeatureType


class JsonParser(BaseDataProcessor[JsonParserConfig]):
    """Json Parser.

    Parse a json string using pydantic as parsing backend.

    This processor is much faster then the `RepairJsonParser`
    but it assumes that all json strings valid.
    """

    def __init__(self, config: JsonParserConfig) -> None:
        """Constructor."""
        super(JsonParser, self).__init__(config)
        self._feature_model = self._build_feature_model()

    def _build_feature_model(self) -> BaseModel:
        """Build pydantic model for object validation."""
        return pydantic_model_from_features(
            features={"parsed": Sequence(self.config.scheme)}
        )

    def __getstate__(self):
        """Avoid pickle pydantic model type defined at runtime."""
        d = self.__dict__.copy()
        d.pop("_feature_model")
        return d

    def __setstate__(self, d):
        """Recreate pydantic model type at runtime."""
        self.__dict__ = d
        self._feature_model = self._build_feature_model()

    def map_features(self, features: Features) -> Features:
        """Map features."""
        # make sure the feature to parse is a string
        raise_feature_equals(
            self.config.json_str,
            self.config.json_str.index_features(features),
            Value("string"),
        )
        # return expected scheme
        return {"parsed": self.config.scheme}

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Parse a batch of json strings.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): parsed batch
            index (list[int]): pass through of the input indices
        """
        # combine all json strings to a batch json string
        json_strings = self.config.json_str.index_batch(examples)
        large_json_string = '{"parsed": [%s]}' % ",".join(json_strings)
        # load batch in one validation step
        parsed_batch = self._feature_model.model_validate_json(
            large_json_string
        )
        # return parsed object and source indices
        return parsed_batch.model_dump(), range(len(index))
