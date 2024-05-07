"""Repair Json Parser."""
from typing import Any

import json_repair
from datasets import Features, Value
from datasets.features.features import FeatureType
from pydantic import BaseModel

from hyped.common.feature_checks import raise_feature_equals
from hyped.common.feature_key import FeatureKey
from hyped.common.pydantic import pydantic_model_from_features
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class RepairJsonParserConfig(BaseDataProcessorConfig):
    """Repair Json Parser Config.

    Parse a json string using the `json_repair` library.

    Attributes:
        json (FeatureKey): feature to parse
        scheme (FeatureType): feature scheme of the parsed object
        skip_json_loads(str):
            Flag indicating whether to first try loading the json
            string or directly assume the string to be broken
    """

    json_str: FeatureKey
    scheme: FeatureType

    skip_json_loads: bool = False


class RepairJsonParser(BaseDataProcessor[RepairJsonParserConfig]):
    """Repair Json Parser.

    Parse a json string using the `json_repair` library.
    """

    def __init__(self, config: RepairJsonParserConfig) -> None:
        """Constructor."""
        super(RepairJsonParser, self).__init__(config)
        self._feature_model = self._build_feature_model()

    def _build_feature_model(self) -> BaseModel:
        """Build pydantic model for object validation."""
        return pydantic_model_from_features(
            features={"parsed": self.config.scheme}
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

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example."""
        # parse json string
        obj = json_repair.loads(
            json_str=self.config.json_str.index_example(example),
            skip_json_loads=self.config.skip_json_loads,
        )
        # validate parsed object
        return self._feature_model(parsed=obj).model_dump()
