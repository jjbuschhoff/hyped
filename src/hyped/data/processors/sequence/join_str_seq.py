"""Join String Sequence Data Processor."""
from typing import Any

from datasets import Features, Value

from hyped.common.feature_checks import raise_feature_is_sequence
from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class JoinStringSequenceConfig(BaseDataProcessorConfig):
    """Join String Sequence Data Processor Config.

    Concatenate a sequence of strings creating a new string
    formed by adding a specified delimiter in between every
    pair of adjacent strings.

    Attributes:
        sequence (FeatureKey):
            feature key to the sequence of strings to join
        delimiter (str):
            delimiter to use for joining the string sequence.
            Defaults to whitespace character.
        output (str):
            output feature name. Defaults to `joined_string`
    """

    sequence: FeatureKey
    delimiter: str = " "
    output: str = "joined_string"


class JoinStringSequence(BaseDataProcessor[JoinStringSequenceConfig]):
    """Join String Sequence Data Processor Config.

    Concatenate a sequence of strings creating a new string
    formed by adding a specified delimiter in between every
    pair of adjacent strings.
    """

    def map_features(self, features: Features) -> Features:
        """Map dataset features.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): output dataset features
        """
        # make sure the feature exists and is a sequence
        # of strings
        raise_feature_is_sequence(
            self.config.sequence,
            self.config.sequence.index_features(features),
            Value("string"),
        )
        # returns a string feature
        return Features({self.config.output: Value("string")})

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example.

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): processed example
        """
        # get the string sequence and join
        return {
            self.config.output: self.config.delimiter.join(
                self.config.sequence.index_example(example)
            )
        }
