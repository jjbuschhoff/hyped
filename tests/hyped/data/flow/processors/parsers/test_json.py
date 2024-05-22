import json
import pickle

from datasets import Features, Sequence, Value

from hyped.data.flow.processors.parsers.json import (
    JsonParser,
    JsonParserConfig,
)
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class BaseJsonParserTest(BaseDataProcessorTest):
    # processor type
    processor_type = JsonParser

    def test_pickle_processor(self, processor):
        # pickle and unpickle processor
        reconstructed = pickle.loads(pickle.dumps(processor))
        # make sure the underlying feature model is the same
        assert (
            processor._feature_model.model_json_schema()
            == reconstructed._feature_model.model_json_schema()
        )


class TestJsonParser_Value(BaseJsonParserTest):
    # processor config
    processor_config = JsonParserConfig(
        scheme=Features({"value": Value("int32")})
    )
    # input
    input_features = Features({"json_str": Value("string")})
    input_data = {
        "json_str": [
            json.dumps({"value": 0}),
            json.dumps({"value": 1}),
            json.dumps({"value": 2}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "parsed": [
            {"value": 0},
            {"value": 1},
            {"value": 2},
        ]
    }


class TestJsonParser_Sequence(BaseJsonParserTest):
    # processor
    processor_type = JsonParser
    processor_config = JsonParserConfig(
        scheme=Sequence(Value("int32"), length=3)
    )
    # input
    input_features = Features({"json_str": Value("string")})
    input_data = {
        "json_str": [
            json.dumps([0, 0, 0]),
            json.dumps([1, 2, 3]),
            json.dumps([3, 2, 1]),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "parsed": [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
        ]
    }


class TestJsonParser_Nested(BaseJsonParserTest):
    # processor
    processor_type = JsonParser
    processor_config = JsonParserConfig(
        scheme=Features(
            {
                "a": Value("int32"),
                "b": Sequence(Value("int32")),
                "c": {
                    "x": Value("int32"),
                    "y": Sequence({"i": Value("int32")}),
                },
            }
        )
    )
    # input
    input_features = Features({"json_str": Value("string")})
    input_data = {
        "json_str": [
            json.dumps(
                {
                    "a": 0,
                    "b": [0, 0],
                    "c": {
                        "x": 0,
                        "y": [
                            {"i": 0},
                            {"i": 0},
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "a": 1,
                    "b": [2, 2],
                    "c": {"x": 3, "y": [{"i": 4}, {"i": 5}, {"i": 6}]},
                }
            ),
        ]
    }
    input_index = [0, 1]
    # expected output
    expected_output_data = {
        "parsed": [
            {
                "a": 0,
                "b": [0, 0],
                "c": {
                    "x": 0,
                    "y": [
                        {"i": 0},
                        {"i": 0},
                    ],
                },
            },
            {
                "a": 1,
                "b": [2, 2],
                "c": {"x": 3, "y": [{"i": 4}, {"i": 5}, {"i": 6}]},
            },
        ]
    }
