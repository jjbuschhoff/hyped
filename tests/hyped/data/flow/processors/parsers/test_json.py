import json
import pickle

import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.common.feature_checks import check_object_matches_feature
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


class TestJsonParser_ClassLabel(BaseJsonParserTest):
    # processor config
    processor_config = JsonParserConfig(
        scheme=Features({"label": ClassLabel(names=["A", "B", "C"])}),
    )
    # input
    input_features = Features({"json_str": Value("string")})
    input_data = {
        "json_str": [
            json.dumps({"label": "A"}),
            json.dumps({"label": "B"}),
            json.dumps({"label": "C"}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "parsed": [
            {"label": 0},
            {"label": 1},
            {"label": 2},
        ]
    }


class TestJsonParser_CatchNoError(BaseJsonParserTest):
    # processor config
    processor_config = JsonParserConfig(
        scheme=Features({"value": Value("int32")}),
        catch_validation_errors=True,
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
        ],
        "error": [
            None,
            None,
            None,
        ],
    }


class TestJsonParser_CatchWithError(BaseJsonParserTest):
    # processor config
    processor_config = JsonParserConfig(
        scheme=Features({"value": Value("int32")}),
        catch_validation_errors=True,
    )
    # input
    input_features = Features({"json_str": Value("string")})
    input_data = {
        "json_str": [
            json.dumps({"value": "A"}),
            json.dumps({"value": "B"}),
            json.dumps({"value": "C"}),
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "parsed": [
            {"value": None},
            {"value": None},
            {"value": None},
        ],
        "error": [
            "some-error",
            "some-error",
            "some-error",
        ],
    }

    @pytest.mark.asyncio
    async def test_case(self, processor, input_refs, output_refs):
        cls = type(self)
        # check input data
        input_keys = set(cls.input_data.keys())
        assert processor.required_input_keys.issubset(input_keys)
        assert check_object_matches_feature(
            cls.input_data,
            {k: Sequence(v) for k, v in cls.input_features.items()},
        )

        # build default index if not specifically given
        assert len(cls.input_index) == len(next(iter(cls.input_data.values())))

        # apply processor
        output, output_index = await processor.batch_process(
            cls.input_data, cls.input_index, cls.rank
        )

        # check output format
        assert isinstance(output, dict)
        for key, val in output.items():
            assert isinstance(val, list)
            assert len(val) == len(output_index)

        # check output matches features
        assert check_object_matches_feature(
            output, {k: Sequence(v) for k, v in output_refs.feature_.items()}
        )

        # check output matches expectation
        assert output["parsed"] == cls.expected_output_data["parsed"]
        assert [
            output["error"][i] != ""
            for i in range(len(cls.expected_output_data["error"]))
        ]
