import json

import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.processors.parsers.json import JsonParser, JsonParserConfig
from tests.hyped.data.processors.base import BaseTestDataProcessor


class TestJsonParser(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            (
                {"val": 0},
                json.dumps({"val": 0}),
                Features({"val": Value("int32")}),
            ),
            (
                {"val": 0},
                json.dumps({"val": 0}),
                Features({"val": ClassLabel(names=["A", "B"])}),
            ),
            (
                {"val": 0},
                json.dumps({"val": "A"}),
                Features({"val": ClassLabel(names=["A", "B"])}),
            ),
            (
                {"val": 1},
                json.dumps({"val": "B"}),
                Features({"val": ClassLabel(names=["A", "B"])}),
            ),
            (
                {"z": [0, 1, 2]},
                json.dumps({"z": [0, 1, 2]}),
                Features({"z": Sequence(Value("int32"), length=3)}),
            ),
            ([1, 2, 3], json.dumps([1, 2, 3]), Sequence(Value("int32"))),
            (
                [{"A": 1}, {"A": 0}],
                json.dumps([{"A": 1}, {"A": 0}]),
                Sequence(Features({"A": Value("int32")})),
            ),
        ]
    )
    def obj_string_scheme(self, request):
        return request.param

    @pytest.fixture
    def obj(self, obj_string_scheme):
        return obj_string_scheme[0]

    @pytest.fixture
    def json_string(self, obj_string_scheme):
        return obj_string_scheme[1]

    @pytest.fixture
    def scheme(self, obj_string_scheme):
        return obj_string_scheme[2]

    @pytest.fixture
    def in_features(self):
        return Features({"json": Value("string")})

    @pytest.fixture
    def processor(self, scheme):
        return JsonParser(JsonParserConfig(json_str="json", scheme=scheme))

    @pytest.fixture
    def in_batch(self, json_string):
        return {"json": [json_string]}

    @pytest.fixture
    def expected_out_features(self, scheme):
        return Features({"parsed": scheme})

    @pytest.fixture
    def expected_out_batch(self, obj):
        return {"parsed": [obj]}
