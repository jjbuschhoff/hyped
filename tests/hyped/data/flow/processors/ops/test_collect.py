from unittest.mock import MagicMock

import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_checks import check_feature_equals
from hyped.data.flow.processors.ops.collect import (
    CollectFeatures,
    CollectFeaturesConfig,
    CollectFeaturesInputRefs,
    Const,
    FeatureCollection,
)
from hyped.data.flow.refs.ref import FeatureRef
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest

mock_flow = MagicMock()
mock_flow.add_processor = MagicMock()
# create feature refs with different feature types
int_ref = FeatureRef(
    key_="int", feature_=Value("int32"), node_id_=0, flow_=mock_flow
)
str_ref = FeatureRef(
    key_="str", feature_=Value("string"), node_id_=1, flow_=mock_flow
)
float_ref = FeatureRef(
    key_="float", feature_=Value("float"), node_id_=2, flow_=mock_flow
)


class TestFeatureCollection(object):
    def test_basics(self):
        # pack features of same type in sequence
        FeatureCollection(collection=[int_ref, int_ref]).feature
        # can pack arbitrary types in dict
        FeatureCollection(collection={"a": str_ref, "b": int_ref}).feature
        # cannot pack features of different types in sequence
        with pytest.raises(TypeError):
            FeatureCollection(collection=[int_ref, str_ref]).feature

    @pytest.mark.parametrize(
        "collection,feature",
        [
            (FeatureCollection(collection={}), Features()),
            (
                FeatureCollection(collection={"a": str_ref, "b": int_ref}),
                Features({"a": Value("string"), "b": Value("int32")}),
            ),
            (
                FeatureCollection(collection=[int_ref, int_ref]),
                Sequence(Value("int32"), length=2),
            ),
            (
                FeatureCollection(collection=[str_ref, str_ref, "const"]),
                Sequence(Value("string"), length=3),
            ),
            (
                FeatureCollection(
                    collection=[int_ref, int_ref, Const(0, Value("int32"))]
                ),
                Sequence(Value("int32"), length=3),
            ),
        ],
    )
    def test_feature(self, collection, feature):
        assert check_feature_equals(collection.feature, feature)

    @pytest.mark.parametrize(
        "collection,refs",
        [
            (FeatureCollection(collection={}), {}),
            (
                FeatureCollection(collection={"a": str_ref, "b": int_ref}),
                {"a": str_ref, "b": int_ref},
            ),
            (
                FeatureCollection(collection=[int_ref, int_ref]),
                {"[0]": int_ref, "[1]": int_ref},
            ),
            (
                FeatureCollection(collection=[str_ref, str_ref, "const"]),
                {"[0]": str_ref, "[1]": str_ref},
            ),
            (
                FeatureCollection(
                    collection=[int_ref, int_ref, Const(0, Value("int32"))]
                ),
                {"[0]": int_ref, "[1]": int_ref},
            ),
            (
                FeatureCollection(
                    collection={
                        "a": [int_ref, int_ref],
                        "b": {"x": [str_ref, str_ref]},
                    }
                ),
                {
                    "a[0]": int_ref,
                    "a[1]": int_ref,
                    "b.x[0]": str_ref,
                    "b.x[1]": str_ref,
                },
            ),
        ],
    )
    def test_named_refs(self, collection, refs):
        assert collection.named_refs == refs

    @pytest.mark.parametrize(
        "collection,inputs,expected_output",
        [
            (FeatureCollection(collection={}), {}, []),
            (
                FeatureCollection(collection={"a": str_ref, "b": int_ref}),
                {"a": ["A"], "b": [1]},
                [{"a": "A", "b": 1}],
            ),
            (
                FeatureCollection(collection=[int_ref, int_ref]),
                {"[0]": [1], "[1]": [1]},
                [[1, 1]],
            ),
            (
                FeatureCollection(collection=[str_ref, str_ref, "const"]),
                {"[0]": ["var"], "[1]": ["var"]},
                [["var", "var", "const"]],
            ),
            (
                FeatureCollection(
                    collection=[int_ref, int_ref, Const(0, Value("int32"))]
                ),
                {"[0]": [1], "[1]": [1]},
                [[1, 1, 0]],
            ),
        ],
    )
    def test_collect_values(self, collection, inputs, expected_output):
        batch_size = (
            0 if len(inputs) == 0 else len(next(iter(inputs.values())))
        )
        collected = collection._collect_values(inputs, batch_size)
        assert collected == expected_output


class BaseCollectFeaturesTest(BaseDataProcessorTest):
    # processor
    processor_type = CollectFeatures
    processor_config = CollectFeaturesConfig()
    # collection
    collection: FeatureCollection

    @pytest.fixture
    def input_refs(self):
        return CollectFeaturesInputRefs(collection=type(self).collection)

    @pytest.fixture
    def processor(self):
        cls = type(self)
        processor = cls.processor_type.from_config(cls.processor_config)
        processor.call(collection=cls.collection)
        return processor

    def test_input_refs_properties(self, input_refs):
        assert type(input_refs).required_keys == set()
        assert input_refs.flow == mock_flow


class TestCollectFeatures_mapping(BaseCollectFeaturesTest):
    # collection
    collection = FeatureCollection(collection={"a": int_ref, "b": str_ref})
    # inputs
    input_features = {
        "a": int_ref.feature_,
        "b": str_ref.feature_,
    }
    input_data = {
        "a": [i for i in range(100)],
        "b": [str(i) for i in range(100)],
    }
    input_index = list(range(100))
    # expected output
    expected_output_data = {
        "collected": [{"a": i, "b": str(i)} for i in range(100)]
    }

    def test_new_processor_on_reuse(self, processor):
        cls = type(self)

        mock_flow.reset_mock()
        # call same processor multiple times
        processor.call(collection=cls.collection)
        processor.call(collection=cls.collection)
        # this should end up being two different
        # processor instances in the flow
        calls = mock_flow.add_processor_node.mock_calls
        assert len(calls) == 2
        assert calls[0].args[0] != calls[1].args[0]


class TestCollectFeatures_sequence(BaseCollectFeaturesTest):
    # collection
    collection = FeatureCollection(collection=[int_ref, int_ref])
    # inputs
    input_features = {
        "[0]": int_ref.feature_,
        "[1]": int_ref.feature_,
    }
    input_data = {
        "[0]": [i for i in range(100)],
        "[1]": [i for i in range(100)],
    }
    input_index = list(range(100))
    # expected output
    expected_output_data = {"collected": [[i, i] for i in range(100)]}


class TestCollectFeatures_nested(BaseCollectFeaturesTest):
    # collection
    collection = FeatureCollection(
        collection={
            "a": {
                "b": int_ref,
                "c": [
                    {"x": str_ref, "y": str_ref},
                    {"x": str_ref, "y": str_ref},
                ],
            }
        }
    )
    # inputs
    input_features = {
        "a.b": int_ref.feature_,
        "a.c[0].x": str_ref.feature_,
        "a.c[0].y": str_ref.feature_,
        "a.c[1].x": str_ref.feature_,
        "a.c[1].y": str_ref.feature_,
    }
    input_data = {
        "a.b": [i for i in range(100)],
        "a.c[0].x": [str(i) for i in range(100)],
        "a.c[0].y": [str(i) for i in range(100)],
        "a.c[1].x": [str(i) for i in range(100)],
        "a.c[1].y": [str(i) for i in range(100)],
    }
    input_index = list(range(100))
    # expected output
    expected_output_data = {
        "collected": [
            {
                "a": {
                    "b": i,
                    "c": [
                        {"x": str(i), "y": str(i)},
                        {"x": str(i), "y": str(i)},
                    ],
                }
            }
            for i in range(100)
        ]
    }
