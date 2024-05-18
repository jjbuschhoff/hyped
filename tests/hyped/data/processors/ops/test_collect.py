from unittest.mock import MagicMock

import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_checks import check_feature_equals
from hyped.data.flow.ref import FeatureRef
from hyped.data.processors.ops.collect import (
    CollectFeatures,
    CollectFeaturesConfig,
    CollectFeaturesInputRefs,
    FeatureCollection,
)
from tests.hyped.data.processors.base_test import BaseDataProcessorTest

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
        ],
    )
    def test_feature(self, collection, feature):
        assert check_feature_equals(collection.feature, feature)

    @pytest.mark.parametrize(
        "collection,refs",
        [
            (FeatureCollection(collection={}), set()),
            (
                FeatureCollection(collection={"a": str_ref, "b": int_ref}),
                (str_ref, int_ref),
            ),
            (
                FeatureCollection(collection=[int_ref, int_ref]),
                (int_ref, int_ref),
            ),
        ],
    )
    def test_refs(self, collection, refs):
        assert collection.refs == set(refs)


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
        assert type(input_refs).keys == set()
        assert all(k == str(hash(r)) for k, r in input_refs.named_refs.items())
        assert input_refs.flow == mock_flow


class TestCollectFeatures_mapping(BaseCollectFeaturesTest):
    # collection
    collection = FeatureCollection(collection={"a": int_ref, "b": str_ref})
    # inputs
    input_features = {
        str(hash(int_ref)): int_ref.feature_,
        str(hash(str_ref)): str_ref.feature_,
    }
    input_data = {
        str(hash(int_ref)): [i for i in range(100)],
        str(hash(str_ref)): [str(i) for i in range(100)],
    }
    input_index = list(range(100))
    # expected output
    expected_output = {
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
        calls = mock_flow.add_processor.mock_calls
        assert len(calls) == 2
        assert calls[0].args[0] != calls[1].args[0]


class TestCollectFeatures_sequence(BaseCollectFeaturesTest):
    # collection
    collection = FeatureCollection(collection=[int_ref, int_ref])
    # inputs
    input_features = {
        str(hash(int_ref)): int_ref.feature_,
    }
    input_data = {
        str(hash(int_ref)): [i for i in range(100)],
    }
    input_index = list(range(100))
    # expected output
    expected_output = {"collected": [[i, i] for i in range(100)]}
