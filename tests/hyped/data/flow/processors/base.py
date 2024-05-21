from unittest.mock import MagicMock

import pytest
from datasets import Features, Sequence

from hyped.common.feature_checks import check_object_matches_feature
from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.refs.inputs import InputRefs
from hyped.data.flow.refs.outputs import OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


class BaseDataProcessorTest:
    # processor to test
    processor_type: type[BaseDataProcessor]
    processor_config: BaseDataProcessorConfig
    # input values
    input_features: Features
    input_data: Batch
    input_index: None | list[int] = None
    # expected output
    expected_output: None | Batch = None
    expected_output_index: None | list[int] = None
    # others
    rank: int = 0

    @pytest.fixture
    def processor(self):
        cls = type(self)
        return cls.processor_type.from_config(cls.processor_config)

    @pytest.fixture
    def input_refs(self, processor) -> InputRefs:
        cls = type(self)

        n, f = -1, MagicMock()
        input_refs = {
            k: FeatureRef(key_=k, feature_=v, node_id_=n, flow_=f)
            for k, v in cls.input_features.items()
        }
        return processor._in_refs_type(**input_refs)

    @pytest.fixture
    def output_refs(self, processor, input_refs) -> OutputRefs:
        return processor._out_refs_type(processor.config, input_refs, -1)

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
        input_index = (
            cls.input_index
            if cls.input_index is not None
            else list(range(len(next(iter(cls.input_data.values())))))
        )
        assert len(input_index) == len(next(iter(cls.input_data.values())))

        # apply processor
        output, output_index = await processor.batch_process(
            cls.input_data, input_index, cls.rank
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
        if cls.expected_output is not None:
            assert output == cls.expected_output
        if cls.expected_output_index is not None:
            assert output_index == cls.expected_output_index
