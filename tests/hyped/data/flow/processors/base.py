from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from datasets import Features, Sequence

from hyped.common.feature_checks import (
    check_feature_equals,
    check_object_matches_feature,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.outputs import OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BaseDataProcessorTest:
    # processor to test
    processor_type: type[BaseDataProcessor]
    processor_config: BaseDataProcessorConfig
    # input values
    input_features: Features
    input_data: None | Batch = None
    input_index: None | list[int] = None
    # expected output
    expected_output_features: None | Features = None
    expected_output_data: None | Batch = None
    # expected errors
    expected_execution_error: None | type[Exception] = None
    expected_input_verification_error: None | type[Exception] = None
    # others
    rank: int = 0

    @pytest.fixture
    def exec_error_handler(self):
        cls = type(self)
        return (
            pytest.raises(cls.expected_execution_error)
            if cls.expected_execution_error is not None
            else nullcontext()
        )

    @pytest.fixture
    def input_verification_error_handler(self):
        cls = type(self)
        return (
            pytest.raises(cls.expected_input_verification_error)
            if cls.expected_input_verification_error is not None
            else nullcontext()
        )

    @pytest.fixture
    def processor(self):
        cls = type(self)
        return cls.processor_type.from_config(cls.processor_config)

    @pytest.fixture
    def input_refs(
        self, processor, input_verification_error_handler
    ) -> None | InputRefs:
        cls = type(self)

        n, f = "in", MagicMock()
        input_refs = {
            k: FeatureRef(key_=k, feature_=v, node_id_=n, flow_=f)
            for k, v in cls.input_features.items()
        }

        with input_verification_error_handler:
            return processor._in_refs_type(**input_refs)

    @pytest.fixture
    def output_refs(self, processor, input_refs) -> None | OutputRefs:
        # error catched in input verification
        if input_refs is None:
            return None
        # build output feature references
        return processor._out_refs_type(
            input_refs.flow,
            "out",
            processor._out_refs_type.build_features(
                processor.config, input_refs
            ),
        )

    @pytest.mark.asyncio
    async def test_case(
        self, processor, input_refs, output_refs, exec_error_handler
    ):
        cls = type(self)

        if input_refs is None:
            # catched input verification error
            return
        else:
            # make sure no input verification error was specified and
            assert cls.expected_input_verification_error is None

        assert output_refs is not None

        # check output features
        if cls.expected_output_features is not None:
            assert check_feature_equals(
                output_refs.feature_, cls.expected_output_features
            )

        # only test the feature management, don't run the processor
        if cls.input_data is None:
            return

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
        if len(cls.input_data) > 0:
            assert len(input_index) == len(next(iter(cls.input_data.values())))

        # build the io context
        io = IOContext(
            node_id=-1,
            inputs=cls.input_features,
            outputs=processor._out_refs_type.build_features(
                processor.config, input_refs
            ),
        )

        with exec_error_handler:
            # apply processor
            output = await processor.batch_process(
                cls.input_data, input_index, cls.rank, io
            )

        # expected error was catched, cutoff test here
        if cls.expected_execution_error is not None:
            return

        # check output format
        assert isinstance(output, dict)
        for key, val in output.items():
            assert isinstance(val, list)
            assert len(val) == len(input_index)

        # check output matches features
        assert check_object_matches_feature(
            output, {k: Sequence(v) for k, v in output_refs.feature_.items()}
        ), (output, {k: Sequence(v) for k, v in output_refs.feature_.items()})

        # check output matches expectation
        if cls.expected_output_data is not None:
            assert output == cls.expected_output_data
