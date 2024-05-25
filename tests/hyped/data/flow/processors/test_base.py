from unittest.mock import MagicMock, call

import pytest
from datasets import Features, Value
from typing_extensions import Annotated

from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.data.flow.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.refs.outputs import (
    ConditionalOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.refs.ref import FeatureRef


class MockOutputRefs(OutputRefs):
    out: Annotated[FeatureRef, OutputFeature(Value("int32"))]
    out_cond: Annotated[
        FeatureRef,
        ConditionalOutputFeature(Value("int32"), lambda c, _: c.output_cond),
    ]


class MockInputRefs(InputRefs):
    x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]


class MockProcessorConfig(BaseDataProcessorConfig):
    c: float = 0.0
    output_cond: bool = False


class MockProcessor(
    BaseDataProcessor[MockProcessorConfig, MockInputRefs, MockOutputRefs]
):
    process = MagicMock(return_value={"out": 0})


@pytest.mark.parametrize("output_cond", [False, True])
class TestBaseDataProcessor:
    def test_init(self, output_cond):
        a = MockProcessor()
        b = MockProcessor(MockProcessorConfig())
        # test default values
        assert a.config.c == b.config.c == 0.0

        a = MockProcessor(c=1.0)
        b = MockProcessor(MockProcessorConfig(c=1.0))
        # test setting value
        assert a.config.c == b.config.c == 1.0

    def test_properties(self, output_cond):
        # create mock processor
        proc = MockProcessor.from_config(
            MockProcessorConfig(output_cond=output_cond)
        )
        # check config and input keys property
        assert isinstance(proc.config, MockProcessorConfig)
        assert proc.required_input_keys == {"x"}

    def test_call(self, output_cond):
        # build expected output features
        out_features = Features(
            {"out": Value("int32")}
            if not output_cond
            else {"out": Value("int32"), "out_cond": Value("int32")}
        )

        # mock flow instance
        mock_flow = MagicMock()
        # create processor instance
        proc = MockProcessor.from_config(
            MockProcessorConfig(output_cond=output_cond)
        )
        # create mock inputs
        mock_inputs = MockInputRefs(
            x=FeatureRef(
                key_=tuple(),
                node_id_=-1,
                flow_=mock_flow,
                feature_=Value("int32"),
            )
        )

        # this should add the processor to the mock flow
        out = proc.call(x=mock_inputs.x)
        # check the output features
        assert out.feature_ == out_features

        # make sure the call for adding the processor was made with the correct values
        assert mock_flow.add_processor_node.call_count == 1
        (
            call_proc,
            call_input_refs,
            call_out_features,
        ) = mock_flow.add_processor_node.call_args.args
        assert call_proc == proc
        assert call_input_refs == mock_inputs
        assert call_out_features == out_features

    @pytest.mark.asyncio
    async def test_batch_process(self, output_cond):
        # create mock instance
        proc = MockProcessor.from_config(
            MockProcessorConfig(output_cond=output_cond)
        )
        # create dummy inputs
        rank = 0
        index = list(range(10))
        batch = {"x": index}
        # run batch process
        out_batch = await proc.batch_process(batch, index, rank)
        # check output
        assert out_batch == {"out": [0] * 10}
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank) for i in index]
        proc.process.assert_has_calls(calls, any_order=True)
