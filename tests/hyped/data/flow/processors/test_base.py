from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from datasets import Value
from typing_extensions import Annotated

from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.data.flow.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


class MockOutputRefs(OutputRefs):
    out: Annotated[FeatureRef, OutputFeature(Value("int32"))]


class MockInputRefs(InputRefs):
    x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]


class MockProcessorConfig(BaseDataProcessorConfig):
    c: float = 0.0


class MockProcessor(
    BaseDataProcessor[MockProcessorConfig, MockInputRefs, MockOutputRefs]
):
    process = MagicMock(return_value={"out": 0})


class TestBaseDataProcessor:
    def test_init(self):
        a = MockProcessor()
        b = MockProcessor(MockProcessorConfig())
        # test default values
        assert a.config.c == b.config.c == 0.0

        a = MockProcessor(c=1.0)
        b = MockProcessor(MockProcessorConfig(c=1.0))
        # test setting value
        assert a.config.c == b.config.c == 1.0

    def test_properties(self):
        # create mock processor
        proc = MockProcessor.from_config(MockProcessorConfig())
        # check config and input keys property
        assert isinstance(proc.config, MockProcessorConfig)
        assert proc.required_input_keys == {"x"}

    def test_call(self):
        # create processor instance
        proc = MockProcessor.from_config(MockProcessorConfig())
        # create mock inputs
        mock_inputs = InputRefs()

        # patch flow property of input refs
        with patch(
            "hyped.data.flow.refs.inputs.InputRefs.flow",
            callable=PropertyMock,
        ):
            # this should add the processor to the mock flow
            out = proc.call(inputs=mock_inputs)
            # check output and make sure processor was added
            mock_inputs.flow.add_processor_node.assert_called_with(
                proc, mock_inputs
            )

        # test error on to many inputs
        with pytest.raises(TypeError):
            proc.call(inputs=mock_inputs, x=None)

    @pytest.mark.asyncio
    async def test_batch_process(self):
        # create mock instance
        proc = MockProcessor.from_config(MockProcessorConfig())
        # create dummy inputs
        rank = 0
        index = list(range(10))
        batch = {"x": index}
        # run batch process
        out_batch, out_index = await proc.batch_process(batch, index, rank)
        # check output
        assert out_index == index
        assert out_batch == {"out": [0] * 10}
        # make sure the process function was called for each input sample
        calls = [call({"x": i}, i, rank) for i in index]
        proc.process.assert_has_calls(calls, any_order=True)
