from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.noop import NoOp, NoOpConfig
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestNoOp_Value(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected output
    expected_output_data = {"y": list(range(100))}


class TestNoOp_Sequence(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features({"x": Sequence(Value("int32"))})
    input_data = {"x": [[i, i] for i in range(100)]}
    input_index = list(range(100))
    # expected output
    expected_output_data = {"y": [[i, i] for i in range(100)]}


class TestNoOp_Features(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features(
        {"x": {"a": Value("int32"), "b": Value("string")}}
    )
    input_data = {"x": [{"a": i, "b": str(i)} for i in range(100)]}
    input_index = list(range(100))
    # expected output
    expected_output_data = {"y": [{"a": i, "b": str(i)} for i in range(100)]}
