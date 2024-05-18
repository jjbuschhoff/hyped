from tests.hyped.data.processors.base_test import BaseDataProcessorTest
from hyped.data.processors.ops.noop import NoOp, NoOpConfig
from datasets import Features, Sequence, Value


class TestNoOp_Value(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected output
    expected_output = {"y": list(range(100))}

class TestNoOp_Sequence(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features({"x": Sequence(Value("int32"))})
    input_data = {"x": [[i, i] for i in range(100)]}
    input_index = list(range(100))
    # expected output
    expected_output = {"y": [[i, i] for i in range(100)]}

class TestNoOp_Features(BaseDataProcessorTest):
    # processor
    processor_type = NoOp
    processor_config = NoOpConfig()
    # input
    input_features = Features(
        {
            "x": {
                "a": Value("int32"),
                "b": Value("string")
            }
        }
    )
    input_data = {
        "x": [
            {"a": i, "b": str(i)} for i in range(100)
        ]
    }
    input_index = list(range(100))
    # expected output
    expected_output = {
        "y": [
            {"a": i, "b": str(i)} for i in range(100)
        ]
    }
