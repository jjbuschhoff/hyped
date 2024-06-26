from datasets import Features, Value

from hyped.data.flow.processors.ops import unary
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestNeg(BaseDataProcessorTest):
    processor_type = unary.Neg
    processor_config = unary.NegConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-1, 2, 0]}


class TestInvert(BaseDataProcessorTest):
    processor_type = unary.Invert
    processor_config = unary.InvertConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-2, 1, -1]}


class TestAbs(BaseDataProcessorTest):
    processor_type = unary.Abs
    processor_config = unary.AbsConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 2, 0]}
