from datasets import Features, Value

from hyped.data.flow.processors.ops import binary
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestComparator(BaseDataProcessorTest):
    processor_type = binary.Comparator
    processor_config = binary.ComparatorConfig(op=lambda a, b: False)

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, False]}


class TestEquals(BaseDataProcessorTest):
    processor_type = binary.Equals
    processor_config = binary.EqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, False]}


class TestNotEquals(BaseDataProcessorTest):
    processor_type = binary.NotEquals
    processor_config = binary.NotEqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLessThan(BaseDataProcessorTest):
    processor_type = binary.LessThan
    processor_config = binary.LessThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestLessThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.LessThanOrEqual
    processor_config = binary.LessThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, True, False]}


class TestGreaterThan(BaseDataProcessorTest):
    processor_type = binary.GreaterThan
    processor_config = binary.GreaterThanConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestGreaterThanOrEqual(BaseDataProcessorTest):
    processor_type = binary.GreaterThanOrEqual
    processor_config = binary.GreaterThanOrEqualConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, True]}


class TestLogicalOp(BaseDataProcessorTest):
    processor_type = binary.LogicalOp
    processor_config = binary.LogicalOpConfig(op=lambda a, b: False)

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, False]}


class TestLogicalAnd(BaseDataProcessorTest):
    processor_type = binary.LogicalAnd
    processor_config = binary.LogicalAndConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestLogicalOr(BaseDataProcessorTest):
    processor_type = binary.LogicalOr
    processor_config = binary.LogicalOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLogicalXOr(BaseDataProcessorTest):
    processor_type = binary.LogicalXOr
    processor_config = binary.LogicalXOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestClosedOp_Int32_Int32(BaseDataProcessorTest):
    processor_type = binary.ClosedOp
    processor_config = binary.ClosedOpConfig(op=lambda a, b: 0)

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 0], "b": [0, 0, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestClosedOp_Int16_Int32(BaseDataProcessorTest):
    processor_type = binary.ClosedOp
    processor_config = binary.ClosedOpConfig(op=lambda a, b: 0)

    input_features = Features({"a": Value("int16"), "b": Value("int32")})
    input_data = {"a": [0, 0, 0], "b": [0, 0, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestClosedOp_Float32_Float32(BaseDataProcessorTest):
    processor_type = binary.ClosedOp
    processor_config = binary.ClosedOpConfig(op=lambda a, b: 0.0)

    input_features = Features({"a": Value("float32"), "b": Value("float32")})
    input_data = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [0.0, 0.0, 0.0]}


class TestClosedOp_Int32_Float32(BaseDataProcessorTest):
    processor_type = binary.ClosedOp
    processor_config = binary.ClosedOpConfig(op=lambda a, b: 0.0)

    input_features = Features({"a": Value("int32"), "b": Value("float32")})
    input_data = {"a": [0, 0, 0], "b": [0.0, 0.0, 0.0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [0.0, 0.0, 0.0]}


class TestAdd(BaseDataProcessorTest):
    processor_type = binary.Add
    processor_config = binary.AddConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 4, 6]}


class TestSub(BaseDataProcessorTest):
    processor_type = binary.Sub
    processor_config = binary.SubConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 0, 0]}


class TestMul(BaseDataProcessorTest):
    processor_type = binary.Mul
    processor_config = binary.MulConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 9]}


class TestPow(BaseDataProcessorTest):
    processor_type = binary.Pow
    processor_config = binary.PowConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [1, 2, 3], "b": [1, 2, 3]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 4, 27]}


class TestMod(BaseDataProcessorTest):
    processor_type = binary.Mod
    processor_config = binary.ModConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 5, 6], "b": [2, 2, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [0, 1, 2]}


class TestFloorDiv(BaseDataProcessorTest):
    processor_type = binary.FloorDiv
    processor_config = binary.FloorDivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [2, 1, 2]}


class TestDiv(BaseDataProcessorTest):
    processor_type = binary.Div
    processor_config = binary.DivConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [4, 6, 9], "b": [2, 4, 4]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [2.0, 1.5, 2.25]}
