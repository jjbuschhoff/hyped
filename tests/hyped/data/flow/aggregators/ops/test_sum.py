from datasets import Features, Value

from hyped.data.flow.aggregators.ops.sum import (
    SumAggregator,
    SumAggregatorConfig,
)
from tests.hyped.data.flow.aggregators.base import BaseDataAggregatorTest


class TestSum(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = SumAggregator
    aggregator_config = SumAggregatorConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected initial value
    expected_initial_value = {"value": 0}
    expected_initial_state = None
    # expected output
    expected_output_value = {"value": sum(range(100))}
    expected_output_state = None


class TestSumWithOffset(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = SumAggregator
    aggregator_config = SumAggregatorConfig(start=-10)
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected initial value
    expected_initial_value = {"value": -10}
    expected_initial_state = None
    # expected output
    expected_output_value = {"value": sum(range(100)) - 10}
    expected_output_state = None
