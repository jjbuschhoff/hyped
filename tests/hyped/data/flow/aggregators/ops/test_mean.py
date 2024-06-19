from datasets import Features, Value

from hyped.data.flow.aggregators.ops.mean import (
    MeanAggregator,
    MeanAggregatorConfig,
)
from tests.hyped.data.flow.aggregators.base import BaseDataAggregatorTest


class TestMean(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MeanAggregator
    aggregator_config = MeanAggregatorConfig()
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected initial value
    expected_initial_value = {"value": 0}
    expected_initial_state = 0
    # expected output
    expected_output_value = {"value": sum(range(100)) / 100.0}
    expected_output_state = 100


class TestMeanWithOffset(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MeanAggregator
    aggregator_config = MeanAggregatorConfig(start=-10, start_count=5)
    # input
    input_features = Features({"x": Value("int32")})
    input_data = {"x": list(range(100))}
    input_index = list(range(100))
    # expected initial value
    expected_initial_value = {"value": -10}
    expected_initial_state = 5
    # expected output
    expected_output_value = {"value": (sum(range(100)) - 10 * 5) / (100 + 5)}
    expected_output_state = 105
