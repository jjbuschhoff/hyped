from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence

from hyped.common.feature_checks import check_object_matches_feature
from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    Batch,
    DataAggregationManager,
)
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.ref import AggregationRef, FeatureRef

UNSET = object()


class BaseDataAggregatorTest:
    # aggregator to test
    aggregator_type: type[BaseDataAggregator]
    aggregator_config: BaseDataAggregatorConfig
    # input values
    input_features: Features
    input_data: Batch
    input_index: None | list[int] = None
    # expected initial state
    expected_initial_value: None | Any = UNSET
    expected_initial_state: None | Any = UNSET
    # expected output
    expected_output_value: None | Any = UNSET
    expected_output_state: None | Any = UNSET
    # others
    rank: int = 0

    aggregation_name: str = "name"

    @pytest.fixture
    def aggregator(self):
        cls = type(self)
        return cls.aggregator_type.from_config(cls.aggregator_config)

    @pytest.fixture
    def input_refs(self, aggregator) -> InputRefs:
        cls = type(self)
        n, f = "in", MagicMock()
        input_refs = {
            k: FeatureRef(key_=k, feature_=v, node_id_=n, flow_=f)
            for k, v in cls.input_features.items()
        }
        return aggregator._in_refs_type(**input_refs)

    @pytest.fixture
    def aggregation_ref(self, aggregator, input_refs) -> AggregationRef:
        return AggregationRef(
            node_id_="out", flow_=input_refs.flow, type_=aggregator._value_type
        )

    @pytest.fixture
    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    def manager(self, mock_manager, aggregator) -> DataAggregationManager:
        cls = type(self)
        # Mock the multiprocessing manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=MagicMock())
        # create aggregation manager
        cls = type(self)
        return DataAggregationManager(
            aggregators={cls.aggregation_name: aggregator},
            in_features={cls.aggregation_name: cls.input_features},
        )

    @pytest.mark.asyncio
    async def test_case(
        self, manager, aggregator, input_refs, aggregation_ref
    ):
        cls = type(self)
        # check input data
        input_keys = set(cls.input_data.keys())
        assert aggregator.required_input_keys.issubset(input_keys)
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

        # check initial aggregation state
        if cls.expected_initial_value != UNSET:
            assert (
                manager._value_buffer[cls.aggregation_name]
                == cls.expected_initial_value
            ), (
                f"Expected {manager._value_buffer[cls.aggregation_name]}, "
                f"got {cls.expected_initial_value}"
            )
        if cls.expected_initial_state != UNSET:
            assert (
                manager._state_buffer[cls.aggregation_name]
                == cls.expected_initial_state
            ), (
                f"Expected {manager._state_buffer[cls.aggregation_name]}, "
                f"got {cls.expected_initial_state}"
            )

        # run aggregation
        await manager.aggregate(
            aggregator, cls.input_data, input_index, cls.rank
        )

        # check aggregation state after execution
        if cls.expected_output_value != UNSET:
            assert (
                manager._value_buffer[cls.aggregation_name]
                == cls.expected_output_value
            ), (
                f"Expected {manager._value_buffer[cls.aggregation_name]}, "
                f"got {cls.expected_output_value}"
            )
        if cls.expected_output_state != UNSET:
            assert (
                manager._state_buffer[cls.aggregation_name]
                == cls.expected_output_state
            ), (
                f"Expected {manager._state_buffer[cls.aggregation_name]}, "
                f"got {cls.expected_output_ctx}"
            )
