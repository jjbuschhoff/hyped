from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from datasets import Features

from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    DataAggregationManager,
)
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.ref import AggregationRef

mock_init_value = MagicMock()
mock_init_state = MagicMock()
mock_ctx = MagicMock()
mock_value = MagicMock()
mock_state = MagicMock()


class MockAggregator(
    BaseDataAggregator[BaseDataAggregatorConfig, InputRefs, MagicMock]
):
    initialize = MagicMock(return_value=(mock_init_value, mock_init_state))
    extract = AsyncMock(return_value=mock_ctx)
    update = AsyncMock(return_value=(mock_value, mock_state))


class TestDataAggregationManager:
    @pytest.fixture
    def aggregators(self):
        return {"agg": MockAggregator()}

    @pytest.fixture
    def in_features(self):
        return {"agg": MagicMock(spec=Features)}

    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    def test_initialization(self, mock_manager, aggregators, in_features):
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=MagicMock())

        manager = DataAggregationManager(aggregators, in_features)

        assert mock_manager.dict.called
        assert mock_manager.Lock.called
        assert isinstance(manager._value_buffer, dict)
        assert isinstance(manager._state_buffer, dict)
        assert isinstance(manager._locks, dict)
        assert isinstance(manager._lookup, defaultdict)
        # check initial buffer values
        assert manager.values_proxy["agg"] == mock_init_value
        assert manager._value_buffer["agg"] == mock_init_value
        assert manager._state_buffer["agg"] == mock_init_state

    @pytest.mark.asyncio
    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    async def test_safe_update(self, mock_manager, aggregators, in_features):
        mock_lock = MagicMock()
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=mock_lock)

        # create data aggregation manager
        manager = DataAggregationManager(aggregators, in_features)

        # mock extracted value
        mock_ctx = MagicMock()
        # call update
        await manager._safe_update("agg", aggregators["agg"], mock_ctx)
        # make sure the lock has been acquired and released
        assert mock_lock.acquire.called
        assert mock_lock.release.called
        # make sure update is called with the expected arguments
        aggregators["agg"].update.assert_called_with(
            mock_init_value, mock_ctx, mock_init_state
        )
        # check updated values
        assert manager._value_buffer["agg"] == mock_value
        assert manager._state_buffer["agg"] == mock_state

        # call update
        await manager._safe_update("agg", aggregators["agg"], mock_ctx)
        # make sure update is called with the expected arguments
        aggregators["agg"].update.assert_called_with(
            mock_value, mock_ctx, mock_state
        )

    @pytest.mark.asyncio
    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    async def test_aggregate(self, mock_manager, aggregators, in_features):
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=MagicMock())

        agg = aggregators["agg"]
        aggregators = {
            "agg1": agg,
            "agg2": agg,
        }
        in_features = {"agg1": in_features["agg"], "agg2": in_features["agg"]}
        # create data aggregation manager
        manager = DataAggregationManager(aggregators, in_features)
        manager._safe_update = MagicMock(side_effect=manager._safe_update)

        mock_input = MagicMock()
        mock_index = MagicMock()
        mock_rank = MagicMock()
        # aggregate
        await manager.aggregate(agg, mock_input, mock_index, mock_rank)
        # check aggregator calls
        agg.extract.assert_called_once_with(mock_input, mock_index, mock_rank)
        assert {
            manager._safe_update.mock_calls[i].args[0]
            for i in range(len(manager._safe_update.mock_calls))
        } == {"agg1", "agg2"}


class TestDataAggregator:
    def test_call(self):
        # mock input value, flow and node id
        mock_x = MagicMock()
        mock_flow = MagicMock()

        # create the mock input ref instance
        mock_in_ref = MagicMock()
        mock_in_ref.flow = mock_flow
        mock_in_ref.flow.add_processor_node = MagicMock(return_value="")

        # create the mock aggregator
        mock_aggregator = MockAggregator()
        mock_aggregator._in_refs_type = MagicMock(return_value=mock_in_ref)

        # call aggregator with mock input
        ref = mock_aggregator.call(x=mock_x)

        # check the input reference is build correctly
        mock_aggregator._in_refs_type.assert_called_with(x=mock_x)
        # make sure the aggregator node is added to the data flow
        mock_in_ref.flow.add_processor_node.assert_called_with(
            mock_aggregator, mock_in_ref, None
        )

        # make sure the aggregation reference is correct
        assert ref == AggregationRef(
            node_id_="", flow_=mock_flow, type_=MagicMock
        )

    def test_properties(self):
        # create the mock aggregator
        mock_aggregator = MockAggregator()
        # check required input keys property
        mock_required_keys = PropertyMock()
        mock_aggregator._in_refs_type = MagicMock()
        mock_aggregator._in_refs_type.required_keys = mock_required_keys
        assert mock_aggregator.required_input_keys == mock_required_keys
