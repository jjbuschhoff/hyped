from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from datasets import Features, Value

from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    DataAggregationManager,
)
from hyped.data.flow.core.nodes.base import IOContext
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef
from tests.hyped.data.flow.core.mock import MockAggregator

mock_init_value = MagicMock()
mock_init_state = MagicMock()
mock_ctx = MagicMock()
mock_value = MagicMock()
mock_state = MagicMock()


class MockAggregator(MockAggregator):
    initialize = MagicMock(return_value=(mock_init_value, mock_init_state))
    extract = AsyncMock(return_value=mock_ctx)
    update = AsyncMock(return_value=(mock_value, mock_state))


class TestDataAggregationManager:
    @pytest.fixture
    def node_ids(self):
        return ["node_id"]

    @pytest.fixture
    def aggregators(self):
        return [MockAggregator]

    @pytest.fixture
    def io_contexts(self, node_ids):
        return [
            IOContext(
                node_id=node_id,
                inputs=Features(),
                outputs=Features({"out": Value("int32")}),
            )
            for node_id in node_ids
        ]

    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    def test_initialization(
        self, mock_manager, node_ids, aggregators, io_contexts
    ):
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=MagicMock())

        manager = DataAggregationManager(aggregators, io_contexts)

        assert mock_manager.dict.called
        assert mock_manager.Lock.called
        assert isinstance(manager._value_buffer, dict)
        assert isinstance(manager._state_buffer, dict)
        assert isinstance(manager._locks, dict)
        # check initial buffer values
        for node_id in node_ids:
            assert manager.values_proxy[node_id] == mock_init_value
            assert manager._value_buffer[node_id] == mock_init_value
            assert manager._state_buffer[node_id] == mock_init_state

    @pytest.mark.asyncio
    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    async def test_safe_update(self, mock_manager, aggregators, io_contexts):
        mock_lock = MagicMock()
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=mock_lock)

        # create data aggregation manager
        manager = DataAggregationManager(aggregators, io_contexts)

        for io_ctx, aggregator in zip(io_contexts, aggregators):
            # mock extracted value
            mock_ctx = MagicMock()
            # call update
            await manager._safe_update(io_ctx, aggregator, mock_ctx)
            # make sure the lock has been acquired and released
            assert mock_lock.acquire.called
            assert mock_lock.release.called
            # make sure update is called with the expected arguments
            aggregator.update.assert_called_with(
                mock_init_value, mock_ctx, mock_init_state, io_ctx
            )
            # check updated values
            assert manager._value_buffer[io_ctx.node_id] == mock_value
            assert manager._state_buffer[io_ctx.node_id] == mock_state

            # call update
            await manager._safe_update(io_ctx, aggregator, mock_ctx)
            # make sure update is called with the expected arguments
            aggregator.update.assert_called_with(
                mock_value, mock_ctx, mock_state, io_ctx
            )

    @pytest.mark.asyncio
    @patch("hyped.data.flow.core.nodes.aggregator._manager")
    async def test_aggregate(self, mock_manager, aggregators, io_contexts):
        # Mock the _manager object
        mock_manager.dict = MagicMock(side_effect=lambda x: dict(x))
        mock_manager.Lock = MagicMock(return_value=MagicMock())
        # create data aggregation manager
        manager = DataAggregationManager(aggregators, io_contexts)
        manager._safe_update = MagicMock(side_effect=manager._safe_update)

        mock_input = MagicMock()
        mock_index = MagicMock()
        mock_rank = MagicMock()

        for io_ctx, aggregator in zip(io_contexts, aggregators):
            # aggregate
            await manager.aggregate(
                aggregator, mock_input, mock_index, mock_rank, io_ctx
            )
            # check aggregator calls
            aggregator.extract.assert_called_once_with(
                mock_input, mock_index, mock_rank, io_ctx
            )
            manager._safe_update.assert_called_once_with(
                io_ctx, aggregator, mock_ctx
            )


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
        mock_aggregator._in_refs_type = MagicMock()
        mock_aggregator._in_refs_validator.validate = MagicMock(
            return_value=mock_in_ref
        )

        # call aggregator with mock input
        ref = mock_aggregator.call(x=mock_x)

        # check the input reference is build correctly
        mock_aggregator._in_refs_type.assert_called_with(x=mock_x)
        # make sure the aggregator node is added to the data flow
        mock_in_ref.flow.add_processor_node.assert_called_with(
            mock_aggregator, mock_in_ref, Features({"y": Value("int64")})
        )

        # make sure the aggregation reference is correct
        assert (
            ref.ptr
            == FeatureRef(
                node_id_="",
                key_=tuple(),
                flow_=mock_flow,
                feature_=Features({"y": Value("int64")}),
            ).ptr
        )

    def test_properties(self):
        # create the mock aggregator
        mock_aggregator = MockAggregator()
        # check required input keys property
        mock_required_keys = PropertyMock()
        mock_aggregator._in_refs_validator = MagicMock()
        mock_aggregator._in_refs_validator.required_keys = mock_required_keys
        assert mock_aggregator.required_input_keys == mock_required_keys
