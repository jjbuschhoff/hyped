"""Provides base classes for data aggregators in a data flow graph.

This module defines the base class for data aggregators, which manage the aggregation
of data within a data flow graph. It includes generic classes for defining data aggregators
with configurable input types and aggregation logic.

Classes:
    - :class:`DataAggregationManager`: Manager for handling data aggregation operations.
    - :class:`BaseDataAggregatorConfig`: Base class for data aggregator configurations.
    - :class:`BaseDataAggregator`: Base class for data aggregators.

Usage Example:
    Define a custom data aggregator by subclassing :code:`BaseDataAggregator`:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.flow.core.nodes.aggregator import (
            BaseDataAggregator, BaseDataAggregatorConfig
        )
        from hyped.data.flow.core.refs.inputs import InputRefs
        from datasets.features.features import Features
        from typing import Annotated

        class CustomInputRefs(InputRefs):
            x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]

        class CustomConfig(BaseDataAggregatorConfig):
            threshold: int

        class CustomAggregator(BaseDataAggregator[CustomConfig, CustomInputRefs, int]):
            def initialize(self, features: Features) -> tuple[int, Any]:
                # Define initialization logic here
                return 0, {}

            async def extract(self, inputs: Batch, index: list[int], rank: int) -> Any:
                # Define extraction logic here
                return sum(v for v in inputs["x"] if v >= self.config.treshold)

            async def update(self, val: int, ctx: Any, state: Any) -> tuple[int, Any]:
                # Define update logic here
                return val + ctx, state

    In this example, :class:`CustomAggregator` extends :class:`BaseDataAggregator` and
    implements the :class:`BaseDataAggregator.initialize`, :class:`BaseDataAggregator.extract`,
    and :class:`BaseDataAggregator.update` methods to define custom aggregation logic.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
from abc import ABC, abstractmethod
from multiprocessing.managers import SyncManager
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar

from hyped.common.lazy import LazyStaticInstance

from ..refs.inputs import InputRefs
from ..refs.outputs import OutputRefs
from .base import BaseNode, BaseNodeConfig, IOContext

Batch: TypeAlias = dict[str, list[Any]]


def _sync_manager_factory() -> SyncManager:
    """Factory function for creating a SyncManager instance.

    Returns:
        SyncManager: An instance of SyncManager.
    """
    manager = SyncManager(ctx=mp.context.DefaultContext)
    manager.start()
    return manager


# create global sync manager
_manager = LazyStaticInstance[SyncManager](_sync_manager_factory)


class DataAggregationManager(object):
    """Manager for handling data aggregation operations.

    This class manages data aggregators, their thread-safe buffers, and synchronization
    locks to ensure safe concurrent updates during the data processing.

    Attributes:
        _value_buffer (dict): Thread-safe buffer for aggregation values.
        _state_buffer (dict): Thread-safe buffer for aggregation states.
        _locks (dict): Locks for synchronizing access to aggregators.
    """

    def __init__(
        self,
        aggregators: list[BaseDataAggregator],
        io_contexts: list[IOContext],
    ) -> None:
        """Initialize the DataAggregationManager.

        Args:
            aggregators (dict[str, BaseDataAggregator]): A list of aggregators.
            io_contexts (list[IOContext]): A list of contexts correspoding to the aggregators.
        """
        global _manager
        # create buffers
        value_buffer = {}
        state_buffer = {}
        # fill buffers with initial values from aggregators
        for agg, io in zip(aggregators, io_contexts):
            (
                value_buffer[io.node_id],
                state_buffer[io.node_id],
            ) = agg.initialize(io)
        # create thread-safe buffers
        self._value_buffer = _manager.dict(value_buffer)
        self._state_buffer = _manager.dict(state_buffer)
        # create a lock for each entry to synchronize access
        self._locks = {ctx.node_id: _manager.Lock() for ctx in io_contexts}
        self._locks = _manager.dict(self._locks)

    @property
    def values_proxy(self) -> MappingProxyType[str, Any]:
        """Get a read-only view of the aggregation values.

        Returns:
            MappingProxyType[str, Any]: A read-only view of the aggregation values.
        """
        return MappingProxyType(self._value_buffer)

    async def _safe_update(
        self, io: IOContext, aggregator: BaseDataAggregator, ctx: Any
    ) -> None:
        """Safely update an aggregation value.

        Args:
            io (IOContext): The io context object indicating the specific node to execute.
            aggregator (BaseDataAggregator): The aggregator object.
            ctx (Any): The context values extracted from the input batch.
        """
        assert io.node_id in self._value_buffer
        # get the running event loop
        loop = asyncio.get_running_loop()
        # acquire the lock for the current aggregator
        await loop.run_in_executor(None, self._locks[io.node_id].acquire)
        # get current value and context
        value = self._value_buffer[io.node_id]
        state = self._state_buffer[io.node_id]
        # compute udpated value and context
        value, state = await aggregator.update(value, ctx, state, io)
        # write new values to buffers
        self._value_buffer[io.node_id] = value
        self._state_buffer[io.node_id] = state
        # release lock
        self._locks[io.node_id].release()

    async def aggregate(
        self,
        aggregator: BaseDataAggregator,
        inputs: Batch,
        index: list[int],
        rank: int,
        io: IOContext,
    ) -> None:
        """Perform aggregation for a batch of inputs.

        Args:
            aggregator (BaseDataAggregator): The aggregator object.
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the aggregator execution.
        """
        # extract values required for update from current input batch
        # and update the aggregated value and state
        ctx = await aggregator.extract(inputs, index, rank, io)
        await self._safe_update(io, aggregator, ctx)


class BaseDataAggregatorConfig(BaseNodeConfig):
    """Base configuration class for data aggregators.

    This class serves as the base configuration class for data aggregators.
    It inherits from :class:`BaseConfig`, providing basic configuration
    functionality for data aggregation tasks.
    """


C = TypeVar("C", bound=BaseDataAggregatorConfig)
I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataAggregator(BaseNode[C, I, O], ABC):
    """Base class for data aggregators.

    This class serves as the base for all data aggregators, defining the necessary
    interfaces and methods for implementing custom aggregators.

    Attributes:
        _in_refs_type (Type[I]): The type of input references expected by the aggregator.
        _value_type (Type[T]): The type of the aggregation value.
    """

    @property
    def required_input_keys(self) -> set[str]:
        """Retrieves the set of input keys required by the processor.

        Returns:
            set[str]: The set of input keys.
        """
        return self._in_refs_validator.required_keys

    def call(self, **kwargs) -> O:
        """Call the data aggregator with the provided inputs.

        This method builds inputs from keyword arguments, adds the aggregator
        to the data flow, and returns a reference to the aggregation outputs.

        Args:
            **kwargs: Keyword arguments specifying feature references to be
                passed as inputs to the aggregator.

        Returns:
            O: The output references produced by the aggregator.
        """
        # build inputs from keyword arguments
        inputs = self._in_refs_type(**kwargs)
        inputs = self._in_refs_validator.validate(inputs)
        # compute output features and add the processor to the data flow
        out_features = self._out_refs_type.build_features(self.config, inputs)
        node_id = inputs.flow.add_processor_node(self, inputs, out_features)
        # return the output feature refs
        return self._out_refs_type(inputs.flow, node_id, out_features)

    @abstractmethod
    def initialize(self, io: IOContext) -> tuple[O, Any]:
        """Initialize the aggregator with the given features.

        Args:
            io (IOContext): The execution context object wrapping the
                input and output features.

        Returns:
            tuple[O, Any]: The initial value and state for the aggregator.
        """
        ...

    @abstractmethod
    async def extract(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Any:
        """Extract necessary values from the inputs for aggregation.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            Any: The extracted context values required for aggregation.
        """
        ...

    @abstractmethod
    async def update(
        self, val: I, state: Any, ctx: Any, io: IOContext
    ) -> tuple[O, Any]:
        """Update the aggregation value and context.

        Args:
            val (I): The current aggregation value.
            state (Any): The current aggregation state.
            ctx (Any): The context values extracted from the input batch.
            io (IOContext): Context information for the aggregator execution.

        Returns:
            tuple[O, Any]: The updated aggregation value and state.
        """
        ...
