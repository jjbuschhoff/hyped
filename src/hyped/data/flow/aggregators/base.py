"""Provides base classes for data aggregators in a data flow graph.

This module defines the base class for data aggregators, which manage the aggregation
of data within a data flow graph. It includes generic classes for defining data aggregators
with configurable input types and aggregation logic.

Classes:
    - :class:`DataAggregationRef`: Reference to a data aggregation node in the data flow.
    - :class:`DataAggregationManager`: Manager for handling data aggregation operations.
    - :class:`BaseDataAggregatorConfig`: Base class for data aggregator configurations.
    - :class:`BaseDataAggregator`: Base class for data aggregators.

Usage Example:
    Define a custom data aggregator by subclassing :code:`BaseDataAggregator`:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.aggregators.base import (
            BaseDataAggregator, BaseDataAggregatorConfig, DataAggregationRef, DataAggregationManager
        )
        from hyped.data.flow.refs.inputs import InputRefs
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
from collections import defaultdict
from multiprocessing.managers import SyncManager
from types import MappingProxyType
from typing import Any, Generic, TypeAlias, TypeVar

from datasets import Features
from pydantic import BaseModel

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar
from hyped.common.lazy import LazyStaticInstance
from hyped.data.flow.refs.inputs import InputRefs

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


class DataAggregationRef(BaseModel):
    """Reference to a data aggregation node in the data flow.

    Attributes:
        node_id_ (int): The ID of the aggregation node.
        flow_ (object): The data flow object.
        type_ (type): The type of the aggregation value.
    """

    node_id_: int
    """The ID of the aggregation node."""

    flow_: object
    """The data flow object."""

    type_: type
    """The type of the aggregation value."""


class DataAggregationManager(object):
    """Manager for handling data aggregation operations.

    This class manages data aggregators, their thread-safe buffers, and synchronization
    locks to ensure safe concurrent updates during the data processing.

    Attributes:
        _value_buffer (dict): Thread-safe buffer for aggregation values.
        _state_buffer (dict): Thread-safe buffer for aggregation states.
        _locks (dict): Locks for synchronizing access to aggregators.
        _lookup (dict): Lookup table mapping aggregators to their respective names.
    """

    def __init__(
        self,
        aggregators: dict[str, BaseDataAggregator],
        in_features: dict[str, Features],
    ) -> None:
        """Initialize the DataAggregationManager.

        Args:
            aggregators (dict[str, BaseDataAggregator]): A dictionary of aggregators.
            in_features (dict[str, Features]): A dictionary of input features.
        """
        global _manager
        # create buffers
        value_buffer = {}
        state_buffer = {}
        # fill buffers with initial values from aggregators
        for name, agg in aggregators.items():
            value_buffer[name], state_buffer[name] = agg.initialize(
                in_features[name]
            )
        # create thread-safe buffers
        self._value_buffer = _manager.dict(value_buffer)
        self._state_buffer = _manager.dict(state_buffer)
        # create a lock for each entry to synchronize access
        self._locks = {name: _manager.Lock() for name in aggregators.keys()}
        self._locks = _manager.dict(self._locks)

        # create aggregator lookup
        self._lookup: dict[BaseDataAggregator, list[str]] = defaultdict(set)
        for name, agg in aggregators.items():
            self._lookup[agg].add(name)

    @property
    def values_proxy(self) -> MappingProxyType[str, Any]:
        """Get a read-only view of the aggregation values.

        Returns:
            MappingProxyType[str, Any]: A read-only view of the aggregation values.
        """
        return MappingProxyType(self._value_buffer)

    async def _safe_update(
        self, name: str, aggregator: BaseDataAggregator, ctx: Any
    ) -> None:
        """Safely update an aggregation value.

        Args:
            name (str): The name of the aggregator.
            aggregator (BaseDataAggregator): The aggregator object.
            ctx (Any): The context values extracted from the input batch.
        """
        assert name in self._value_buffer
        # get the running event loop
        loop = asyncio.get_running_loop()
        # acquire the lock for the current aggregator
        await loop.run_in_executor(None, self._locks[name].acquire)
        # get current value and context
        value = self._value_buffer[name]
        state = self._state_buffer[name]
        # compute udpated value and context
        value, state = await aggregator.update(value, ctx, state)
        # write new values to buffers
        self._value_buffer[name] = value
        self._state_buffer[name] = state
        # release lock
        self._locks[name].release()

    async def aggregate(
        self,
        aggregator: BaseDataAggregator,
        inputs: Batch,
        index: list[int],
        rank: int,
    ) -> None:
        """Perform aggregation for a batch of inputs.

        Args:
            aggregator (BaseDataAggregator): The aggregator object.
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.
        """
        # extract values required for update from current input batch
        ctx = await aggregator.extract(inputs, index, rank)
        # update all entries
        await asyncio.gather(
            *(
                self._safe_update(name, aggregator, ctx)
                for name in self._lookup[aggregator]
            )
        )


class BaseDataAggregatorConfig(BaseConfig):
    """Base configuration class for data aggregators.

    This class serves as the base configuration class for data aggregators.
    It inherits from :class:`BaseConfig`, providing basic configuration
    functionality for data aggregation tasks.
    """


C = TypeVar("C", bound=BaseDataAggregatorConfig)
I = TypeVar("I", bound=InputRefs)
T = TypeVar("T")


class BaseDataAggregator(BaseConfigurable[C], Generic[C, I, T], ABC):
    """Base class for data aggregators.

    This class serves as the base for all data aggregators, defining the necessary
    interfaces and methods for implementing custom aggregators.

    Attributes:
        _in_refs_type (Type[I]): The type of input references expected by the aggregator.
        _value_type (Type[T]): The type of the aggregation value.
    """

    def __init__(self, config: None | C = None, **kwargs) -> None:
        """Initialize the data aggregator.

        Initializes the data aggregator with the given configuration. If no configuration is
        provided, a new configuration is created using the provided keyword arguments.

        Args:
            config (C, optional): The configuration object for the data aggregator.
                If not provided, a configuration is created based on the given keyword arguments.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(BaseDataAggregator, self).__init__(config, **kwargs)
        self._in_refs_type = solve_typevar(type(self), I)
        self._value_type = solve_typevar(type(self), T)

    @property
    def required_input_keys(self) -> set[str]:
        """Retrieves the set of input keys required by the processor.

        Returns:
            set[str]: The set of input keys.
        """
        return self._in_refs_type.required_keys

    def call(self, **kwargs) -> DataAggregationRef:
        """Call the data aggregator with the provided inputs.

        This method builds inputs from keyword arguments, adds the aggregator
        to the data flow, and returns a reference to the aggregation node.

        Args:
            **kwargs: Keyword arguments specifying feature references to be
                passed as inputs to the aggregator.

        Returns:
            DataAggregationRef: The reference to the aggregation node.
        """
        # build inputs from keyword arguments and add the aggregator to the flow
        inputs = self._in_refs_type(**kwargs)
        node_id = inputs.flow.add_processor_node(self, inputs, None)
        # build the aggregation reference
        return DataAggregationRef(
            node_id_=node_id, flow_=inputs.flow, type_=self._value_type
        )

    @abstractmethod
    def initialize(self, features: Features) -> tuple[T, Any]:
        """Initialize the aggregator with the given features.

        Args:
            features (Features): The features to initialize the aggregator with.

        Returns:
            tuple[T, Any]: The initial value and state for the aggregator.
        """
        ...

    @abstractmethod
    async def extract(self, inputs: Batch, index: list[int], rank: int) -> Any:
        """Extract necessary values from the inputs for aggregation.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            Any: The extracted context values required for aggregation.
        """
        ...

    @abstractmethod
    async def update(self, val: T, state: Any, ctx: Any) -> tuple[T, Any]:
        """Update the aggregation value and context.

        Args:
            val (T): The current aggregation value.
            state (Any): The current aggregation state.
            ctx (Any): The context values extracted from the input batch.

        Returns:
            tuple[T, Any]: The updated aggregation value and state.
        """
        ...
