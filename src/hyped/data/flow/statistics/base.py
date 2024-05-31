"""This module defines all base modules and utilities for tracking data statistics.

This module provides the basic infrastructure for managing data statistics within a data
flow graph. This includes the follwing main components 

- :class:`StatisticRef`: Holds references to a statistic node within the flow graph.
- :class:`StatisticReport`: Manages the registration and tracking of statistics, ensuring thread-safe 
  updates using locks.
- :class:`BaseDataStatisticConfig`: Provides a base configuration for statistics, leveraging Pydantic for validation.
- :class:`BaseDataStatistic`: Abstract base class that requires implementation of methods for initializing values, 
  extracting context from inputs, and updating statistics.

Example Usage:

.. code-block:: python

    from hyped.data.flow import DataFlow
    # Import necessary classes from the module
    from hyped.data.statistics.base import (
        StatisticReport,
        BaseDataStatistic,
        BaseDataStatisticConfig,
    )
    from hyped.data.flow.refs.inputs import (
        InputRefs, CheckFeatureEquals
    )
    from datasets.features.features import Value
    from typing_extensions import Annotated

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]

    class CustomConfig(BaseDataStatisticConfig):
        a: float

    class CustomStatistic(BaseDataStatistic[int, CustomConfig, CustomInputRefs]):
        def initial_value(self, manager: SyncManager) -> int:
            return 0

        async def extract(self, inputs: Batch, index: int, rank: int) -> int:
            return sum(inputs['x'])

        async def update(self, value: int, ctx: int) -> int:
            return value + ctx

    # Create a data flow
    flow = DataFlow(...)
    
    custom_stat = CustomStatistic(config=CustomStatisticConfig())    
    ref = custom_stat.call(x=flow.src_features.value)

    # Create and register the statistic
    report = StatisticReport()
    # Track the statistic in a context
    with report.track(custom_stat=ref):
        # Perform data processing that updates the statistic
        ...
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from functools import partial
from multiprocessing.managers import SyncManager
from typing import Any, Callable, Generator, Generic, TypeAlias, TypeVar

from datasets import Features
from pydantic import BaseModel

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar
from hyped.common.lazy import LazyStaticInstance
from hyped.data.flow.refs.inputs import InputRefs
from hyped.data.flow.refs.ref import FeatureRef

Batch: TypeAlias = dict[str, list[Any]]
Sample: TypeAlias = dict[str, Any]
Getter: TypeAlias = Callable[[], Any]
Setter: TypeAlias = Callable[[Any], None]


def _sync_manager_factory() -> SyncManager:
    """Factory function for creating a SyncManager instance.

    Returns:
        SyncManager: An instance of SyncManager.
    """
    manager = SyncManager(ctx=mp.context.DefaultContext)
    manager.start()
    return manager


_manager = LazyStaticInstance[SyncManager](_sync_manager_factory)


class StatisticRef(BaseModel):
    """Reference to a statistic in a data flow graph.

    Attributes:
        node_id_ (int): The ID of the node in the flow graph.
        flow_ (object): The data flow graph object.
    """

    node_id_: int
    """
    The identifier of the node within the data flow graph.

    This attribute represents the identifier of the node within the data flow
    graph to which the referenced statistic belongs.
    """

    flow_: object
    """
    The data flow graph.

    This attribute represents the data flow graph to which the statistic reference belongs.
    It provides context for the statistic reference within the overall data processing flow.
    """


class StatisticReport(object):
    """Statistic Report managing and tracking statistics within a data processing system."""

    def __init__(self) -> None:
        """Initialize the StatisticReport with empty content and locks."""
        global _manager
        self.content = _manager.dict()
        self.locks = dict()

    def register(
        self, name: str, stat: BaseDataStatistic
    ) -> tuple[Getter, Setter, mp.Lock]:
        """Register a new statistic to the report.

        This method registers a statistic by its name and initializes the necessary lock and
        value for the statistic. It ensures that each statistic name is unique within the report.

        Args:
            name (str): The name of the statistic to register.
            stat (BaseDataStatistic): The statistic instance to register.

        Returns:
            tuple[Getter, Setter, mp.Lock]: A tuple containing the getter, setter, and lock for
                the registered statistic. The getter is a callable that retrieves the current
                value of the statistic. The setter is a callable that updates the value of
                the statistic. The lock is used to ensure thread-safe operations.

        Raises:
            ValueError: If a statistic with the given name is already registered.
        """
        global _manager

        if name in self.content:
            raise ValueError(
                f"A statistic with the name '{name}' is already registered."
            )

        # create lock and initial value
        lock = _manager.Lock()
        value = stat.initial_value(_manager)
        # write to report
        self.content[name] = value
        self.locks[name] = lock
        # return the initial value and the lock
        return (
            partial(self.content.__getitem__, name),
            partial(self.content.__setitem__, name),
            lock,
        )

    def track(self, **named_statistics: StatisticRef) -> ExitStack:
        """Track multiple statistics by their references.

        Args:
            **named_statistics (StatisticRef): Named statistic references to track.

        Returns:
            ExitStack: An exit stack managing the context for tracked statistics.
        """
        # create an exit-stack to manage all
        # context managers of the statistics
        stack = ExitStack()

        for name, ref in named_statistics.items():
            # register the statistic
            stat = ref.flow_.get_processor(ref.node_id_)
            getter, setter, lock = self.register(name, stat)
            # add the context manager to the stack
            report_ctx = stat.report_to(getter, setter, lock)
            stack.enter_context(report_ctx)

        return stack


class BaseDataStatisticConfig(BaseConfig):
    """Base configuration class for data statistics.

    This class serves as the base configuration class for data processors.
    It inherits from `BaseConfig`, a Pydantic model, providing basic configuration
    functionality for data processing tasks.
    """

    pass


T = TypeVar("T")
C = TypeVar("C", bound=BaseDataStatisticConfig)
I = TypeVar("I", bound=InputRefs)


class BaseDataStatistic(Generic[T, C, I], BaseConfigurable[C], ABC):
    """Abstract base class for data statistics.

    This class serves as the base for all data statistics, representing nodes in a data flow graph.
    Subclasses of `BaseDataStatistic` implement specific logic for initializing, processing, and
    reporting statistics.

    Attributes:
        _report_entries (set[tuple[T, mp.Lock]]): Set of tuples containing the value and lock for reporting.
        _in_refs_type (Type[I]): The type of input references expected by the statistic.
    """

    def __init__(self, config: None | C = None, **kwargs: Any) -> None:
        """Initialize the data statistic.

        Initializes the data statistic with the given configuration. If no configuration is provided,
        a new configuration is created using the provided keyword arguments.

        Args:
            config (C, optional): The configuration object for the statistic. If not provided,
                a configuration is created based on the given keyword arguments.
            **kwargs (Any): Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(BaseDataStatistic, self).__init__(config, **kwargs)

        self._in_refs_type = solve_typevar(type(self), I)
        self._report_entries: set[tuple[T, mp.Lock]] = set()

    @property
    def is_tracked(self) -> bool:
        """Check if the statistic is currently being tracked.

        Returns:
            bool: True if the statistic is being tracked, False otherwise.
        """
        return len(self._report_entries) > 0

    def call(self, **kwargs: FeatureRef) -> StatisticRef:
        """Build inputs from keyword arguments and add the statistic to the flow graph.

        Args:
            **kwargs (FeatureRef): Keyword arguments representing feature references.

        Returns:
            StatisticRef: A reference to the statistic.
        """
        # build inputs from keyword arguments
        inputs = self._in_refs_type(**kwargs)
        # add statistic to flow graph
        node_id = inputs.flow.add_processor_node(self, inputs, Features())
        # return the statistic reference
        return StatisticRef(node_id_=node_id, flow_=inputs.flow)

    @contextmanager
    def report_to(
        self, getter: Getter, setter: Setter, lock: mp.Lock
    ) -> Generator[None, None, None]:
        """Context manager for reporting the statistic.

        This context manager is used to track and report statistics in a thread-safe manner.
        It manages the registration and deregistration of getter, setter, and lock entries
        for reporting purposes.

        Args:
            getter (Getter): The function or callable to get the current value of the statistic.
            setter (Setter): The function or callable to set the updated value of the statistic.
            lock (mp.Lock): The lock associated with the value to ensure thread-safe updates.

        Returns:
            ContextManager: Context manager that removes the (getter, setter, lock)-entry on exit.
        """
        entry = (getter, setter, lock)
        self._report_entries.add(entry)
        yield
        self._report_entries.remove(entry)

    @abstractmethod
    def initial_value(self, manager: SyncManager) -> T:
        """Abstract method to get the initial value for the statistic.

        Args:
            manager (SyncManager): The sync manager to use.

        Returns:
            T: The initial value for the statistic.
        """
        ...

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> None:
        """Process a batch of inputs and update the statistic.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.
        """
        # extract context values from the inputs
        ctx = await self.extract(inputs, index, rank)

        # get the current event loop
        loop = asyncio.get_running_loop()

        async def _safe_update(getter, setter, lock):
            # acquire the lock non-blocking
            await loop.run_in_executor(None, lock.acquire)
            # compute the updated value
            val = await self.update(getter(), ctx)
            # hide the communication overhead of the set operation
            # in a separate thread
            await loop.run_in_executor(None, partial(setter, val))
            # release the lock
            lock.release()

        await asyncio.gather(
            *(
                _safe_update(getter, setter, lock)
                for getter, setter, lock in self._report_entries
            )
        )

    async def extract(self, inputs: Batch, index: int, rank: int) -> Any:
        """Extract context values from inputs for updating the statistic.

        This method is responsible for processing the input batch and extracting relevant context
        values needed for updating the statistic. The context values typically include features or
        other data that will be used in the `update` method to modify the statistic.

        Args:
            inputs (Batch): The batch of input samples.
            index (int): The index of the input to extract context values for.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            Any: The extracted context values, which will be passed to the `update` method.
        """
        raise NotImplementedError()

    async def update(self, value: T, ctx: Any) -> T:
        """Update the statistic with the given value and context.

        This method is responsible for updating the statistic using the provided value and context.
        The context, provided by the `extract` method, includes the necessary data needed to accurately
        update the statistic.

        Args:
            value (T): The current value of the statistic to be updated.
            ctx (Any): The context for updating the statistic, as extracted from inputs.

        Returns:
            T: The updated statistic value to be reported.
        """
        raise NotImplementedError()
