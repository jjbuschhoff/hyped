"""Manages lazy evaluation and caching of data flow outputs.

This module provides the :class:`LazyFlowOutput` class, which is responsible for lazily 
evaluating and caching the results of a data flow execution. The class wraps 
around a read-only dictionary-like proxy of input data and a data flow executor. 
It ensures that the computation is executed only once per set of input values 
and caches the result for subsequent accesses.
"""

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Hashable, Iterable

from .executor import DataFlowExecutor


class LazyFlowOutput(Mapping):
    """A read-only dict-like object that lazily executes and caches a data flow graph.

    This class provides a way to access the results of a data flow execution,
    recalculating only when the input data changes. It ensures that the
    computation is executed only once per set of input values and caches the
    result for subsequent accesses.
    """

    def __init__(
        self,
        input_proxy: MappingProxyType[str, Any],
        executor: DataFlowExecutor,
    ) -> None:
        """Initialize the :code:`LazyFlowOutput`.

        Args:
            input_proxy (MappingProxyType[str, Any]): A read-only proxy for the input data.
            executor (DataFlowExecutor): An executor that handles the data flow execution.
        """
        self._proxy = input_proxy
        self._executor = executor
        # snapshot
        self._proxy_snapshot = None
        self._out_snapshot = None

    def keys(self) -> Iterable[Hashable]:
        """Get the keys of the output features.

        Returns:
            Iterable[Hashable]: An iterable of the output feature keys.
        """
        return self._executor.collect.feature_.keys()

    def _get_values(self) -> MappingProxyType[Hashable, Any]:
        """Compute and cache the output values if the inputs have changed.

        Returns:
            MappingProxyType[Hashable, Any]: A read-only proxy to the computed output values.
        """
        proxy_snapshot = dict(self._proxy)
        # check
        if self._proxy_snapshot != proxy_snapshot:
            # build batch of inputs
            inputs = {k: [v] for k, v in proxy_snapshot.items()}
            # execute the flow executor on the inputs
            loop = asyncio.new_event_loop()
            future = self._executor.execute(inputs, index=[0], rank=0)
            output = loop.run_until_complete(future)
            # close the event loop
            loop.close()
            # parse the outputs and store them as the snapshot
            self._proxy_snapshot = proxy_snapshot
            self._out_snapshot = {k: v[0] for k, v in output.items()}

        return MappingProxyType(self._out_snapshot)

    def __getitem__(self, key: Hashable) -> Any:
        """Get the value associated with the specified key.

        Args:
            key (Hashable): The key of the desired output value.

        Returns:
            Any: The value associated with the specified key.

        Raises:
            KeyError: If the key is not in the output features.
        """
        if key not in self.keys():
            raise KeyError(key)

        return self._get_values()[key]

    def __iter__(self):
        """Get an iterator over the keys of the output features.

        Returns:
            Iterator[Hashable]: An iterator over the output feature keys.
        """
        return iter(self.keys())

    def __len__(self):
        """Get the number of output features.

        Returns:
            int: The number of output features.
        """
        return len(self.keys())

    def __repr__(self):
        """Get the string representation of the LazyFlowOutput.

        Returns:
            str: The string representation of the LazyFlowOutput.
        """
        return "LazyFlowOutput(input_proxy=%s, executor=%s)" % (
            self._proxy,
            self._executor,
        )

    def __str__(self):
        """Get the string representation of the LazyFlowOutput.

        Returns:
            str: The string representation of the LazyFlowOutput.
        """
        return str(dict(self))
