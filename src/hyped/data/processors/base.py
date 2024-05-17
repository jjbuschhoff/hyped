from __future__ import annotations

import inspect
import asyncio
from abc import ABC
from typing import Any, TypeVar, Generic

from hyped.data.ref import FeatureRef
from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar

Batch: TypeVar = dict[str, list[Any]]
Sample: TypeVar = dict[str, Any]


class BaseDataProcessorConfig(BaseConfig):
    pass


# import after processor config to avoid circular imports
from .inputs import InputRefs
from .outputs import OutputRefs

C = TypeVar("C", bound=BaseDataProcessorConfig)
I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataProcessor(BaseConfigurable[C], Generic[C, I, O], ABC):
    
    def __init__(self, config: C) -> None:
        self._config = config
        # check whether the process function is a coroutine
        self._is_process_async = inspect.iscoroutinefunction(self.process)

        self._in_refs_type = solve_typevar(type(self), I)
        self._out_refs_type = solve_typevar(type(self), O)

    @classmethod
    def from_config(cls, config: C) -> BaseDataProcessor:
        return cls(config)

    @property
    def config(self) -> C:
        return self._config

    def call(self, inputs: None | I = None, **kwargs) -> O:

        if inputs is not None and len(kwargs) != 0:
            raise TypeError(
                "Please specify either 'inputs' or keyword arguments, but not "
                "both."
            )

        # build inputs from keyword arguments if needed
        inputs = (
            inputs if inputs is not None
            else self._in_refs_type(**kwargs)
        )
        # add new node to flow and return the output refs
        node_id = inputs.flow.add_processor(self, inputs.named_refs)
        return self._out_refs_type(self.config, inputs, node_id)

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> tuple[dict[FeatureRef, list[Any]], list[int]]:

        # apply process function to each sample in the input batch
        keys = inputs.keys()
        outputs = [
            self.process(dict(zip(keys, values)), i, rank)
            for i, values in zip(index, zip(*inputs.values()))
        ]
        # gather all outputs in case the process function
        # is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        # pack output samples to batch format
        return {
            key: [d[key] for d in outputs]
            for key in self._out_refs_type._feature_names
        }, index

    async def process(
        self, inputs: Sample, index: int, rank: int
    ) -> dict[FeatureRef, Any]:
        ...

    def process(
        self, inputs: Sample, index: int, rank: int
    ) -> dict[FeatureRef, Any]:
        ...

