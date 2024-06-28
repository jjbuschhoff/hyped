"""Provides base classe for data processors in a data flow graph.

This module defines the base class for data processors, which represent
nodes in a data flow graph. It includes generic classes for defining
data processors with configurable input and output types.

Classes:
    - :class:`BaseDataProcessorConfig`: Base class for data processor configurations.
    - :class:`BaseDataProcessor`: Base class for data processors in a data flow graph.

Usage Example:
    Define a custom data processor by subclassing :class:`BaseDataProcessor`:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.flow.core.nodes.processor import (
            BaseDataProcessor, BaseDataProcessorConfig
        )
        from hyped.data.flow.core.refs.inputs import (
            InputRefs, CheckFeatureEquals
        )
        from hyped.data.flow.core.refs.outputs import (
            OutputRefs, OutputFeature
        )
        from datasets.features.features import Value
        from typing_extensions import Annotated

        class CustomInputRefs(InputRefs):
            x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]

        class CustomOutputRefs(OutputRefs):
            y: Annotated[FeatureRef, OutputFeature(Value("string"))]
        
        class CustomConfig(BaseDataProcessorConfig):
            k: int

        # Define a custom data processor class
        class CustomProcessor(
            BaseDataProcessor[CustomConfig, CustomInputRefs, CustomOutputRefs]
        ):
            async def process(self, inputs, index, rank, io):
                # Define processing logic here
                return str(inputs["x"]) * self.config.k

    In this example, :class:`CustomDataProcessor` extends :class:`BaseDataProcessor` and
    implements the :class:`BaseDataProcessor.process` method to define custom processing logic.
"""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from datasets import Features
from typing_extensions import TypeAlias

from ..refs.inputs import InputRefs
from ..refs.outputs import OutputRefs
from .base import BaseNode, BaseNodeConfig, IOContext

Batch: TypeAlias = dict[str, list[Any]]
Sample: TypeAlias = dict[str, Any]


class BaseDataProcessorConfig(BaseNodeConfig):
    """Base configuration class for data processors.

    This class serves as the base configuration class for data processors.
    It inherits from `BaseConfig`, a Pydantic model, providing basic configuration
    functionality for data processing tasks.
    """


C = TypeVar("C", bound=BaseDataProcessorConfig)
I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataProcessor(BaseNode[C, I, O], ABC):
    """Base class for data processors in a data flow graph.

    This class serves as the base for all data processors, representing nodes in a data flow graph.
    Subclasses of `BaseDataProcessor` implement specific process functions that map input features
    to output features. Custom data processors must either override the `batch_process` method or
    the `process` method to define their processing logic.

    Attributes:
        _is_process_async (bool): A flag indicating whether the :class:`BaseDataProcessor.process`
            function is asynchronous.
    """

    def __init__(self, config: None | C = None, **kwargs) -> None:
        """Initialize the data processor.

        Initializes the data processor with the given configuration. If no configuration is provided,
        a new configuration is created using the provided keyword arguments.

        Args:
            config (C, optional): The configuration object for the data processor. If not provided,
                a configuration is created based on the given keyword arguments.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(BaseDataProcessor, self).__init__(config, **kwargs)
        # check whether the process function is a coroutine
        self._is_process_async = inspect.iscoroutinefunction(self.process)

    @property
    def required_input_keys(self) -> set[str]:
        """Retrieves the set of input keys required by the processor.

        Returns:
            set[str]: The set of input keys.
        """
        return self._in_refs_validator.required_keys

    def call(self, **kwargs) -> O:
        """Calls the data processor with the provided inputs and returns the output reference.

        This method first prepares the inputs, then adds the processor to the data
        flow and returns a feature reference to the output features of the processor.

        Args:
            **kwargs: Keyword arguments specifying feature references to be passed
                as inputs to the processor.

        Returns:
            O: The output references produced by the processor.
        """
        # build inputs from keyword arguments
        inputs = self._in_refs_type(**kwargs)
        inputs = self._in_refs_validator.validate(inputs)
        # compute output features and add the processor to the data flow
        out_features = self._out_refs_type.build_features(self.config, inputs)
        node_id = inputs.flow.add_processor_node(self, inputs, out_features)
        # return the output feature refs
        return self._out_refs_type(inputs.flow, node_id, out_features)

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs and returns the corresponding batch of outputs.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch of output samples, must keep the order of the input batch.
        """
        # apply process function to each sample in the input batch
        keys = inputs.keys()
        outputs = [
            self.process(dict(zip(keys, values)), i, rank, io)
            for i, values in zip(index, zip(*inputs.values()))
        ]
        # gather all outputs in case the process function
        # is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        # pack output samples to batch format
        return {key: [d[key] for d in outputs] for key in outputs[0].keys()}

    @overload
    async def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        ...

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Processes a single input sample synchronously and returns the corresponding output sample.

        Asynchronous processing is also supported by defining this function as :class:`async`.

        This method should be overridden by subclasses to define the processing logic.

        Args:
            inputs (Sample): The input sample to be processed.
            index (int): The index associated with the input sample.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Sample: The processed output sample.
        """
        raise NotImplementedError()
