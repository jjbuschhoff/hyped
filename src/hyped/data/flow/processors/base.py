"""Provides base classe for data processors in a data flow graph.

This module defines the base class for data processors, which represent
nodes in a data flow graph. It includes generic classes for defining
data processors with configurable input and output types.

Classes:
    - :class:`BaseDataProcessorConfig`: Base class for data processor configurations.
    - :class:`BaseDataProcessor`: Base class for data processors in a data flow graph.

Usage Example:
    Define a custom data processor by subclassing `BaseDataProcessor`:

    .. code-block:: python

        # Import necessary classes from the module
        from hyped.data.processors.base import (
            BaseDataProcessor, BaseDataProcessorConfig
        )
        from hyped.data.flow.refs.inputs import (
            InputRefs, CheckFeatureEquals
        )
        from hyped.data.flow.refs.outputs import (
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
            async def process(self, inputs, index, rank):
                # Define processing logic here
                return str(inputs["x"]) * self.config.k

    In this example, :class:`CustomDataProcessor` extends :class:`BaseDataProcessor` and
    implements the :class:`BaseDataProcessor.process` method to define custom processing logic.
"""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC
from typing import Any, Generic, TypeVar

from typing_extensions import TypeAlias

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar
from hyped.data.flow.refs.inputs import InputRefs
from hyped.data.flow.refs.outputs import OutputRefs

Batch: TypeAlias = dict[str, list[Any]]
Sample: TypeAlias = dict[str, Any]


class BaseDataProcessorConfig(BaseConfig):
    """Base configuration class for data processors.

    This class serves as the base configuration class for data processors.
    It inherits from `BaseConfig`, a Pydantic model, providing basic configuration
    functionality for data processing tasks.
    """


C = TypeVar("C", bound=BaseDataProcessorConfig)
I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataProcessor(BaseConfigurable[C], Generic[C, I, O], ABC):
    """Base class for data processors in a data flow graph.

    This class serves as the base for all data processors, representing nodes in a data flow graph.
    Subclasses of `BaseDataProcessor` implement specific process functions that map input features
    to output features. Custom data processors must either override the `batch_process` method or
    the `process` method to define their processing logic.

    Attributes:
        _is_process_async (bool): A flag indicating whether the :class:`BaseDataProcessor.process`
            function is asynchronous.
        _in_refs_type (Type[I]): The type of input references expected by the processor.
        _out_refs_type (Type[O]): The type of output references produced by the processor.
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
        # build final configuration from given arguments
        if config is None:
            config = self.config_type(**kwargs)
        elif len(kwargs) is not None:
            config = config.model_copy(update=kwargs)

        self._config = config
        # check whether the process function is a coroutine
        self._is_process_async = inspect.iscoroutinefunction(self.process)

        self._in_refs_type = solve_typevar(type(self), I)
        self._out_refs_type = solve_typevar(type(self), O)

    @classmethod
    def from_config(cls, config: C) -> BaseDataProcessor:
        """Creates a data processor instance from the provided configuration.

        Args:
            config (C): The configuration object for the data processor.

        Returns:
            BaseDataProcessor: An instance of the data processor.
        """
        return cls(config)

    @property
    def config(self) -> C:
        """Retrieves the configuration object for the data processor.

        Returns:
            C: The configuration object.
        """
        return self._config

    @property
    def required_input_keys(self) -> set[str]:
        """Retrieves the set of input keys required by the processor.

        Returns:
            set[str]: The set of input keys.
        """
        return self._in_refs_type.required_keys

    def call(self, inputs: None | I = None, **kwargs) -> O:
        """Calls the data processor with the provided inputs and returns the output reference.

        This method prepares the inputs, either from the provided argument or from the
        keyword arguments, then adds the processor to the data flow and returns a feature
        reference to the output features of the processor.

        Args:
            inputs (None | I, optional): The input references to the processor. Defaults to None.
            **kwargs: Keyword arguments to be passed as inputs instead of the `inputs` argument.

        Returns:
            O: The output references produced by the processor.

        Raises:
            TypeError: If both `inputs` and keyword arguments are specified.
        """
        if inputs is not None and len(kwargs) != 0:
            raise TypeError(
                "Please specify either 'inputs' or keyword arguments, but not "
                "both."
            )

        # build inputs from keyword arguments if needed
        inputs = inputs if inputs is not None else self._in_refs_type(**kwargs)
        # add new node to flow and return the output refs
        return inputs.flow.add_processor_node(self, inputs)

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int
    ) -> tuple[Batch, list[int]]:
        """Processes a batch of inputs and returns the corresponding batch of outputs.

        Args:
            inputs (Batch): The batch of input samples.
            index (list[int]): The indices associated with the input samples.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            tuple[Batch, list[int]]: The batch of output samples and the corresponding indices.
        """
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
            key: [d[key] for d in outputs] for key in outputs[0].keys()
        }, index

    async def process(self, inputs: Sample, index: int, rank: int) -> Sample:
        """Processes a single input sample asynchronously and returns the corresponding output sample.

        This method should be overridden by subclasses to define the processing logic.

        Args:
            inputs (Sample): The input sample to be processed.
            index (int): The index associated with the input sample.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            Sample: The processed output sample.
        """
        ...

    def process(self, inputs: Sample, index: int, rank: int) -> Sample:
        """Processes a single input sample synchronously and returns the corresponding output sample.

        Asynchronous processing is also supported by defining this function as :class:`async`.

        This method should be overridden by subclasses to define the processing logic.

        Args:
            inputs (Sample): The input sample to be processed.
            index (int): The index associated with the input sample.
            rank (int): The rank of the processor in a distributed setting.

        Returns:
            Sample: The processed output sample.
        """
        raise NotImplementedError()
