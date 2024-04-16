"""Base Data Processor."""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from itertools import chain, repeat
from types import GeneratorType
from typing import Any, ClassVar, Generator, Iterable, TypeVar

from datasets import Features
from datasets.iterable_dataset import _batch_to_examples, _examples_to_batch

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.common.feature_key import FeatureKey, FeatureKeyCollection


class BaseDataProcessorConfig(BaseConfig):
    """Base Data Processor Config.

    Attributes:
        keep_input_features (bool):
            whether to pipe input features to output or only output
            features generated by the data processor
        output_format (None | dict[str, FeatureKeyCollection]):
            specifies the output feature scheme, i.e. the structure
            of the new features computed by the data processor
    """

    # specify fields that will not be parsed for feature keys
    # by default the output format is ignored as the feature keys
    # refer to output features and not required input features
    _IGNORE_KEYS_FROM_FIELDS: ClassVar[list[str]] = ["output_format"]

    # attributes
    keep_input_features: bool = True
    output_format: None | FeatureKeyCollection = None

    @property
    def required_feature_keys(self) -> Iterable[FeatureKey]:
        """Required Feature Keys.

        Iterator over all feature keys required for execution of the
        corresponding data processor.
        """

        def _iter_feature_keys(col):
            if isinstance(col, FeatureKey):
                yield col

            if isinstance(col, (list, tuple)):
                yield from chain.from_iterable(map(_iter_feature_keys, col))

            if isinstance(col, FeatureKeyCollection):
                yield from col.feature_keys

            if isinstance(col, dict):
                yield from chain.from_iterable(
                    map(_iter_feature_keys, col.values())
                )

        yield from chain.from_iterable(
            map(
                _iter_feature_keys,
                (
                    getattr(self, k)
                    for k in self.model_fields.keys()
                    if k not in type(self)._IGNORE_KEYS_FROM_FIELDS
                ),
            )
        )


T = TypeVar("T", bound=BaseDataProcessorConfig)


class BaseDataProcessor(BaseConfigurable[T], ABC):
    """Abstract Base Data Processor.

    Provides basic functionality of a data-processor. Sub-types need to
    specify the `map_features` and either the `process` or
    `internal_batch_process` function.

    Arguments:
        config (BaseDataProcessorConfig): data processor configuration
    """

    def __init__(self, config: BaseDataProcessorConfig) -> None:
        """Initialize Data Processor.

        Arguments:
            config (BaseDataProcessorConfig): processor config
        """
        self._config = config
        self._in_features: Features = None
        self._raw_features: Features = None
        self._new_features: Features = None
        # check whether the process function is a coroutine
        self._is_process_gen = inspect.isgeneratorfunction(self.process)
        self._is_process_async = inspect.iscoroutinefunction(self.process)
        self._is_process_async_gen = inspect.isasyncgenfunction(self.process)

    @classmethod
    def from_config(cls, config: BaseDataProcessorConfig) -> BaseDataProcessor:
        """Instantiate data processor from the given config.

        Arguments:
            config (BaseDataProcessorConfig): data processor configuration
        """
        return cls(config)

    @property
    def config(self) -> BaseDataProcessorConfig:
        """Get the processor configuration.

        Returns:
            config (BaseDataProcessorConfig): config
        """
        return self._config

    @property
    def is_prepared(self) -> bool:
        """Check if the processor is prepared and ready for execution.

        Returns:
            is_prepared (bool): boolean indicating if the processor is prepared
        """
        return (self._in_features is not None) and (
            self._raw_features is not None
        )

    def prepare(self, features: Features) -> Features:
        """Prepare the processor for execution.

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features of the output of the processor
        """
        # save input features
        self._in_features = features
        # map input features to output features
        # copy as preparation might disturb features inplace
        self._raw_features = Features(self.map_features(features.copy()))
        # apply output scheme to new features
        if (
            (self.config.output_format is not None)
            and isinstance(self.config.output_format, dict)
            and (len(self.config.output_format) > 0)
        ):
            self._new_features = self.config.output_format.collect_features(
                self._raw_features
            )
            assert isinstance(self._new_features, Features)
        # return output features
        return self.out_features

    @property
    def required_feature_keys(self) -> list[FeatureKey]:
        """Input dataset feature keys required for execution of the processor.

        These must be contained in the `in_features`.

        Returns:
            feature_keys (list[FeatureKey]): list of required feature keys
        """
        # TODO: make list unique
        return list(self.config.required_feature_keys)

    @property
    def in_features(self) -> Features:
        """Input dataset features available to processor.

        Returns:
            features (Features): input dataset features
        """
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data processor not prepared. Did you forget to "
                "call `prepare` before execution?"
            )
        # return features
        return self._in_features

    @property
    def raw_features(self) -> Features:
        """New (raw) dataset features generated by the processor.

        These are the raw features before applying the `output_format`.
        If `outout_format` is not set, the `new_features` and
        `raw_features` match.

        Returns:
            features (Features): new dataset features
        """
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data processor not prepared. Did you forget to "
                "call `prepare` before execution?"
            )
        # return features
        return self._raw_features

    @property
    def new_features(self) -> Features:
        """New dataset features generated by the processor.

        These are the features after applying the `output_format`.
        If `outout_format` is not set, the `new_features` and
        `raw_features` match.

        Returns:
            features (Features): new dataset features
        """
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data processor not prepared. Did you forget to "
                "call `prepare` before execution?"
            )
        # return features
        return (
            self._new_features
            if self._new_features is not None
            else self._raw_features
        )

    @property
    def out_features(self) -> Features:
        """All output dataset features.

        All output features of the processor. Includes both input
        features and new features generated by the processor. On conflicts,
        the new features are prioritized.

        Returns:
            features (Features): complete output dataset features
        """
        if self.config.keep_input_features:
            return Features(self.in_features | self.new_features)
        else:
            return self.new_features

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: int,
        return_index: bool = False,
    ) -> dict[str, list[Any]] | tuple[dict[str, list[Any]], list[int]]:
        """Process a batch of examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank
            return_index (bool):
                whether to return the source index for each output example

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            index (list[int]):
                the source indices to each example. Only returned when
                `return_index` is set to true.
        """
        # process batch
        out_batch, src_index = self.internal_batch_process(
            examples, index, rank
        )

        if self.config.output_format is not None:
            out_batch = self.config.output_format.collect_batch(out_batch)

        if self.config.keep_input_features:
            # check if the src index is not range(n)
            if (len(index) != len(src_index)) or any(
                i != j for i, j in enumerate(src_index)
            ):
                # gather input features to each output example
                # this filters/duplicates or reorders the input
                # features to match the output batch
                examples = {
                    k: [v[i] for i in src_index] for k, v in examples.items()
                }
                # gather the out indices
                index = [index[i] for i in src_index]

            # add input features to output batch
            out_batch = dict(examples | out_batch)

        # return output examples
        return (out_batch, index) if return_index else out_batch

    def _sync_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Synchoronous batch process.

        Called by the default `internal_batch_process` when the implementation of
        the `process` function returns exactly one output example.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            src_index (list[int]):
                the index of the source example in the input batch that
                generated the output example, specifically the i-th output
                element is generated from the src_index[i]-th input example
        """
        # process each example one at a time and
        # stack them into a batch
        out_batch = _examples_to_batch(
            [
                self.process(x, index=i, rank=rank)
                for i, x in zip(index, _batch_to_examples(examples))
            ]
        )
        # return the output batch
        return out_batch, range(len(index))

    async def _async_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Asynchoronous batch process.

        Called by the default `internal_batch_process` when the
        `process` function is implemented as a coroutine which
        returns exactly one output example.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            src_index (list[int]):
                the index of the source example in the input batch that
                generated the output example, specifically the i-th output
                element is generated from the src_index[i]-th input example
        """
        # process and gather all output examples
        out_examples = await asyncio.gather(
            *(
                self.process(x, index=i, rank=rank)
                for i, x in zip(index, _batch_to_examples(examples))
            )
        )
        # stack output examples into batch
        return _examples_to_batch(out_examples), range(len(index))

    def _sync_gen_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Synchoronous generator batch process.

        Called by the default `internal_batch_process` when the
        `process` function is implemented as a generator that could
        yield any number of output examples for a single input example.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            src_index (list[int]):
                the index of the source example in the input batch that
                generated the output example, specifically the i-th output
                element is generated from the src_index[i]-th input example
        """
        generators = []
        for j, i, x in zip(
            range(len(index)), index, _batch_to_examples(examples)
        ):
            # process example and zip output generator with
            # the source index
            y = self.process(x, index=i, rank=rank)
            idx_and_out = zip(repeat(j), y)
            # add to the collection of all generators
            generators.append(idx_and_out)

        # chain all generators and separate the source index
        # from the generated output examples
        idx_and_out_packed = chain.from_iterable(generators)
        idx_and_out = tuple(zip(*idx_and_out_packed))

        # check if any output is generated
        if len(idx_and_out) == 0:
            empty = {key: [] for key in self.raw_features.keys()}
            return empty, []

        # pack examples into batch and return
        src_index, out_examples = idx_and_out
        return _examples_to_batch(out_examples), src_index

    async def _async_gen_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Asynchoronous generator batch process.

        Called by the default `internal_batch_process` when the
        `process` function is implemented as a coroutine generator
        that could yield any number of output examples for a single
        input example.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            src_index (list[int]):
                the index of the source example in the input batch that
                generated the output example, specifically the i-th output
                element is generated from the src_index[i]-th input example
        """

        async def consume(i, g):
            """Helper function to consume a async generator."""
            return zip(repeat(i), [x async for x in g])

        # process each example and consume it's output
        # sample generator in a separate coroutine
        coro = [
            consume(j, self.process(x, index=i, rank=rank))
            for j, i, x in zip(
                range(len(index)), index, _batch_to_examples(examples)
            )
        ]

        # chain all generators and separate the source index
        # from the generated output examples
        idx_and_out_packed = chain.from_iterable(await asyncio.gather(*coro))
        idx_and_out = tuple(zip(*idx_and_out_packed))
        # check if any output is generated
        if len(idx_and_out) == 0:
            empty = {key: [] for key in self.raw_features.keys()}
            return empty, []

        # pack examples into batch and return
        src_index, out_examples = idx_and_out
        return _examples_to_batch(out_examples), src_index

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Internal batch process.

        By default, routes the batch of examples to the internal batch
        process function appropriate to the specific implementation of
        the `process` function.

        Specifically the routing is as follows:

            def process(x): return y        => sync_batch_process
            def process(x): yield y         => sync_gen_batch_process
            async def process(x): return y  => async_batch_process
            async def process(x): yield y   => async_gen_batch_process

        Each of these functions by default pass every example of the batch
        to the `process` function one-by-one and gather the ouputs into a
        batch.

        Overwrite this function to process a whole batch simultaneously
        instead of single examples.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out_batch (dict[str, list[Any]]): processed examples
            src_index (list[int]):
                the index of the source example in the input batch that
                generated the output example, specifically the i-th output
                element is generated from the src_index[i]-th input example
        """
        if self._is_process_async or self._is_process_async_gen:
            # apply coroutine appropriate to the process function type
            future = (
                self._async_gen_batch_process
                if self._is_process_async_gen
                else self._async_batch_process
            )(examples, index, rank)
            # get the event loop to run the async function
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(future)

        # run sync
        return (
            self._sync_gen_batch_process
            if self._is_process_gen
            else self._sync_batch_process
        )(examples, index, rank)

    async def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """Abstract process coroutine.

        Called by `internal_batch_process`. Needs to be overwritten in
        sub-classes.

        The function can either return the output example directly, or it can
        be a generator yielding a number of generated examples. This is handy
        for data augmentation or filtering tasks. An example is filtered out
        when the generator is empty. (i.e. `yield from []`).

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]|Generator[dict[str, Any], None, None]):
                processed example or generator over examples
        """
        ...

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """Abstract process method.

        Called by `internal_batch_process`. Needs to be overwritten in
        sub-classes.

        The function can either return the output example directly, or it can
        be a generator yielding a number of generated examples. This is handy
        for data augmentation or filtering tasks. An example is filtered out
        when the generator is empty. (i.e. `yield from []`).

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]|Generator[dict[str, Any], None, None]):
                processed example or generator over examples
        """
        raise NotImplementedError

    @abstractmethod
    def map_features(self, features: Features) -> Features:
        """Map dataset features.

        Map input features to *new* features. This specifies the exact
        output of the `process` function.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features):
                new dataset features. Note that this is not the output feature
                map but only the features generated by the data processor
        """
        ...
