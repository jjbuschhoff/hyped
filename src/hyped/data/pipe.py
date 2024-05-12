"""Data Pipe."""
from __future__ import annotations

import warnings
from collections import deque
from copy import deepcopy
from itertools import chain
from typing import Any, Iterable, Literal

import datasets
import pyarrow as pa
from pydantic import Field
from torch.utils.data import get_worker_info
from typing_extensions import TypeAlias

from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.common.feature_checks import check_feature_equals
from hyped.common.feature_key import FeatureDict, FeatureKey
from hyped.data.processors.statistics.base import BaseDataStatistic
from hyped.data.processors.statistics.report import statistics_report_manager

from .processors.base import BaseDataProcessor, BaseDataProcessorConfig

DatasetType: TypeAlias = (
    datasets.Dataset
    | datasets.DatasetDict
    | datasets.IterableDataset
    | datasets.IterableDatasetDict
)


class DataPipeConfig(BaseDataProcessorConfig):
    """Data Pipeline Configuration."""

    keep_input_features: Literal[True] = Field(default=True, init_var=False)
    output_format: Literal[None] = Field(default=None, init_var=False)


class DataPipe(list, BaseDataProcessor[DataPipeConfig]):
    """Data Pipe.

    A Data Pipe is a sequence of data processors. It provides useful
    functionality such as passing a batch of examples through the
    sequence or pipe.
    """

    def __init__(self, processors: list[BaseDataProcessor] = []) -> None:
        """Initialize a Data Pipe.

        Arguments:
            processors (list[BaseDataProcessor]): the initial pipe of processors
        """
        # check types of processors
        if not all(isinstance(p, BaseDataProcessor) for p in processors):
            raise TypeError(
                "All processors used in a data pipe must inherit from `%s`"
                % BaseDataProcessor
            )
        # initialize pipe as list of processors
        list.__init__(self, processors)
        BaseDataProcessor.__init__(self, config=DataPipeConfig())

    @classmethod
    def from_config(cls, config: DataPipeConfig) -> DataPipe:
        """Instantiate data pipe from the given config.

        Arguments:
            config (DataPipeConfig):
                data pipe configuration

        Returns:
            pipe (DataPipe):
                data pipeline
        """
        raise NotImplementedError()

    @property
    def is_prepared(self) -> bool:
        """Check if the data pipe is prepared and ready for execution.

        This also verifies the feature pipe, i.e. checks that the output
        of any processor matches the input of the following one.
        """
        return (
            # check all processors of the pipe
            all(p.is_prepared for p in self)
            and (self._in_features is not None)
            and (
                (len(self) == 0)
                or (
                    check_feature_equals(
                        self[0].in_features, self._in_features
                    )
                    and all(
                        check_feature_equals(p1.out_features, p2.in_features)
                        for p1, p2 in zip(self[:-1], self[1:])
                    )
                )
            )
        )

    def map_features(self, features: datasets.Features) -> datasets.Features:
        """Prepare all data processors of the data pipe for execution.

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features of the output of the processor
        """
        # prepare all processors
        for p in self:
            features = p.prepare(features)
        # return final output features
        return features

    @property
    def required_feature_keys(self) -> list[FeatureKey]:
        """List of all required feature keys."""
        if not self.is_prepared:
            raise RuntimeError("Data pipe not prepared")

        required_keys = set()
        new_features = datasets.Features()

        def _feature_exists(
            key: FeatureKey, features: datasets.Features
        ) -> bool:
            """Check whether a feature at a key exists."""
            try:
                key.index_features(features)
                return True
            except (KeyError, IndexError):
                return False

        for proc in self:
            # update required feature keys with all feature keys that are not
            # present in the features up to this point
            required_keys.update(
                {
                    k
                    for k in proc.required_feature_keys
                    if not _feature_exists(k, new_features)
                }
            )
            # keep track of features created within the data pipe
            new_features = new_features | proc.new_features

        return required_keys

    @property
    def raw_features(self) -> datasets.Features:
        """Raw dataset features generated by data pipe."""
        # aggregate all new features created through the pipe
        # TODO: this only accumulates all generated features
        #       but does not handle the removal of features
        #       throughout the pipeline by for example a filter
        #       features processor
        features = datasets.Features()
        for p in self:
            features.update(p.new_features)

        return features

    @property
    def new_features(self) -> datasets.Features:
        """New dataset features generated by data pipe."""
        return self.raw_features

    @property
    def out_features(self) -> datasets.Features:
        """All output dataset features."""
        return self.in_features if len(self) == 0 else self[-1].out_features

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> dict[str, list[Any]] | tuple[dict[str, list[Any]], list[int]]:
        """Process a batch of examples.

        Arguments:
            examples (dict[str, list[Any]]):
                batch of examples to process
            index (list[int]):
                dataset indices of the examples
            rank (int):
                execution process rank
            return_index (bool):
                whether to return the source index for each output example

        Returns:
            out_batch (dict[str, list[Any]]):
                processed examples
            index (list[int]):
                the source indices to each example. Only returned when
                `return_index` is set to true.
        """
        iterable = self.iter_batch_process(
            examples=examples,
            index=index,
            rank=rank,
            return_index=return_index,
        )
        # return the last item of the iterable which corresponds
        # to the output of the last data processor
        return deque(iterable, maxlen=1).pop()

    def iter_batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> Iterable[
        dict[str, list[Any]] | tuple[dict[str, list[Any]], list[int]]
    ]:
        """Apply each data processor to the batch of examples.

        Yields the output of each data processor when ready.

        Arguments:
            examples (dict[str, list[Any]]):
                batch of examples to process
            index (list[int]):
                dataset indices of the examples
            rank (int):
                execution process rank
            return_index (bool):
                whether to return the source index for each output example

        Returns:
            out_iter (
                Iterable[dict[str, list[Any]]
                | tuple[dict[str, list[Any]], list[int]]]
            ):
                iterator over output batch from each processor, includes
                the source indices when `return_index=True`
        """
        if rank is None:
            # try to get multiprocessing rank from pytorch worker info
            worker_info = get_worker_info()
            rank = None if worker_info is None else worker_info.id

        # make sure the pipeline is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data Pipe is not prepared. This happens either when the "
                "`prepare` function was not called, or when a processor "
                "of the pipe is re-prepared with different features."
            )
        # apply each processor
        for p in self:
            examples, index = p.batch_process(
                examples, index, rank, return_index=True
            )
            # yield the output of the current data processor
            yield (examples, index) if return_index else examples

    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        """Internal Batch Process."""
        raise NotImplementedError()

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """Process a single example."""
        raise NotImplementedError()

    def _batch_process_to_pyarrow(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
    ) -> pa.Table:
        # convert to pyarrow table with correct schema
        return pa.table(
            data=self.batch_process(examples, index, rank),
            schema=convert_features_to_arrow_schema(self.out_features),
        )

    def apply(
        self,
        data: DatasetType,
        **kwargs,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Apply the data pipe to a dataset.

        Arguments:
            data (DatasetType):
                source dataset(s)
            **kwargs (dict[str, Any]):
                arguments forwarded to datasets `.map` function

        Returns:
            out (DatasetType):
                processed dataset(s)
        """
        # get the dataset features
        if isinstance(data, (datasets.Dataset, datasets.IterableDataset)):
            features = data.features
        elif isinstance(
            data, (datasets.DatasetDict, datasets.IterableDatasetDict)
        ):
            features = next(iter(data.values())).features
        else:
            raise ValueError(
                "Expected one of `datasets.Dataset`, `datasets.DatasetDict`, "
                "`datasets.IterableDataset` or `datasets.IterableDatasetDict`,"  # noqa: E501
                "got %s" % type(data)
            )

        if features is not None:
            # prepare the data pipe for the dataset
            self.prepare(features)

        elif not self.is_prepared:
            raise RuntimeError(
                "Dataset features unknown, please manually prepare the data "
                "pipe by calling the `.prepare` function with appropriate "
                "features."
            )

        # check if the data pipe contains and statistics that are expected
        # to be computed while running the data pipeline
        if (
            isinstance(data, (datasets.Dataset, datasets.DatasetDict))
            and any(isinstance(p, BaseDataStatistic) for p in self)
            and not statistics_report_manager.is_empty()
        ):
            # load from cache file defaults to false
            kwargs["load_from_cache_file"] = kwargs.get(
                "load_from_cache_file", False
            )
            # warn it dataset is loaded from cache
            if kwargs["load_from_cache_file"]:
                warnings.warn(
                    "Loading map result from cache file will not compute "
                    "statistics, set `load_from_cache_file=False` to avoid "
                    "this behavior.",
                    UserWarning,
                )

        # apply data pipe to dataset
        data = self.internal_apply(data, **kwargs)

        if isinstance(
            data, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # set output features for lazy datasets manually
            if isinstance(data, datasets.IterableDataset):
                data.info.features = self.out_features
            elif isinstance(data, datasets.IterableDatasetDict):
                for split in data.values():
                    split.info.features = self.out_features

        return data

    def internal_apply(
        self,
        data: DatasetType,
        **kwargs,
    ) -> DatasetType:
        """Internal apply function.

        Arguments:
            data (DatasetType):
                source dataset(s)
            **kwargs:
                keyword arguments passed to the map function
                appropriate for the given dataset type

        Returns:
            out (DatasetType):
                processed dataset(s)
        """
        # required settings
        kwargs["batched"] = True
        kwargs["with_indices"] = True
        # for non-iterable datasets the map function provide the rank
        if isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
            kwargs["with_rank"] = True

        if isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
            # use pyarrow table as output format for in-memory
            # datasets that support caching since it includes
            # the output feature information
            return data.map(self._batch_process_to_pyarrow, **kwargs)

        elif isinstance(
            data, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            return data.map(
                self.batch_process,
                remove_columns=set(self.in_features.keys())
                - set(self.out_features.keys()),
                **kwargs,
            )

        raise ValueError("Unexpected Dataset type, got %s" % data)
