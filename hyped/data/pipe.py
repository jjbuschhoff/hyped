import warnings
from collections import deque
from copy import deepcopy
from typing import Any, Iterable

import datasets
import pyarrow as pa
from torch.utils.data import get_worker_info

from hyped.data.processors.statistics.base import BaseDataStatistic
from hyped.data.processors.statistics.report import statistics_report_manager
from hyped.utils.arrow import convert_features_to_arrow_schema

from .processors.base import BaseDataProcessor


class DataPipe(list):
    """Data Pipe

    A Data Pipe is a sequence of data processors. It provides useful
    functionality such as passing a batch of examples through the
    sequence or pipe.

    Arguments:
        processors (list[BaseDataProcessor]): the initial pipe of processors
    """

    def __init__(self, processors: list[BaseDataProcessor] = []) -> None:
        # check types of processors
        if not all(isinstance(p, BaseDataProcessor) for p in processors):
            raise TypeError(
                "All processors used in a data pipe must inherit from `%s`"
                % BaseDataProcessor
            )
        # initialize pipe as list of processors
        list.__init__(self, processors)

        # save input features
        # usually the input features can be inferred from the first
        # data processor in the pipeline, however this doesn't work
        # in case the pipeline is empty
        self._in_features: datasets.Features = None

    def prepare(self, features: datasets.Features) -> datasets.Features:
        """Prepare all data processors of the data pipe for execution

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features of the output of the processor
        """
        # save a copy of the input features
        self._in_features = deepcopy(features)
        # prepare all processors
        for p in self:
            features = p.prepare(features)
        # return final output features
        return self.out_features

    @property
    def is_prepared(self) -> bool:
        """Check if the data pipe is prepared and ready for execution
        This also verifies the feature pipe, i.e. checks that the output
        of any processor matches the input of the following one.
        """
        return (
            # check all processors of the pipe
            all(p.is_prepared for p in self)
            and all(
                p1.out_features == p2.in_features
                for p1, p2 in zip(self[:-1], self[1:])
            )
        )

    @property
    def in_features(self) -> datasets.Features:
        """Input dataset features available to data pipe"""
        return self[0].in_features if len(self) > 0 else self._in_features

    @property
    def new_features(self) -> datasets.Features:
        """New dataset features generated by data pipe"""
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
    def out_features(self) -> datasets.Features:
        """All output features of the processor. Includes both input
        features and new features generated by the data pipe. On conflicts,
        the new features are prioritized.
        """
        return self[-1].out_features if len(self) > 0 else self.in_features

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
    ) -> dict[str, list[Any]]:
        """Process a batch of examples

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out (dict[str, list[Any]]): processed examples
        """
        iterable = self.iter_batch_process(
            examples=examples, index=index, rank=rank
        )
        # return the last item of the iterable which corresponds
        # to the output of the last data processor
        return deque(iterable, maxlen=1).pop()

    def iter_batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
    ) -> Iterable[dict[str, list[Any]]]:
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
            yield examples

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
        data: (
            datasets.Dataset
            | datasets.DatasetDict
            | datasets.IterableDataset
            | datasets.IterableDatasetDict
        ),
        **kwargs,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Apply the data pipe to a dataset

        Arguments:
            data (Dataset|DatasetDict|IterableDataset|IterableDatasetDict):
                source dataset(s)
            **kwargs (dict[str, Any]):
                arguments forwarded to datasets `.map` function

        Returns:
            out (datasets.Dataset|datasets.DatasetDict): processed dataset(s)
        """

        # TODO: test this preparation logic
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

        # TODO: test this behavior
        # check if the data pipe contains and statistics that are expected
        # to be computed while running the data pipeline
        if (
            isinstance(data, (datasets.Dataset, datasets.DatasetDict))
            and any(isinstance(p, BaseDataStatistic) for p in self)
            and not statistics_report_manager.is_empty
        ):
            # load from cache file defaults to false
            kwargs["load_from_cache_file"] = kwargs.get(
                "load_from_cache_file", False
            )
            # warn it dataset is loaded from cache
            if kwargs["load_from_cache_file"]:
                warnings.warn(
                    "Loading map result from cache file will not compute "
                    "statistics, set `load_from_cache_file` to False to avoid "
                    "this behavior.",
                    UserWarning,
                )

        # required settings
        kwargs["batched"] = True
        kwargs["with_indices"] = True
        # for in-memory datasets let the map function provide the rank
        if isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
            kwargs["with_rank"] = True

        if isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
            # use pyarrow table as output format for in-memory
            # datasets that support caching since it includes
            # the output feature information
            data = data.map(self._batch_process_to_pyarrow, **kwargs)

        elif isinstance(
            data, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            data = data.map(
                self.batch_process,
                remove_columns=set(self.in_features.keys())
                - set(self.out_features.keys()),
                **kwargs,
            )
            # set output features for lazy datasets manually
            if isinstance(data, datasets.IterableDataset):
                data.info.features = self.out_features
            elif isinstance(data, datasets.IterableDatasetDict):
                for split in data.values():
                    split.info.features = self.out_features

        return data
