import asyncio
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from typing import Any

import pyarrow as pa
import pytest
from datasets import Dataset, Features, Value

from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor


class BaseTestDataProcessor(ABC):
    @pytest.fixture
    @abstractmethod
    def in_features(self, request) -> Features:
        ...

    @pytest.fixture
    @abstractmethod
    def processor(self, request) -> BaseDataProcessor:
        ...

    @pytest.fixture
    def in_batch(self, request) -> None | dict[str, list[Any]]:
        return None

    @pytest.fixture
    def expected_out_features(self, request) -> None | Features:
        return None

    @pytest.fixture
    def expected_out_batch(self, request) -> None | dict[str, list[Any]]:
        return None

    @pytest.fixture
    def expected_err_on_prepare(self) -> None | type[Exception]:
        return None

    @pytest.fixture
    def expected_err_on_process(self) -> None | type[Exception]:
        return None

    @pytest.fixture
    def kwargs_for_post_prepare_checks(
        self, processor, in_features, expected_out_features
    ):
        return {
            "processor": processor,
            "in_features": in_features,
            "expected_out_features": expected_out_features,
        }

    @pytest.fixture
    def kwargs_for_post_process_checks(self, processor, expected_out_batch):
        return {
            "processor": processor,
            "expected_out_batch": expected_out_batch,
        }

    def post_prepare_checks(
        self,
        processor,
        in_features,
        expected_out_features,
    ):
        # processor should be prepared at this point
        assert processor.is_prepared

        # check new and output features
        if expected_out_features is not None:
            assert processor.new_features == expected_out_features

            if processor.config.keep_input_features:
                assert processor.out_features == (
                    in_features | expected_out_features
                )
            else:
                assert processor.out_features == expected_out_features

    def post_process_checks(
        self,
        index,
        in_batch,
        out_batch,
        processor,
        expected_out_batch,
    ):
        if expected_out_batch is not None:
            # check output content
            if processor.config.keep_input_features:
                # apply source index to input batch and update with
                # expected output batch to achieve full expected output
                prepared_batch = {
                    k: [v[i] for i in index] for k, v in in_batch.items()
                }
                assert out_batch == (prepared_batch | expected_out_batch)
            else:
                assert out_batch == expected_out_batch

    def err_handler(self, err_type) -> AbstractContextManager:
        return nullcontext() if err_type is None else pytest.raises(err_type)

    def test_case(
        self,
        processor,
        in_features,
        in_batch,
        expected_err_on_prepare,
        expected_err_on_process,
        kwargs_for_post_prepare_checks,
        kwargs_for_post_process_checks,
    ):
        # check types of objects generated by fixtures
        assert isinstance(in_features, Features)
        assert isinstance(processor, BaseDataProcessor)

        # prepare and check output features
        with self.err_handler(expected_err_on_prepare):
            processor.prepare(in_features)
        # we catched an expected error
        if expected_err_on_prepare is not None:
            return

        # run post preparation checks
        self.post_prepare_checks(**kwargs_for_post_prepare_checks)

        if in_batch is not None:
            # make sure the batch follows the feature mapping
            in_schema = convert_features_to_arrow_schema(in_features)
            table = pa.table(in_batch, schema=in_schema)
            # create batch index
            batch_size = len(table)
            index = list(range(batch_size))
            # apply processor
            with self.err_handler(expected_err_on_process):
                out_batch, index = processor.batch_process(
                    in_batch, index=index, rank=0, return_index=True
                )

            if expected_err_on_process is not None:
                return

            # make sure the output batch aligns with the output features
            out_schema = convert_features_to_arrow_schema(
                processor.out_features
            )
            pa.table(out_batch, schema=out_schema)

            # run post process checks
            self.post_process_checks(
                index, in_batch, out_batch, **kwargs_for_post_process_checks
            )

    @pytest.fixture
    def map_batch_size(self):
        return 1000

    def test_case_with_pipe(
        self,
        processor,
        in_features,
        in_batch,
        expected_err_on_prepare,
        expected_err_on_process,
        kwargs_for_post_process_checks,
        map_batch_size,
    ):
        if (
            (expected_err_on_prepare is not None)
            or (expected_err_on_process is not None)
            or (in_batch is None)
        ):
            pytest.skip(
                "Test is not expected to execute processor, "
                "no reason to run it from a data pipe"
            )

        in_batch = deepcopy(in_batch)
        in_features = deepcopy(in_features)
        # add index feature to input batch
        in_batch_size = len(next(iter(in_batch.values())))
        in_batch["__index__"] = list(range(in_batch_size))
        in_features["__index__"] = Value("int32")

        # check types of objects generated by fixtures
        assert isinstance(in_features, Features)
        assert isinstance(processor, BaseDataProcessor)
        # convert input batch to a dataset
        in_schema = convert_features_to_arrow_schema(in_features)
        ds = Dataset(pa.table(in_batch, schema=in_schema))

        # apply processor to dataset using data pipe
        # can only use one process for map in pytest session
        # as the session is multi-threaded and forking a
        # process might lead to deadlock in child
        out_batch = (
            DataPipe([processor])
            .apply(ds, batch_size=map_batch_size, num_proc=1)
            .to_dict()
        )

        if processor.config.keep_input_features:
            # pop index features from input and output batch
            in_batch.pop("__index__")
            index = out_batch.pop("__index__")
            # run post process checks
            self.post_process_checks(
                index, in_batch, out_batch, **kwargs_for_post_process_checks
            )
