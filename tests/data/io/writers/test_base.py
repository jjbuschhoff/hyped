import multiprocessing as mp
from itertools import product

import datasets
import pytest

from hyped.data.io.writers.base import BaseDatasetConsumer


class DummyDatasetConsumer(BaseDatasetConsumer):
    def __init__(self, **kwargs):
        super(DummyDatasetConsumer, self).__init__(**kwargs)
        # keep track of total examples processed by the consumer
        self.total_consumed_examples = mp.Value("i", 0)

    def consume_example(self, **kwargs):
        with self.total_consumed_examples.get_lock():
            self.total_consumed_examples.value += 1


class TestBaseDatasetConsumer(object):
    @pytest.mark.parametrize("num_samples", [1024, 2048, 10000])
    @pytest.mark.parametrize(
        "num_proc, num_shards", product([2, 4, 8, 16], repeat=2)
    )
    def test(self, num_samples, num_proc, num_shards):
        # create dummy dataset
        ds = datasets.Dataset.from_dict({"a": range(num_samples)})
        ds = ds.to_iterable_dataset(num_shards=num_shards)
        # consume dataset
        consumer = DummyDatasetConsumer(num_proc=num_proc)
        consumer.consume(ds)
        # make sure the full dataset was consumed
        assert consumer.total_consumed_examples.value == num_samples
