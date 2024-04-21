from itertools import chain

import numpy as np
import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.statistics.sequence.seq_val_hist import (
    SequenceValueHistogram,
    SequenceValueHistogramConfig,
)
from tests.hyped.data.processors.statistics.base import BaseTestDataStatistic


class TestSequenceValueHistogram(BaseTestDataStatistic):
    @pytest.fixture
    def in_features(self):
        return Features({"A": Sequence(Value("float64"))})

    @pytest.fixture(
        params=[
            list(range(10)),
            [0, 0, 0, 1, 1, 1, 3, 3, 3],
            np.random.randint(-10, 10, size=100).tolist(),
            np.random.randint(-10, 10, size=5000).tolist(),
        ]
    )
    def in_batch(self, request):
        return {"A": [request.param] * 2}

    @pytest.fixture
    def statistic(self, in_batch):
        return SequenceValueHistogram(
            SequenceValueHistogramConfig(
                statistic_key="A.histogram",
                feature_key="A",
                low=min(chain.from_iterable(in_batch["A"])),
                high=max(chain.from_iterable(in_batch["A"])),
                num_bins=(
                    max(chain.from_iterable(in_batch["A"]))
                    - min(chain.from_iterable(in_batch["A"]))
                    + 1
                ),
            )
        )

    @pytest.fixture
    def expected_init_value(self, statistic):
        return [0] * statistic.config.num_bins

    @pytest.fixture
    def expected_stat_value(self, in_batch, statistic):
        vals = np.asarray(in_batch["A"]).flatten() - int(statistic.config.low)
        return np.bincount(vals, minlength=statistic.config.num_bins).tolist()
