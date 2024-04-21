import numpy as np
import pytest
from datasets import Features, Sequence, Value

from hyped.data.processors.statistics.sequence.seq_len_hist import (
    SequenceLengthHistogram,
    SequenceLengthHistogramConfig,
)
from tests.hyped.data.processors.statistics.base import BaseTestDataStatistic


class TestSequenceLengthHistogram(BaseTestDataStatistic):
    @pytest.fixture
    def in_features(self):
        return Features({"A": Sequence(Value("float64"))})

    @pytest.fixture(
        params=[
            [1, 2, 3, 4],
            [1, 1, 1, 1],
            np.random.randint(0, 100, size=100).tolist(),
            np.random.randint(0, 100, size=500).tolist(),
        ]
    )
    def lengths(self, request):
        return request.param

    @pytest.fixture
    def in_batch(self, lengths):
        return {"A": [[0] * l for l in lengths]}

    @pytest.fixture
    def statistic(self, in_batch):
        return SequenceLengthHistogram(
            SequenceLengthHistogramConfig(
                statistic_key="A.histogram", feature_key="A", max_length=100
            )
        )

    @pytest.fixture
    def expected_init_value(self, statistic):
        return [0] * 101

    @pytest.fixture
    def expected_stat_value(self, in_batch, statistic):
        vals = np.asarray(list(map(len, in_batch["A"])))
        return np.bincount(vals, minlength=statistic.config.num_bins).tolist()
