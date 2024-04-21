from collections import Counter
from itertools import chain

import numpy as np
import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.processors.statistics.sequence.disc_seq_val_hist import (
    DiscreteSequenceValueHistogram,
    DiscreteSequenceValueHistogramConfig,
)
from tests.hyped.data.processors.statistics.base import BaseTestDataStatistic


class TestDiscreteHistogramWithStrings(BaseTestDataStatistic):
    @pytest.fixture(params=[Value, ClassLabel])
    def in_features(self, request):
        if request.param is Value:
            return Features({"A": Sequence(Value("string"))})
        if request.param is ClassLabel:
            return Features(
                {
                    "A": Sequence(
                        ClassLabel(names=list(map(str, range(-10, 10))))
                    )
                }
            )

    @pytest.fixture(
        params=[
            list(range(10)),
            [0, 0, 0, 1, 1, 1, 3, 3, 3],
            np.random.randint(-10, 10, size=100).tolist(),
            np.random.randint(-10, 10, size=5000).tolist(),
        ]
    )
    def in_batch(self, request, in_features):
        if isinstance(in_features["A"].feature, Value):
            return {"A": [list(map(str, request.param))] * 3}
        elif isinstance(in_features["A"].feature, ClassLabel):
            return {"A": [in_features["A"].feature.str2int(request.param)] * 3}

    @pytest.fixture
    def statistic(self, in_batch):
        return DiscreteSequenceValueHistogram(
            DiscreteSequenceValueHistogramConfig(
                statistic_key="A.histogram",
                feature_key="A",
            )
        )

    @pytest.fixture
    def expected_init_value(self, statistic):
        return {}

    @pytest.fixture
    def expected_stat_value(self, in_features, in_batch, statistic):
        if isinstance(in_features["A"].feature, Value):
            return dict(Counter(chain.from_iterable(in_batch["A"])))
        elif isinstance(in_features["A"].feature, ClassLabel):
            return dict(
                Counter(
                    chain.from_iterable(
                        map(in_features["A"].feature.int2str, in_batch["A"])
                    )
                )
            )
