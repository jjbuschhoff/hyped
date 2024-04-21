from collections import Counter

import numpy as np
import pytest
from datasets import ClassLabel, Features, Value

from hyped.data.processors.statistics.value.disc_hist import (
    DiscreteHistogram,
    DiscreteHistogramConfig,
)
from tests.hyped.data.processors.statistics.base import BaseTestDataStatistic


class TestDiscreteHistogramWithStrings(BaseTestDataStatistic):
    @pytest.fixture(params=[Value, ClassLabel])
    def in_features(self, request):
        if request.param is Value:
            return Features({"A": Value("string")})
        if request.param is ClassLabel:
            return Features(
                {"A": ClassLabel(names=list(map(str, range(-10, 10))))}
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
        if isinstance(in_features["A"], Value):
            return {"A": list(map(str, request.param))}
        elif isinstance(in_features["A"], ClassLabel):
            return {"A": in_features["A"].str2int(request.param)}

    @pytest.fixture
    def statistic(self, in_batch):
        return DiscreteHistogram(
            DiscreteHistogramConfig(
                statistic_key="A.histogram",
                feature_key="A",
            )
        )

    @pytest.fixture
    def expected_init_value(self, statistic):
        return {}

    @pytest.fixture
    def expected_stat_value(self, in_features, in_batch, statistic):
        if isinstance(in_features["A"], Value):
            return dict(Counter(in_batch["A"]))
        elif isinstance(in_features["A"], ClassLabel):
            return dict(Counter(in_features["A"].int2str(in_batch["A"])))
