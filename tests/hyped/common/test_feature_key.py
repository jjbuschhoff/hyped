import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_checks import check_feature_equals
from hyped.common.feature_key import FeatureKey


class TestFeatureKey(object):
    def test_basics(self):
        with pytest.raises(
            ValueError, match="First entry of a feature key must be a string"
        ):
            FeatureKey(1)

        # test basics on single entry key
        key = FeatureKey("key")
        assert isinstance(key, FeatureKey)
        assert len(key) == 1
        assert isinstance(key[0], str) and (key[0] == "key")

        # test basics multi-entry key
        key = FeatureKey("key", 1, slice(5))
        assert len(key) == 3
        assert isinstance(key[0], str)
        assert isinstance(key[1], int)
        assert isinstance(key[2], slice)
        # test slicing
        assert isinstance(key[:1], FeatureKey)
        assert isinstance(key[1:], tuple) and not isinstance(
            key[1:], FeatureKey
        )
        # test string representations of feature key
        str(key)
        repr(key)

    @pytest.mark.parametrize(
        "key,features,feature",
        [
            (
                FeatureKey("key"),
                Features({"key": Value("int32")}),
                Value("int32"),
            ),
            (
                FeatureKey("A", "B"),
                Features({"A": {"B": Value("int32")}}),
                Value("int32"),
            ),
            (
                FeatureKey("A", 0),
                Features({"A": Sequence(Value("int32"))}),
                Value("int32"),
            ),
            (
                FeatureKey("A", 1),
                Features({"A": Sequence(Value("int32"))}),
                Value("int32"),
            ),
            (
                FeatureKey("A", slice(None)),
                Features({"A": Sequence(Value("int32"))}),
                Sequence(Value("int32")),
            ),
            (
                FeatureKey("A", slice(None)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=10),
            ),
            (
                FeatureKey("A", slice(5)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=5),
            ),
            (
                FeatureKey("A", slice(-3)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=7),
            ),
            (
                FeatureKey("A", slice(2, 8, 2)),
                Features({"A": Sequence(Value("int32"), length=10)}),
                Sequence(Value("int32"), length=3),
            ),
        ],
    )
    def test_index_features(self, key, features, feature):
        assert check_feature_equals(key.index_features(features), feature)

    @pytest.mark.parametrize(
        "key,features,exc_type",
        [
            (FeatureKey("key"), Features({"X": Value("int32")}), KeyError),
            (
                FeatureKey("A", "B"),
                Features({"A": {"X": Value("int32")}}),
                KeyError,
            ),
            (
                FeatureKey("A", 1),
                Features({"A": Sequence(Value("int32"), length=1)}),
                IndexError,
            ),
        ],
    )
    def test_errors_on_index_features(self, key, features, exc_type):
        with pytest.raises(exc_type):
            key.index_features(features)

    @pytest.mark.parametrize(
        "key,example,value",
        [
            (FeatureKey("key"), {"key": 5}, 5),
            (FeatureKey("A", "B"), {"A": {"B": 5}}, 5),
            (
                FeatureKey("A", slice(None)),
                {"A": list(range(10))},
                list(range(10)),
            ),
            (
                FeatureKey("A", slice(5)),
                {"A": list(range(10))},
                list(range(5)),
            ),
            (
                FeatureKey("A", slice(3, 8)),
                {"A": list(range(10))},
                list(range(3, 8)),
            ),
            (
                FeatureKey("A", slice(3, 8, 2)),
                {"A": list(range(10))},
                list(range(3, 8, 2)),
            ),
        ],
    )
    def test_index_example(self, key, example, value):
        assert key.index_example(example) == value

    @pytest.mark.parametrize(
        "key,batch,values",
        [
            (FeatureKey("key"), {"key": [5]}, [5]),
            (FeatureKey("A", "B"), {"A": [{"B": 5}]}, [5]),
            (FeatureKey("A"), {"A": list(range(10))}, list(range(10))),
            (
                FeatureKey("A", "B"),
                {"A": [{"B": i} for i in range(10)]},
                list(range(10)),
            ),
            (
                FeatureKey("A", 3),
                {"A": [list(range(5)) for i in range(10)]},
                [3 for _ in range(10)],
            ),
            (
                FeatureKey("A", slice(2, 4)),
                {"A": [list(range(5)) for i in range(10)]},
                [list(range(2, 4)) for _ in range(10)],
            ),
        ],
    )
    def test_index_batch(self, key, batch, values):
        assert key.index_batch(batch) == values
