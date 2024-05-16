from contextlib import nullcontext

import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_checks import check_feature_equals
from hyped.common.feature_ref import FeatureKey, FeatureRef, FeatureCollection


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


class TestFeatureRef(object):

    @pytest.mark.parametrize(
        "ref", [
            FeatureRef(
                key=FeatureKey(),
                feature=Value("string"),
                node_id=-1
            ),
            FeatureRef(
                key=FeatureKey(),
                feature=Value("int32"),
                node_id=-1
            ),
            FeatureRef(
                key=FeatureKey(),
                feature=Sequence(Value("int32")),
                node_id=-1
            ),
            FeatureRef(
                key=FeatureKey(),
                feature=Sequence(Value("int32"), length=3),
                node_id=-1
            ),
            FeatureRef(
                key=FeatureKey(),
                feature=Features({"x": Value("int32")}),
                node_id=-1
            ),
            FeatureRef(
                key=FeatureKey(),
                feature=Features(
                    {
                        "x": Value("int32"),
                        "y": Sequence(Value("string")),
                        "z": Sequence(Value("string"), length=3),
                        "a": {
                            "a.x": Value("int32"),
                            "a.y": Sequence(Value("string")),
                            "a.z": Sequence(Value("string"), length=3),
                        }
                    }
                ),
                node_id=-1
            ),
        ]
    )
    def test_serialize(self, ref):
        # serialize and deserialize feature reference
        rec_ref = FeatureRef.model_validate(ref.model_dump())
        rec_ref_json = FeatureRef.model_validate_json(ref.model_dump_json())
        # check reconstructed reference
        assert rec_ref == ref
        assert rec_ref_json == ref

    def test_basics(self):
        
        ref = FeatureRef(
            key=FeatureKey(),
            feature=Features(
                {
                    "x": Value("int32"),
                    "y": Sequence(Value("string")),
                    "z": Sequence(Value("string"), length=3),
                    "a": {
                        "a_x": Value("int32"),
                    }
                }
            ),
            node_id=-1
        )

        # invalid sub-feature key
        with pytest.raises(KeyError):
            ref.invalid
        with pytest.raises(KeyError):
            ref["invalid"]

        # index out of range error
        with pytest.raises(IndexError):
            ref.z[4]

        # basic sub-feature access checks
        assert ref["x"] == ref.x == FeatureRef(
            key=FeatureKey("x"), feature=Value("int32"), node_id=-1
        )
        assert ref["y"] == ref.y == FeatureRef(
            key=FeatureKey("y"), feature=Sequence(Value("string")), node_id=-1
        )
        assert ref.y[:] == FeatureRef(
            key=FeatureKey("y", slice(None)), feature=Sequence(Value("string")), node_id=-1
        )
        assert ref.y[0] == FeatureRef(
            key=FeatureKey("y", 0), feature=Value("string"), node_id=-1
        )
        assert ref.z[0] == FeatureRef(
            key=FeatureKey("z", 0), feature=Value("string"), node_id=-1
        )
        assert ref["a"]["a_x"] == ref["a", "a_x"] == ref.a.a_x == FeatureRef(
            key=FeatureKey("a", "a_x"), feature=Value("int32"), node_id=-1
        )

    def test_hash(self) -> int:

        assert (
            hash(FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0))
            == hash(FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0))
        )
        assert (
            hash(FeatureRef(key=FeatureKey("B"), feature=Value("string"), node_id=0))
            != hash(FeatureRef(key=FeatureKey("A"), feature=Value("string"), node_id=0))
        )
        assert (
            hash(FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0))
            != hash(FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=1))
        )
        assert (
            hash(FeatureRef(key=FeatureKey("B"), feature=Value("string"), node_id=0))
            != hash(FeatureRef(key=FeatureKey("A"), feature=Value("string"), node_id=1))
        )
        # hash only considers the index to the feature and not the feature type
        assert (
            hash(FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0))
            == hash(FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0))
        )


class TestFeatureCollection(object):

    def test_basics(self):

        # pack features of same type in sequence
        FeatureCollection(
            collection=[
                FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
            ]
        )

        with pytest.raises(TypeError):
            # cannot pack features of different types in sequence
            FeatureCollection(
                collection=[
                    FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
                    FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                ]
            )

        # construct either via collection argument or keyword arguments
        A = FeatureCollection(
            collection={
                "a": FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
                "b": FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
            }
        )
        B = FeatureCollection(
            a=FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
            b=FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
        )

        assert A == B
        
        with pytest.raises(TypeError):
            # cannot provide both collection and keyword arguments
            FeatureCollection(collection={}, a={})
    
    @pytest.mark.parametrize(
        "collection,feature", [
            (FeatureCollection(), Features()),
            (
                FeatureCollection(
                    a=FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
                    b=FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                ),
                Features(
                    {"a": Value("string"), "b": Value("int32")}
                )
            ),
            (
                FeatureCollection(
                    collection=[
                        FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                        FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                    ]
                ),
                Sequence(Value("int32"), length=2)
            )
        ]
    )
    def test_feature(self, collection, feature):
        assert check_feature_equals(collection.feature, feature)


    @pytest.mark.parametrize(
        "collection,refs", [
            (FeatureCollection(), set()),
            (
                FeatureCollection(
                    a=FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
                    b=FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                ),
                (
                    FeatureRef(key=FeatureKey(), feature=Value("string"), node_id=0),
                    FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                )
            ),
            (
                FeatureCollection(
                    collection=[
                        FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                        FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                    ]
                ),
                (
                    FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                    FeatureRef(key=FeatureKey(), feature=Value("int32"), node_id=0),
                )
            )
        ]
    )
    def test_refs(self, collection, refs):
        assert collection.refs == set(refs)
