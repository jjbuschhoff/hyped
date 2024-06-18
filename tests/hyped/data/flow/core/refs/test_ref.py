import operator
from unittest.mock import patch

import pytest
from datasets import Features, Sequence, Value

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.core.refs.ref import FeatureRef


class TestFeatureRef(object):
    @pytest.mark.parametrize(
        "ref",
        [
            FeatureRef(
                key_=FeatureKey(),
                feature_=Value("string"),
                node_id_="",
                flow_=None,
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Value("int32"),
                node_id_="",
                flow_=None,
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Sequence(Value("int32")),
                node_id_="",
                flow_=None,
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Sequence(Value("int32"), length=3),
                node_id_="",
                flow_=None,
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Features({"x": Value("int32")}),
                node_id_="",
                flow_=None,
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Features(
                    {
                        "x": Value("int32"),
                        "y": Sequence(Value("string")),
                        "z": Sequence(Value("string"), length=3),
                        "a": {
                            "a.x": Value("int32"),
                            "a.y": Sequence(Value("string")),
                            "a.z": Sequence(Value("string"), length=3),
                        },
                    }
                ),
                node_id_="",
                flow_=None,
            ),
        ],
    )
    def test_serialize(self, ref):
        # serialize and deserialize feature reference
        rec_ref = FeatureRef.model_validate(ref.model_dump())
        rec_ref_json = FeatureRef.model_validate_json(ref.model_dump_json())
        # check reconstructed reference
        assert rec_ref.model_dump() == ref.model_dump()
        assert rec_ref_json.model_dump() == ref.model_dump()

    def test_indexing(self):
        ref = FeatureRef(
            key_=FeatureKey(),
            feature_=Features(
                {
                    "x": Value("int32"),
                    "y": Sequence(Value("string")),
                    "z": Sequence(Value("string"), length=3),
                    "a": {
                        "a_x": Value("int32"),
                    },
                }
            ),
            node_id_="",
            flow_=None,
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
        assert (
            ref["x"].model_dump()
            == ref.x.model_dump()
            == FeatureRef(
                key_=FeatureKey("x"),
                feature_=Value("int32"),
                node_id_="",
                flow_=None,
            ).model_dump()
        )
        assert (
            ref["y"].model_dump()
            == ref.y.model_dump()
            == FeatureRef(
                key_=FeatureKey("y"),
                feature_=Sequence(Value("string")),
                node_id_="",
                flow_=None,
            ).model_dump()
        )
        assert (
            ref.y[:].model_dump()
            == FeatureRef(
                key_=FeatureKey("y", slice(None)),
                feature_=Sequence(Value("string")),
                node_id_="",
                flow_=None,
            ).model_dump()
        )
        assert (
            ref.y[0].model_dump()
            == FeatureRef(
                key_=FeatureKey("y", 0),
                feature_=Value("string"),
                node_id_="",
                flow_=None,
            ).model_dump()
        )
        assert (
            ref.z[0].model_dump()
            == FeatureRef(
                key_=FeatureKey("z", 0),
                feature_=Value("string"),
                node_id_="",
                flow_=None,
            ).model_dump()
        )

        assert (
            ref["a"]["a_x"].model_dump()
            == ref["a", "a_x"].model_dump()
            == ref.a.a_x.model_dump()
            == FeatureRef(
                key_=FeatureKey("a", "a_x"),
                feature_=Value("int32"),
                node_id_="",
                flow_=None,
            ).model_dump()
        )

    def test_ptr_and_hash(self):
        # create references
        refA = FeatureRef(
            key_=FeatureKey(),
            feature_=Value("int32"),
            node_id_="",
            flow_=None,
        )
        refB = FeatureRef(
            key_=FeatureKey(),
            feature_=Value("float32"),
            node_id_="",
            flow_=None,
        )
        refC = FeatureRef(
            key_=FeatureKey("x"),
            feature_=Value("int32"),
            node_id_="",
            flow_=None,
        )
        refD = FeatureRef(
            key_=FeatureKey(),
            feature_=Value("int32"),
            node_id_="1",
            flow_=None,
        )
        refE = FeatureRef(
            key_=FeatureKey(),
            feature_=Value("int32"),
            node_id_="",
            flow_=object(),
        )
        refF = FeatureRef(
            key_=FeatureKey("x"),
            feature_=Value("int32"),
            node_id_="1",
            flow_=object(),
        )

        # feature type is irrelevant for pointer and hash
        assert refA.ptr == refB.ptr
        assert hash(refA) == hash(refA)
        assert hash(refB) == hash(refB)
        assert hash(refA) == hash(refB)

        # pointers and hashes should mismatch
        for other in [refC, refD, refE, refF]:
            assert refA.ptr != other.ptr
            assert hash(refA) != hash(other)


@pytest.mark.parametrize(
    "op, op_fn, dtype",
    [
        (operator.add, "hyped.data.flow.ops.add", "int32"),
        (operator.sub, "hyped.data.flow.ops.sub", "int32"),
        (operator.mul, "hyped.data.flow.ops.mul", "int32"),
        (operator.truediv, "hyped.data.flow.ops.truediv", "int32"),
        (operator.pow, "hyped.data.flow.ops.pow", "int32"),
        (operator.mod, "hyped.data.flow.ops.mod", "int32"),
        (operator.floordiv, "hyped.data.flow.ops.floordiv", "int32"),
        (operator.eq, "hyped.data.flow.ops.eq", "int32"),
        (operator.ne, "hyped.data.flow.ops.ne", "int32"),
        (operator.lt, "hyped.data.flow.ops.lt", "int32"),
        (operator.le, "hyped.data.flow.ops.le", "int32"),
        (operator.gt, "hyped.data.flow.ops.gt", "int32"),
        (operator.ge, "hyped.data.flow.ops.ge", "int32"),
        (operator.and_, "hyped.data.flow.ops.and_", "bool"),
        (operator.or_, "hyped.data.flow.ops.or_", "bool"),
        (operator.xor, "hyped.data.flow.ops.xor_", "bool"),
    ],
)
def test_binary_ops(op, op_fn, dtype):
    # create feature refs
    refA = FeatureRef(
        key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
    )
    refB = FeatureRef(
        key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
    )
    # patch operator function
    with patch(op_fn) as mock:
        # apply operator
        op(refA, refB)
        # make sure the operator function was called correctly
        mock.assert_called_with(refA, refB)


@pytest.mark.parametrize(
    "agg, agg_fn, dtype",
    [
        (FeatureRef.sum_, "hyped.data.flow.ops.sum_", "int32"),
        (FeatureRef.mean_, "hyped.data.flow.ops.mean", "int32"),
    ],
)
def test_aggregator_ops(agg, agg_fn, dtype):
    # create a feature reference
    ref = FeatureRef(
        key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
    )
    # patch aggregator function
    with patch(agg_fn) as mock:
        # apply operator
        agg(ref)
        # make sure the operator function was called correctly
        mock.assert_called_with(ref)
