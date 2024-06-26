import operator
from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from pydantic import BaseModel

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
        with pytest.raises(AttributeError):
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

    def test_get_item(self):
        seq_ref = FeatureRef(
            key_=FeatureKey("seq"),
            feature_=Sequence(Value("int32")),
            node_id_="",
            flow_=object(),
        )
        idx_ref = FeatureRef(
            key_=FeatureKey("idx"),
            feature_=Value("int32"),
            node_id_="1",
            flow_=object(),
        )

        with patch("hyped.data.flow.ops.get_item") as mock:
            # Apply operator
            seq_ref[idx_ref]
            # Make sure the operator function was called correctly
            mock.assert_called_with(seq_ref, idx_ref)

        with pytest.raises(TypeError):
            # error on invalid feature type
            idx_ref[idx_ref]

    def test_sequence_set_item(self):
        seq_ref = FeatureRef(
            key_=FeatureKey("seq"),
            feature_=Sequence(Value("int32")),
            node_id_="",
            flow_=object(),
        )
        idx_ref = FeatureRef(
            key_=FeatureKey("idx"),
            feature_=Value("int32"),
            node_id_="1",
            flow_=object(),
        )
        val_ref = FeatureRef(
            key_=FeatureKey("val"),
            feature_=Value("int32"),
            node_id_="1",
            flow_=object(),
        )

        with patch("hyped.data.flow.ops.set_item") as mock:
            # Apply operator
            seq_ref[idx_ref] = val_ref
            # Make sure the operator function was called correctly
            mock.assert_called_with(seq_ref, idx_ref, val_ref)

        with pytest.raises(TypeError):
            # error on invalid feature type
            val_ref[idx_ref] = val_ref

    def test_mapping_set_item(self):
        src_ref = FeatureRef(
            key_=FeatureKey("seq"),
            feature_=Features({"a": Value("int32"), "b": Value("int32")}),
            node_id_="",
            flow_=object(),
        )
        val_ref = FeatureRef(
            key_=FeatureKey("val"),
            feature_=Value("int32"),
            node_id_="1",
            flow_=object(),
        )

        # patch the eq operator of the feature ref to avoid
        # using the overloaded operator in assert_called_with
        with patch("hyped.data.flow.ops.collect") as mock, patch(
            "hyped.data.flow.core.refs.ref.FeatureRef.__eq__", BaseModel.__eq__
        ):
            # Apply operator
            src_ref.model_copy()["new"] = val_ref

            mock.assert_called_with(
                {"a": src_ref.a, "b": src_ref.b, "new": val_ref}
            )

    @pytest.mark.parametrize(
        "op, op_fn, dtype",
        [
            (operator.neg, "hyped.data.flow.ops.neg", "int32"),
            (operator.abs, "hyped.data.flow.ops.abs_", "int32"),
            (operator.invert, "hyped.data.flow.ops.invert", "int32"),
            (len, "hyped.data.flow.ops.len_", "string"),
        ],
    )
    def test_unary_ops(self, op, op_fn, dtype):
        # Create feature ref
        ref = FeatureRef(
            key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
        )
        # Patch operator function
        with patch(op_fn) as mock:
            # Apply operator
            op(ref)
            # Make sure the operator function was called correctly
            mock.assert_called_with(ref)

    @pytest.mark.parametrize(
        "op, op_fn, dtype",
        [
            # math operators
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
            # concat strings
            (operator.add, "hyped.data.flow.ops.concat", "string"),
            (operator.concat, "hyped.data.flow.ops.concat", "string"),
            # in operator for sequences and string-likes
            (operator.contains, "hyped.data.flow.ops.contains", "string"),
        ],
    )
    def test_binary_ops(self, op, op_fn, dtype):
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
        "op, op_fn, dtype",
        [
            # math operators
            (operator.add, "hyped.data.flow.ops.add", "int32"),
            (operator.sub, "hyped.data.flow.ops.sub", "int32"),
            (operator.mul, "hyped.data.flow.ops.mul", "int32"),
            (operator.truediv, "hyped.data.flow.ops.truediv", "int32"),
            (operator.pow, "hyped.data.flow.ops.pow", "int32"),
            (operator.mod, "hyped.data.flow.ops.mod", "int32"),
            (operator.floordiv, "hyped.data.flow.ops.floordiv", "int32"),
            (operator.and_, "hyped.data.flow.ops.and_", "bool"),
            (operator.or_, "hyped.data.flow.ops.or_", "bool"),
            (operator.xor, "hyped.data.flow.ops.xor_", "bool"),
            # concat strings
            (operator.add, "hyped.data.flow.ops.concat", "string"),
        ],
    )
    def test_reflected_binary_ops(self, op, op_fn, dtype):
        # create feature ref
        ref = FeatureRef(
            key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
        )
        # create mock value for binary operation
        val = object()
        # patch operator function
        with patch(op_fn) as mock:
            # apply reflected operator
            op(val, ref)
            # make sure the reflected operator function was called correctly
            mock.assert_called_with(val, ref)

    @pytest.mark.parametrize(
        "iop, op_fn, dtype",
        [
            # math operators
            (operator.iadd, "hyped.data.flow.ops.add", "int32"),
            (operator.isub, "hyped.data.flow.ops.sub", "int32"),
            (operator.imul, "hyped.data.flow.ops.mul", "int32"),
            (operator.itruediv, "hyped.data.flow.ops.truediv", "int32"),
            (operator.ipow, "hyped.data.flow.ops.pow", "int32"),
            (operator.imod, "hyped.data.flow.ops.mod", "int32"),
            (operator.ifloordiv, "hyped.data.flow.ops.floordiv", "int32"),
            (operator.iand, "hyped.data.flow.ops.and_", "bool"),
            (operator.ior, "hyped.data.flow.ops.or_", "bool"),
            (operator.ixor, "hyped.data.flow.ops.xor_", "bool"),
            # concat strings
            (operator.iadd, "hyped.data.flow.ops.concat", "string"),
            (operator.iconcat, "hyped.data.flow.ops.concat", "string"),
        ],
    )
    def test_inplace_binary_ops(self, iop, op_fn, dtype):
        # create feature refs
        refA = FeatureRef(
            key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
        )
        refB = FeatureRef(
            key_=FeatureKey(), node_id_="", flow_=None, feature_=Value(dtype)
        )

        # inplace operators change the first argument in execution
        # we copy the first argument before for the actual operator call
        mock = MagicMock()
        patched_op_fn = lambda a, b: mock(a.model_copy(), b)
        # patch operator function and since we copied the first argument
        # the objects will be compared using the eq operator in assert_called_with
        # so we have to patch that to not use the overloaded operator here
        with patch(op_fn, patched_op_fn), patch(
            "hyped.data.flow.core.refs.ref.FeatureRef.__eq__", BaseModel.__eq__
        ):
            # apply in-place operator
            iop(refA.model_copy(), refB)
            # make sure the operator function was called correctly
            mock.assert_called_with(refA, refB)

    @pytest.mark.parametrize(
        "agg, agg_fn, dtype",
        [
            (FeatureRef.sum_, "hyped.data.flow.ops.sum_", "int32"),
            (FeatureRef.mean_, "hyped.data.flow.ops.mean", "int32"),
        ],
    )
    def test_aggregator_ops(self, agg, agg_fn, dtype):
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
