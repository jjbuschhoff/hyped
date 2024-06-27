from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from pydantic import BaseModel

from hyped.data.flow import ops
from hyped.data.flow.aggregators.ops.mean import MeanAggregator
from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops import binary, sequence, unary
from hyped.data.flow.processors.ops.collect import CollectFeatures


@pytest.fixture(autouse=True)
def patch_feature_ref_eq_op():
    # patches the feature ref type to use the standard comparator
    # operators instead of the overwritten ones
    # this is required for the mock.assert_called_with checks which
    # make use of the == opeartor to compare the objects

    class PatchedFeatureRef(FeatureRef):
        __eq__ = BaseModel.__eq__
        __ne__ = BaseModel.__ne__

    with (
        patch("hyped.data.flow.core.refs.ref.FeatureRef", PatchedFeatureRef),
        patch("hyped.data.flow.core.graph.FeatureRef", PatchedFeatureRef),
        patch("hyped.data.flow.core.flow.FeatureRef", PatchedFeatureRef),
    ):
        yield


def test_binary_op_constant_inputs_handler():
    mock_flow = MagicMock()
    mock_binary_op = MagicMock()
    # wrap mock binary operator
    wrapped_binary_op = ops._handle_constant_inputs_for_binary_op(
        mock_binary_op
    )
    # create a feature reference instance
    ref = FeatureRef(
        node_id_="", key_=tuple(), flow_=mock_flow, feature_=Value("int32")
    )

    # expected error on only constant inputs
    with pytest.raises(RuntimeError):
        wrapped_binary_op(0, 0)

    # called with only references
    wrapped_binary_op(ref, ref)
    mock_binary_op.assert_called_with(ref, ref)

    with patch("hyped.data.flow.ops.Const") as mock_const:
        # first constant then reference
        wrapped_binary_op(0, ref)
        mock_const.assert_called_with(value=0)
        mock_const(value=0).to.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(
            mock_const(value=0).to(mock_flow).value, ref
        )
        # first reference then constant
        wrapped_binary_op(ref, 1)
        mock_const.assert_called_with(value=1)
        mock_const(value=1).to.assert_called_with(mock_flow)
        mock_binary_op.assert_called_with(
            ref, mock_const(value=1).to(mock_flow).value
        )


def test_collect():
    flow = DataFlow(
        Features(
            {
                "x": Value("string"),
                "y": Value("string"),
            }
        )
    )

    out = ops.collect([flow.src_features.x, flow.src_features.y])
    # make sure the output feature is correct
    assert out.feature_ == Sequence(Value("string"), length=2)
    # make sure the node has been added
    assert out.node_id_ in flow._graph
    assert isinstance(
        flow._graph.nodes[out.node_id_][DataFlowGraph.NodeAttribute.NODE_OBJ],
        CollectFeatures,
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.x.node_id_, out.node_id_)
    assert flow._graph.has_edge(flow.src_features.y.node_id_, out.node_id_)


@pytest.mark.parametrize(
    "op, agg_type",
    [
        (ops.sum_, "hyped.data.flow.ops.SumAggregator"),
        (ops.mean, "hyped.data.flow.ops.MeanAggregator"),
    ],
)
def test_simple_aggregators(op, agg_type):
    flow = DataFlow(Features({"a": Value("int32")}))

    with patch(agg_type) as mock:
        # run operator
        op(flow.src_features.a)
        # make sure the processor was called with the correct inputs
        mock().call.assert_called_once_with(x=flow.src_features.a)


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (ops.add, "hyped.data.flow.processors.ops.binary.Add", "int32"),
        (ops.sub, "hyped.data.flow.processors.ops.binary.Sub", "int32"),
        (ops.mul, "hyped.data.flow.processors.ops.binary.Mul", "int32"),
        (ops.pow, "hyped.data.flow.processors.ops.binary.Pow", "int32"),
        (ops.mod, "hyped.data.flow.processors.ops.binary.Mod", "int32"),
        (
            ops.truediv,
            "hyped.data.flow.processors.ops.binary.TrueDiv",
            "int32",
        ),
        (
            ops.floordiv,
            "hyped.data.flow.processors.ops.binary.FloorDiv",
            "int32",
        ),
        (ops.eq, "hyped.data.flow.processors.ops.binary.Equals", "int32"),
        (ops.ne, "hyped.data.flow.processors.ops.binary.NotEquals", "int32"),
        (ops.lt, "hyped.data.flow.processors.ops.binary.LessThan", "int32"),
        (
            ops.le,
            "hyped.data.flow.processors.ops.binary.LessThanOrEqual",
            "int32",
        ),
        (ops.gt, "hyped.data.flow.processors.ops.binary.GreaterThan", "int32"),
        (
            ops.ge,
            "hyped.data.flow.processors.ops.binary.GreaterThanOrEqual",
            "int32",
        ),
        (ops.and_, "hyped.data.flow.processors.ops.binary.LogicalAnd", "bool"),
        (ops.or_, "hyped.data.flow.processors.ops.binary.LogicalOr", "bool"),
        (ops.xor_, "hyped.data.flow.processors.ops.binary.LogicalXOr", "bool"),
    ],
)
def test_binary_op(op, proc_type, dtype):
    flow = DataFlow(
        Features(
            {
                "a": Value(dtype),
                "b": Value(dtype),
            }
        )
    )

    with patch(proc_type) as mock:
        # run operator
        op(flow.src_features.a, flow.src_features.b)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(
            a=flow.src_features.a, b=flow.src_features.b
        )


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (ops.neg, "hyped.data.flow.processors.ops.unary.Neg", "int32"),
        (ops.abs_, "hyped.data.flow.processors.ops.unary.Abs", "int32"),
        (ops.invert, "hyped.data.flow.processors.ops.unary.Invert", "int32"),
    ],
)
def test_unary_op(op, proc_type, dtype):
    flow = DataFlow(
        Features(
            {
                "a": Value(dtype),
            }
        )
    )

    with patch(proc_type) as mock:
        # run operator
        op(flow.src_features.a)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(
            a=flow.src_features.a,
        )


def test_len_op():
    flow = DataFlow(
        Features(
            {
                "constant_seq": Sequence(Value("int32"), length=5),
                "dynamic_seq": Sequence(Value("int32")),
                "str": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with patch(
        "hyped.data.flow.processors.ops.sequence.SequenceLength"
    ) as mock:
        # test constant length sequence
        out = flow.src_features.constant_seq.length_()
        # make sure the processor was not called and check the output
        assert not mock().call.called
        assert out == 5

    with patch(
        "hyped.data.flow.processors.ops.sequence.SequenceLength"
    ) as mock:
        # test dynamic length sequence
        out = flow.src_features.dynamic_seq.length_()
        # make sure processor was called correctly
        mock().call.assert_called_once_with(a=flow.src_features.dynamic_seq)

    with pytest.raises(NotImplementedError):
        # TODO: string features are not supported yet
        flow.src_features.str.length_()

    with pytest.raises(TypeError):
        # test with invalid feature
        flow.src_features.inv.length_()


def test_concat_op():
    flow = DataFlow(
        Features(
            {
                "seqA": Sequence(Value("int32")),
                "seqB": Sequence(Value("int32")),
                "strA": Value("string"),
                "strB": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with patch(
        "hyped.data.flow.processors.ops.sequence.SequenceConcat"
    ) as mock:
        ops.concat(flow.src_features.seqA, flow.src_features.seqB)
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            a=flow.src_features.seqA, b=flow.src_features.seqB
        )

    with pytest.raises(NotImplementedError):
        # TODO: string features are not supported yet
        ops.concat(flow.src_features.strA, flow.src_features.strB)

    with pytest.raises(TypeError):
        # test with invalid feature
        ops.concat(flow.src_features.inv, flow.src_features.inv)


def test_sequence_get_set_item():
    flow = DataFlow(
        Features(
            {
                "seq": Sequence(Value("int32")),
                "idx": Value("int32"),
                "val": Value("int32"),
            }
        )
    )

    with patch(
        "hyped.data.flow.processors.ops.sequence.SequenceGetItem"
    ) as mock:
        ops.get_item(flow.src_features.seq, flow.src_features.idx)
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq, index=flow.src_features.idx
        )

    with patch(
        "hyped.data.flow.processors.ops.sequence.SequenceSetItem"
    ) as mock:
        ops.set_item(
            flow.src_features.seq, flow.src_features.idx, flow.src_features.val
        )
        # make sure processor was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq,
            index=flow.src_features.idx,
            value=flow.src_features.val,
        )


@pytest.mark.parametrize(
    "op, seq_proc_type",
    [
        (
            FeatureRef.contains_,
            "hyped.data.flow.processors.ops.sequence.SequenceContains",
        ),
        (
            ops.count_of,
            "hyped.data.flow.processors.ops.sequence.SequenceCountOf",
        ),
        (
            ops.index_of,
            "hyped.data.flow.processors.ops.sequence.SequenceIndexOf",
        ),
    ],
)
def test_value_lookup_op(op, seq_proc_type):
    flow = DataFlow(
        Features(
            {
                "seq": Sequence(Value("string")),
                "str": Value("string"),
                "val": Value("string"),
                "inv": Value("int32"),
            }
        )
    )

    with patch(seq_proc_type) as mock:
        # run operator
        op(flow.src_features.seq, flow.src_features.val)
        # make sure the operator was called correctly
        mock().call.assert_called_once_with(
            sequence=flow.src_features.seq,
            value=flow.src_features.val,
        )

    with pytest.raises(NotImplementedError):
        # TODO: string features are not supported yet
        op(flow.src_features.str, flow.src_features.val)

    with pytest.raises(TypeError):
        # test with invalid feature
        op(flow.src_features.inv, flow.src_features.val)


@pytest.mark.parametrize(
    "op, proc_type",
    [(ops.zip_, "hyped.data.flow.processors.ops.sequence.SequenceZip")],
)
def test_multi_sequence_op(op, proc_type):
    flow = DataFlow(
        Features(
            {
                "a": Sequence(Value("string")),
                "b": Sequence(Value("string")),
                "c": Sequence(Value("string")),
            }
        )
    )

    with (
        patch(proc_type) as proc_mock,
        patch("hyped.data.flow.ops.collect") as collect_mock,
    ):
        # call the operator
        op(
            flow.src_features.a,
            flow.src_features.b,
            flow.src_features.c,
        )
        # make sure the features are collected before
        collect_mock.assert_called_once_with(
            [
                flow.src_features.a,
                flow.src_features.b,
                flow.src_features.c,
            ]
        )
        # and the processor is called on the collected features
        proc_mock().call.assert_called_once_with(sequences=collect_mock())
