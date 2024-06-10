from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value

from hyped.data.flow import ops
from hyped.data.flow.aggregators.ops.mean import MeanAggregator
from hyped.data.flow.aggregators.ops.sum import SumAggregator
from hyped.data.flow.flow import DataFlow, DataFlowGraph
from hyped.data.flow.processors.ops import binary
from hyped.data.flow.processors.ops.collect import CollectFeatures
from hyped.data.flow.refs.ref import FeatureRef


def test_binary_op_constant_inputs_handler():
    mock_binary_op = MagicMock()
    # wrap mock binary operator
    wrapped_binary_op = ops._handle_constant_inputs_for_binary_op(
        mock_binary_op
    )
    # create a feature reference instance
    ref = FeatureRef(
        node_id_=-1, key_=tuple(), flow_=None, feature_=Value("int32")
    )

    # expected error on only constant inputs
    with pytest.raises(RuntimeError):
        wrapped_binary_op(0, 0)

    # called with only references
    wrapped_binary_op(ref, ref)
    mock_binary_op.assert_called_with(ref, ref)

    # called with mixture of reference and constants
    with patch("hyped.data.flow.ops.collect") as mock_collect:
        # first constant then reference
        wrapped_binary_op(0, ref)
        mock_collect.assert_called_with({"0": 0}, flow=ref.flow_)
        mock_binary_op.assert_called_with(mock_collect()["0"], ref)
        # first reference then constant
        wrapped_binary_op(ref, 0)
        mock_collect.assert_called_with({"0": 0}, flow=ref.flow_)
        mock_binary_op.assert_called_with(ref, mock_collect()["0"])


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
        flow._graph.nodes[out.node_id_][DataFlowGraph.NodeProperty.PROCESSOR],
        CollectFeatures,
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.x.node_id_, out.node_id_)
    assert flow._graph.has_edge(flow.src_features.y.node_id_, out.node_id_)


@pytest.mark.parametrize(
    "op, agg_type", [(ops.sum_, SumAggregator), (ops.mean, MeanAggregator)]
)
def test_simple_aggregators(op, agg_type):
    flow = DataFlow(Features({"a": Value("int32")}))

    # run operator
    out = op(flow.src_features.a)
    # make sure the node for the binary operation has been added
    assert out.node_id_ in flow._graph
    assert isinstance(
        flow._graph.nodes[out.node_id_][DataFlowGraph.NodeProperty.PROCESSOR],
        agg_type,
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.a.node_id_, out.node_id_)


@pytest.mark.parametrize(
    "op, proc_type, dtype",
    [
        (ops.add, binary.Add, "int32"),
        (ops.sub, binary.Sub, "int32"),
        (ops.mul, binary.Mul, "int32"),
        (ops.pow, binary.Pow, "int32"),
        (ops.mod, binary.Mod, "int32"),
        (ops.truediv, binary.TrueDiv, "int32"),
        (ops.floordiv, binary.FloorDiv, "int32"),
        (ops.eq, binary.Equals, "int32"),
        (ops.ne, binary.NotEquals, "int32"),
        (ops.lt, binary.LessThan, "int32"),
        (ops.le, binary.LessThanOrEqual, "int32"),
        (ops.gt, binary.GreaterThan, "int32"),
        (ops.ge, binary.GreaterThanOrEqual, "int32"),
        (ops.and_, binary.LogicalAnd, "bool"),
        (ops.or_, binary.LogicalOr, "bool"),
        (ops.xor_, binary.LogicalXOr, "bool"),
    ],
)
def test_simple_binary_op(op, proc_type, dtype):
    flow = DataFlow(
        Features(
            {
                "a": Value(dtype),
                "b": Value(dtype),
            }
        )
    )

    # run operator
    out = op(flow.src_features.a, flow.src_features.b)
    # make sure the node for the binary operation has been added
    assert out.node_id_ in flow._graph
    assert isinstance(
        flow._graph.nodes[out.node_id_][DataFlowGraph.NodeProperty.PROCESSOR],
        proc_type,
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.a.node_id_, out.node_id_)
    assert flow._graph.has_edge(flow.src_features.b.node_id_, out.node_id_)
