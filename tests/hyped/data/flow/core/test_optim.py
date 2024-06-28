import networkx as nx
import pytest
from datasets import Features, Value

from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.const import Const
from hyped.data.flow.core.optim import DataFlowGraphOptimizer
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.collect import (
    CollectFeatures,
    NestedContainer,
)

from .mock import MockInputRefs, MockProcessor, mock_input_refs_validator


def new_graph():
    # create graph
    graph = DataFlowGraph()
    # add source node
    src_features = Features({"x": Value("int64")})
    src_node_id = graph.add_source_node(src_features)
    # return graph and source node id
    return graph, src_node_id


def add_processor(graph, node_A, node_B, **kwargs):
    # create processor
    p = MockProcessor(**kwargs)
    i = MockInputRefs(
        a=graph.get_node_output_ref(node_A),
        b=graph.get_node_output_ref(node_B),
    )
    i = mock_input_refs_validator.validate(i)
    o = p._out_refs_type.build_features(p.config, i)
    # add new processor to graph
    return graph.add_processor_node(p, i, o)


def cse_test_cases():
    test_cases = []

    # create trivial case
    graph, src_node_id = new_graph()
    add_processor(graph, src_node_id, src_node_id)
    test_cases.append((graph, graph))

    # create simple graph
    graph, src_node_id = new_graph()
    add_processor(graph, src_node_id, src_node_id)
    add_processor(graph, src_node_id, src_node_id)
    # create target to simple graph
    target, src_node_id = new_graph()
    add_processor(target, src_node_id, src_node_id)
    # add test case
    test_cases.append((graph, target))

    # create slightly more complex graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    add_processor(graph, node_id_1, node_id_2)
    # create target to graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id)
    add_processor(target, node_id_1, node_id_1)
    # add test case
    test_cases.append((graph, target))

    # create graph with no redundant nodes
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(graph, src_node_id, src_node_id, i=1)
    # create target graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(target, src_node_id, src_node_id, i=1)
    # add test case
    test_cases.append((graph, target))

    # create graph with no two branches including redundant nodes
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(graph, src_node_id, node_id_1)
    node_id_2 = add_processor(graph, src_node_id, node_id_1)
    node_id_3 = add_processor(graph, src_node_id, src_node_id, i=1)
    # create target graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id, i=0)
    node_id_2 = add_processor(target, src_node_id, node_id_1)
    node_id_3 = add_processor(target, src_node_id, src_node_id, i=1)
    # add test case
    test_cases.append((graph, target))

    # test constant nodes
    graph, _ = new_graph()
    Const(value=0).to(graph)
    Const(value=0).to(graph)
    Const(value=1).to(graph)
    # create target graph
    target, _ = new_graph()
    Const(value=0).to(target)
    Const(value=1).to(target)
    # add test case
    test_cases.append((graph, target))

    # test collect nodes
    graph, _ = new_graph()
    ref_1 = Const(value=0).to(graph)
    ref_2 = Const(value=1).to(graph)
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    # create target graph
    target, _ = new_graph()
    ref_1 = Const(value=0).to(target)
    ref_2 = Const(value=1).to(target)
    CollectFeatures().call(
        collection=NestedContainer[FeatureRef](data={"a": ref_1, "b": ref_2})
    )
    # add test case
    test_cases.append((graph, target))

    return test_cases


def constant_evaluation_test_cases():
    test_cases = []

    graph, src_node_id = new_graph()
    # add processors
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)

    # no constants to evaluate
    test_cases.append((graph, graph))

    graph, src_node_id = new_graph()
    # add a constant
    const_node_id_1 = Const(value=5).to(graph).node_id_
    # add processors
    node_id_1 = add_processor(graph, src_node_id, const_node_id_1)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)

    # there is a constant but nothing to optimize
    test_cases.append((graph, graph))

    graph, src_node_id = new_graph()
    # add a constant
    const_node_id_1 = Const(value=5).to(graph).node_id_
    const_node_id_2 = Const(value=5).to(graph).node_id_
    # add processors
    node_id_1 = add_processor(graph, const_node_id_1, const_node_id_2)
    node_id_2 = add_processor(graph, const_node_id_1, const_node_id_2)
    add_processor(graph, src_node_id, node_id_1)
    add_processor(graph, src_node_id, node_id_2)

    target, src_node_id = new_graph()
    # both processor nodes should be evaluated to constant nodes
    const_node_id_1 = Const(value={"y": 0}).to(target).node_id_
    const_node_id_2 = Const(value={"y": 0}).to(target).node_id_
    add_processor(target, src_node_id, const_node_id_1)
    add_processor(target, src_node_id, const_node_id_2)

    test_cases.append((graph, target))

    graph, src_node_id = new_graph()
    # add a constant
    const_node_id_1 = Const(value=5).to(graph).node_id_
    const_node_id_2 = Const(value=5).to(graph).node_id_
    # add processors
    node_id_1 = add_processor(graph, const_node_id_1, const_node_id_2)
    node_id_2 = add_processor(graph, const_node_id_1, const_node_id_2)
    node_id_3 = add_processor(graph, node_id_2, node_id_1)
    node_id_4 = add_processor(graph, src_node_id, src_node_id)
    add_processor(graph, src_node_id, node_id_3)
    add_processor(graph, node_id_4, const_node_id_1)

    target, src_node_id = new_graph()
    # both processor nodes should be evaluated to constant nodes
    const_node_id_1 = (
        Const(value={"y": 0}).to(target).node_id_
    )  # original node_id_3
    const_node_id_2 = (
        Const(value={"value": 5}).to(target).node_id_
    )  # original const_node_id_1
    node_id_4 = add_processor(target, src_node_id, src_node_id)
    add_processor(target, src_node_id, const_node_id_1)
    add_processor(target, node_id_4, const_node_id_2)

    test_cases.append((graph, target))

    return test_cases


def optimize_test_cases():
    test_cases = []

    # create trivial case
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    test_cases.append((graph, graph, node_id_1))

    # create simple graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    # create target to simple graph
    target, src_node_id = new_graph()
    node_id_3 = add_processor(target, src_node_id, src_node_id)
    # add test case
    test_cases.append((graph, target, node_id_1))
    test_cases.append((graph, target, node_id_2))

    # create simple graph
    graph, src_node_id = new_graph()
    node_id_1 = add_processor(graph, src_node_id, src_node_id)
    node_id_2 = add_processor(graph, src_node_id, src_node_id)
    node_id_3 = add_processor(graph, src_node_id, src_node_id)
    node_id_4 = add_processor(graph, node_id_1, node_id_2)
    # create target to simple graph
    target, src_node_id = new_graph()
    node_id_1 = add_processor(target, src_node_id, src_node_id)
    node_id_2 = add_processor(target, node_id_1, node_id_1)
    # add test case
    test_cases.append((graph, target, node_id_4))

    return test_cases


def node_match(n1, n2):
    node_type_1 = n1[DataFlowGraph.NodeAttribute.NODE_TYPE]
    node_type_2 = n2[DataFlowGraph.NodeAttribute.NODE_TYPE]

    if node_type_1 != node_type_2:
        return False

    if node_type_1 == DataFlowGraph.NodeType.CONST:
        node_obj_1 = n1[DataFlowGraph.NodeAttribute.NODE_OBJ]
        node_obj_2 = n2[DataFlowGraph.NodeAttribute.NODE_OBJ]
        return node_obj_1.config.value == node_obj_2.config.value

    return True


class TestOptimizer:
    @pytest.mark.parametrize("graph, target", cse_test_cases())
    def test_cse(self, graph, target):
        # apply cse
        optim = DataFlowGraphOptimizer()
        cse_graph = optim.cse(graph)
        # check topology of cse graph
        assert nx.is_isomorphic(cse_graph, target, node_match=node_match)

    @pytest.mark.parametrize("graph, target", constant_evaluation_test_cases())
    def test_optimizer_constant_evaluation(self, graph, target):
        # apply constant evaluation
        optim = DataFlowGraphOptimizer()
        optim_graph = optim.constant_evaluation(graph)
        # check topology of the optimized graph
        assert nx.is_isomorphic(optim_graph, target, node_match=node_match)

    @pytest.mark.parametrize("graph, target, leaf_node", optimize_test_cases())
    def test_optimize(self, graph, target, leaf_node):
        # apply cse
        optim = DataFlowGraphOptimizer()
        optim_graph = optim.optimize(graph, {leaf_node})
        # check topology of cse graph
        assert nx.is_isomorphic(optim_graph, target, node_match=node_match)
