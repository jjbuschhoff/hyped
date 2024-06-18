import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import ExecutionState
from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.const import Const

from .mock import MockAggregator, MockInputRefs, MockProcessor


@pytest.fixture(autouse=True)
def reset_mocks():
    MockProcessor.process.reset_mock()
    MockAggregator.initialize.reset_mock()
    MockAggregator.extract.reset_mock()
    MockAggregator.update.reset_mock()


@pytest.fixture
def setup_graph():
    # create graph
    graph = DataFlowGraph()
    print("INIT", id(graph))
    # add source node
    src_features = Features({"x": Value("int64")})
    src_node = graph.add_source_node(src_features)
    # add constant node
    c = Const(value=0)
    co = c._out_refs_type.build_features(c.config, None)
    const_node = graph.add_processor_node(c, None, co)

    # create nodes
    p = MockProcessor()
    a = MockAggregator()

    # create input refs from source features
    i = MockInputRefs(
        a=graph.get_node_output_ref(src_node).x,
        b=graph.get_node_output_ref(const_node).value,
    )
    # build output features
    po = p._out_refs_type.build_features(p.config, i)
    # add nodes
    proc_node = graph.add_processor_node(p, i, po)
    agg_node = graph.add_processor_node(a, i, None)

    return graph, const_node, proc_node, agg_node


@pytest.fixture
def setup_state(setup_graph):
    graph, const_node, proc_node, agg_node = setup_graph
    # create state
    batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
    state = ExecutionState(graph, batch, index, rank)
    # return setup
    return state, graph, const_node, proc_node, agg_node


@pytest.fixture
def setup_flow(setup_graph):
    graph, const_node, proc_node, agg_node = setup_graph
    print("FLOW", id(graph))
    # create data flow
    flow = DataFlow(Features({"x": Value("int64")}))
    flow._graph = graph
    print(id(flow._graph))
    # return setup
    return flow, graph, const_node, proc_node, agg_node
