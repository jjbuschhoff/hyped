import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import ExecutionState
from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.base import IOContext
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
    pi = p._in_refs_validator.validate(i)
    ai = p._in_refs_validator.validate(i)
    # build output features
    po = p._out_refs_type.build_features(p.config, pi)
    ao = a._out_refs_type.build_features(a.config, ai)
    # add nodes
    proc_node = graph.add_processor_node(p, pi, po)
    agg_node = graph.add_processor_node(a, ai, ao)

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
    # create data flow
    flow = DataFlow(Features({"x": Value("int64")}))
    flow._graph = graph
    # return setup
    return flow, graph, const_node, proc_node, agg_node


@pytest.fixture
def io_contexts(setup_graph):
    graph, const_node, proc_node, agg_node = setup_graph

    return [
        IOContext(
            node_id=proc_node,
            inputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.IN_FEATURES
            ],
            outputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
        ),
        IOContext(
            node_id=agg_node,
            inputs=graph.nodes[agg_node][
                DataFlowGraph.NodeAttribute.IN_FEATURES
            ],
            outputs=graph.nodes[agg_node][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
        ),
    ]
