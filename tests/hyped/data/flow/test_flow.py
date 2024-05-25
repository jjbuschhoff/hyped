import asyncio
from unittest.mock import patch

import datasets
import matplotlib.pyplot as plt
import pytest
from datasets import Features, Value

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.flow import (
    SRC_NODE_ID,
    DataFlow,
    DataFlowExecutor,
    DataFlowGraph,
    ExecutionState,
)
from hyped.data.flow.processors.ops.noop import NoOp, NoOpInputRefs
from hyped.data.flow.refs.ref import FeatureRef

mock_features = Features({"y": Value("int32")})


class MockFeatureRef(FeatureRef):
    def __init__(self, node_id, flow):
        super().__init__(
            key_=tuple(),
            node_id_=node_id,
            flow_=flow,
            feature_=mock_features,
        )


# fixtures
@pytest.fixture
def setup_graph():
    # create the graph
    graph = DataFlowGraph()
    # add the source node
    src_node_id = graph.add_source_node(mock_features)
    src = MockFeatureRef(src_node_id, graph)
    # add a first processor to the graph
    node_id1 = graph.add_processor_node(
        NoOp(), NoOpInputRefs(x=src.y), mock_features
    )
    out1 = MockFeatureRef(node_id1, graph)
    # add a second processor to the graph
    node_id2 = graph.add_processor_node(
        NoOp(), NoOpInputRefs(x=out1.y), mock_features
    )
    out2 = MockFeatureRef(node_id2, graph)
    # return setup
    return graph, src, out1, out2


@pytest.fixture
def setup_executor(setup_graph):
    graph, src, out1, out2 = setup_graph
    executor = DataFlowExecutor(graph, out2)
    return executor, graph, src, out1, out2


@pytest.fixture
def setup_state(setup_graph):
    graph, src, out1, out2 = setup_graph
    batch = {"y": [-1, -2, -3]}
    index = [0, 1, 2]
    rank = 0
    state = ExecutionState(graph, batch, index, rank)
    return state, graph, src, out1, out2


@pytest.fixture
def setup_flow(setup_graph):
    graph, src, out1, out2 = setup_graph
    # create data flow
    flow = DataFlow(mock_features)
    flow._graph = graph
    return flow, src, out1, out2


class TestDataFlowGraph:
    def test_add_processor_node(self):
        # create the graph
        graph = DataFlowGraph()
        # add the source node
        src_node_id = graph.add_source_node(mock_features)
        src = MockFeatureRef(src_node_id, graph)
        # check source node
        assert SRC_NODE_ID in graph
        assert graph.nodes[SRC_NODE_ID][DataFlowGraph.NodeProperty.DEPTH] == 0
        assert (
            graph.nodes[SRC_NODE_ID][DataFlowGraph.NodeProperty.PROCESSOR]
            is None
        )
        assert (
            graph.nodes[SRC_NODE_ID][DataFlowGraph.NodeProperty.FEATURES]
            == mock_features
        )

        # add a first processor to the graph
        proc1, inputs1 = NoOp(), NoOpInputRefs(x=src.y)
        node_id1 = graph.add_processor_node(proc1, inputs1, mock_features)
        # check node
        assert node_id1 in graph
        assert graph.nodes[node_id1][DataFlowGraph.NodeProperty.DEPTH] == 1
        assert (
            graph.nodes[node_id1][DataFlowGraph.NodeProperty.PROCESSOR]
            == proc1
        )
        assert (
            graph.nodes[node_id1][DataFlowGraph.NodeProperty.FEATURES]
            == mock_features
        )
        # check edge
        assert graph.has_edge(SRC_NODE_ID, node_id1)
        for n, r in inputs1.named_refs.items():
            assert n in graph[SRC_NODE_ID][node_id1]
            assert (
                graph[SRC_NODE_ID][node_id1][n][DataFlowGraph.EdgeProperty.KEY]
                == r.key_
            )
        # create the output feature reference for node 1
        out1 = MockFeatureRef(node_id1, graph)

        # add a second processor to the graph
        proc2, inputs2 = NoOp(), NoOpInputRefs(x=out1.y)
        node_id2 = graph.add_processor_node(proc2, inputs2, mock_features)
        # check node
        assert node_id2 in graph
        assert graph.nodes[node_id2][DataFlowGraph.NodeProperty.DEPTH] == 2
        assert (
            graph.nodes[node_id2][DataFlowGraph.NodeProperty.PROCESSOR]
            == proc2
        )
        assert (
            graph.nodes[node_id2][DataFlowGraph.NodeProperty.FEATURES]
            == mock_features
        )
        # check edge
        assert graph.has_edge(node_id1, node_id2)
        for n, r in inputs2.named_refs.items():
            assert n in graph[node_id1][node_id2]
            assert (
                graph[node_id1][node_id2][n][DataFlowGraph.EdgeProperty.KEY]
                == r.key_
            )

    def test_node_depth(self):
        # create the graph and add the source node
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(mock_features)
        src = MockFeatureRef(src_node_id, graph)

        # add first level processors
        proc1, inputs1 = NoOp(), NoOpInputRefs(x=src.y)
        node_id1 = graph.add_processor_node(proc1, inputs1, mock_features)
        out1 = MockFeatureRef(node_id1, graph)

        proc2, inputs2 = NoOp(), NoOpInputRefs(x=src.y)
        node_id2 = graph.add_processor_node(proc2, inputs2, mock_features)
        out2 = MockFeatureRef(node_id2, graph)

        proc3, inputs3 = NoOp(), NoOpInputRefs(x=src.y)
        node_id3 = graph.add_processor_node(proc3, inputs3, mock_features)
        out3 = MockFeatureRef(node_id3, graph)

        # add second level processors
        proc4, inputs4 = NoOp(), NoOpInputRefs(x=out1.y)
        node_id4 = graph.add_processor_node(proc4, inputs4, mock_features)
        out4 = MockFeatureRef(node_id4, graph)

        proc5, inputs5 = NoOp(), NoOpInputRefs(x=out2.y)
        node_id5 = graph.add_processor_node(proc5, inputs5, mock_features)
        out5 = MockFeatureRef(node_id5, graph)

        proc6, inputs6 = NoOp(), NoOpInputRefs(x=out2.y)
        node_id6 = graph.add_processor_node(proc6, inputs6, mock_features)
        out6 = MockFeatureRef(node_id6, graph)

        # check depth property
        assert graph.nodes[src_node_id][DataFlowGraph.NodeProperty.DEPTH] == 0
        assert graph.nodes[node_id1][DataFlowGraph.NodeProperty.DEPTH] == 1
        assert graph.nodes[node_id2][DataFlowGraph.NodeProperty.DEPTH] == 1
        assert graph.nodes[node_id3][DataFlowGraph.NodeProperty.DEPTH] == 1
        assert graph.nodes[node_id4][DataFlowGraph.NodeProperty.DEPTH] == 2
        assert graph.nodes[node_id5][DataFlowGraph.NodeProperty.DEPTH] == 2
        assert graph.nodes[node_id6][DataFlowGraph.NodeProperty.DEPTH] == 2

        # add third level processor
        proc7, inputs7 = NoOp(), NoOpInputRefs(x=out5.y)
        node_id7 = graph.add_processor_node(proc7, inputs7, mock_features)

        # check depth property
        assert graph.nodes[node_id7][DataFlowGraph.NodeProperty.DEPTH] == 3

    def test_graph_depth(self):
        # create the graph and add the source node
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(mock_features)
        src = MockFeatureRef(src_node_id, graph)
        # add two processors
        node_id1 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=src.y), mock_features
        )
        out1 = MockFeatureRef(node_id1, graph)
        node_id2 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out1.y), mock_features
        )
        out2 = MockFeatureRef(node_id2, graph)

        # check depth
        assert graph.depth == 3

        # add another branch to increase depth
        node_id3 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out2.y), mock_features
        )
        out3 = MockFeatureRef(node_id3, graph)
        assert graph.depth == 4

        # add another node at the same depth level
        node_id4 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out2.y), mock_features
        )
        assert graph.depth == 4

    def test_graph_width(self):
        # create the graph and add the source node
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(mock_features)
        src = MockFeatureRef(src_node_id, graph)
        # add two processors
        node_id1 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=src.y), mock_features
        )
        out1 = MockFeatureRef(node_id1, graph)
        node_id2 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out1.y), mock_features
        )
        out2 = MockFeatureRef(node_id2, graph)

        # check width
        assert graph.width == 1

        # add another branch to increase width at depth 1
        node_id3 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out1.y), mock_features
        )
        out3 = MockFeatureRef(node_id3, graph)
        assert graph.width == 2

        # add another node at the same depth level as an existing node
        node_id4 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=src.y), mock_features
        )
        out4 = MockFeatureRef(node_id4, graph)
        assert graph.width == 2

        # add another node at a new depth level
        node_id5 = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=out2.y), mock_features
        )
        out5 = MockFeatureRef(node_id5, graph)
        assert graph.width == 2

    def test_add_processor_invalid_input(self):
        g1 = DataFlowGraph()
        g2 = DataFlowGraph()
        # add source nodes
        g1_src_node_id = g1.add_source_node(mock_features)
        g2_src_node_id = g2.add_source_node(mock_features)
        # create feature refs
        g1_src = MockFeatureRef(g1_src_node_id, g1)
        g2_src = MockFeatureRef(g1_src_node_id, g2)
        # add valid nodes
        g1.add_processor_node(NoOp(), NoOpInputRefs(x=g1_src.y), mock_features)
        g2.add_processor_node(NoOp(), NoOpInputRefs(x=g2_src.y), mock_features)
        # try add invalid node
        with pytest.raises(RuntimeError):
            g1.add_processor_node(
                NoOp(), NoOpInputRefs(x=g2_src.y), mock_features
            )

    def test_dependency_graph(self, setup_graph):
        graph, src, out1, out2 = setup_graph
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        subgraph = graph.dependency_graph(node_id2)
        assert set(subgraph.nodes) == {SRC_NODE_ID, node_id1, node_id2}
        subgraph = graph.dependency_graph(node_id1)
        assert set(subgraph.nodes) == {SRC_NODE_ID, node_id1}


class TestExecutionState:
    @pytest.mark.asyncio
    async def test_wait_for(self, setup_state):
        state, graph, src, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        assert not state.ready[node_id2].is_set()

        # This coroutine should block until the event is set
        async def wait():
            await state.wait_for(node_id2)
            return True

        task = asyncio.create_task(wait())
        await asyncio.sleep(0.1)  # Ensure the task is waiting

        state.ready[node_id2].set()
        result = await task
        assert result is True

    def test_collect_value(self):
        features = Features({"val": {"x": Value("string")}})
        # create the graph and add the source node
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(features)
        src = FeatureRef(
            key_=tuple(), node_id_=src_node_id, flow_=graph, feature_=features
        )
        # create arguments for execution state
        rank = 0
        index = [0, 1, 2]
        batch = {"val": [{"x": "a"}, {"x": "b"}, {"x": "c"}]}
        # create execution state
        state = ExecutionState(graph, batch, index, rank)
        # sollect full node output
        collected = state.collect_value(src)
        assert collected == batch
        # collect sub-feature of node output
        collected = state.collect_value(src.val)
        assert collected == {"x": ["a", "b", "c"]}

    def test_collect_inputs(self, setup_state):
        state, graph, src, out1, out2 = setup_state

        batch = {"y": [-1, -2, -3]}

        # collect inputs for node
        node_id = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=src), mock_features
        )
        collected = state.collect_inputs(node_id)
        assert collected == {"x": [{"y": -1}, {"y": -2}, {"y": -3}]}

        # collect sub-feature inputs for node
        node_id = graph.add_processor_node(
            NoOp(), NoOpInputRefs(x=src.y), mock_features
        )
        collected = state.collect_inputs(node_id)
        assert collected == {"x": [-1, -2, -3]}

    def test_collect_inputs_parent_not_ready(self, setup_state):
        state, graph, src, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        # Ensure the parent's output is not ready
        state.ready[node_id1] = asyncio.Event()

        with pytest.raises(AssertionError):
            state.collect_inputs(node_id2)

    def test_capture_output(self, setup_state):
        state, graph, src, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        output = {"x": [1, 2, 3]}
        state.capture_output(node_id2, output)

        assert state.outputs[node_id2] == output
        assert state.ready[node_id2].is_set()


class TestDataFlowExecutor:
    @pytest.mark.asyncio
    async def test_execute_node(self, setup_executor, setup_state):
        executor, graph, src, out1, out2 = setup_executor
        state, graph, src, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        await executor.execute_node(node_id1, state)

        assert node_id1 in state.outputs
        assert state.ready[node_id1].is_set()

    @pytest.mark.asyncio
    async def test_execute(self, setup_executor):
        executor, graph, src, node_id1, node_id2 = setup_executor
        batch = {"y": [1, 2, 3]}
        index = [0, 1, 2]
        rank = 0

        await executor.execute(batch, index, rank)


class TestDataFlow:
    def test_build_flow(self, setup_flow):
        flow, src, out1, out2 = setup_flow

        # out features only set after build
        with pytest.raises(RuntimeError):
            flow.out_features

        sub_flow = flow.build(collect=out2)
        assert len(sub_flow._graph) == 3
        assert sub_flow.out_features == out2

        sub_flow = flow.build(collect=out1)
        assert len(sub_flow._graph) == 2
        assert sub_flow.out_features == out1

        sub_flow = flow.build(collect=flow.src_features)
        assert len(sub_flow._graph) == 1
        assert sub_flow.out_features == flow.src_features

    def test_batch_process(self, setup_flow):
        # build flow
        flow, src, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create input
        batch = {"y": [0, 1, 2]}
        index = [0, 1, 2]
        rank = 0
        # batch process and check output
        out = flow.batch_process(batch, index, rank)
        assert out == {"y": [0, 1, 2]}

    def test_apply_to_dataset(self, setup_flow):
        # build flow
        flow, src, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"y": list(range(100))}, features=flow.src_features.feature_
        )
        # apply flow to dataset
        out_ds = flow.apply(ds, batch_size=10)
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["y"], out_ds["y"]))

    def test_apply_to_dataset_dict(self, setup_flow):
        # build flow
        flow, src, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"y": list(range(100))}, features=flow.src_features.feature_
        )
        ds_dict = datasets.DatasetDict({"train": ds})
        # apply flow to dataset
        out_ds = flow.apply(ds_dict, batch_size=10)["train"]
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["y"], out_ds["y"]))

    def test_apply_to_iterable_dataset(self, setup_flow):
        # build flow
        flow, src, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"y": list(range(100))}, features=flow.src_features.feature_
        )
        # apply flow to dataset
        out_ds = flow.apply(
            ds.to_iterable_dataset(num_shards=5), batch_size=10
        )
        out_ds = datasets.Dataset.from_generator(
            lambda: (yield from out_ds), features=out_ds.features
        )
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["y"], out_ds["y"]))

    def test_apply_to_iterable_dataset_dict(self, setup_flow):
        # build flow
        flow, src, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"y": list(range(100))}, features=flow.src_features.feature_
        )
        ds_dict = datasets.IterableDatasetDict(
            {"train": ds.to_iterable_dataset(num_shards=5)}
        )
        # apply flow to dataset
        out_ds = flow.apply(ds_dict, batch_size=10)["train"]
        out_ds = datasets.Dataset.from_generator(
            lambda: (yield from out_ds), features=out_ds.features
        )
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["y"], out_ds["y"]))

    @pytest.mark.parametrize(
        "with_edge_labels, edge_label_format",
        [
            (False, "{name}={key}"),
            (True, "{name}={key}"),
            (True, "{name}"),
            (True, "{key}"),
        ],
    )
    def test_plot(self, setup_flow, with_edge_labels, edge_label_format):
        flow, src, out1, out2 = setup_flow
        # Ensure the plot function runs without errors and returns an Axes object
        with patch(
            "matplotlib.pyplot.show"
        ):  # Mock plt.show to avoid displaying the plot during tests
            ax = flow.plot(
                src_node_label="[ROOT]", with_edge_labels=with_edge_labels
            )
            assert isinstance(ax, plt.Axes)

        # Check if node labels are correct
        for node, data in flow._graph.nodes(data=True):
            node_label = (
                "[ROOT]"
                if node == SRC_NODE_ID
                else type(data[DataFlowGraph.NodeProperty.PROCESSOR]).__name__
            )
            assert any(
                node_label in text.get_text() for text in ax.texts
            ), f"Node label {node_label} is missing in the plot."

        # Check if edge labels are correct
        for edge in flow._graph.edges(data=True):
            _, _, data = edge
            edge_label = edge_label_format.format(
                name=data[DataFlowGraph.EdgeProperty.NAME],
                key=data[DataFlowGraph.EdgeProperty.KEY],
            )

            if with_edge_labels:
                assert any(
                    edge_label in text.get_text() for text in ax.texts
                ), f"Edge label {edge_label} is missing in the plot."
            else:
                assert all(
                    edge_label not in text.get_text() for text in ax.texts
                ), f"Edge label {edge_label} in the plot but shouldn't be included."
