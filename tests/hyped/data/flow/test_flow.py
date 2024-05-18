import asyncio

import datasets
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
from hyped.data.flow.ref import FeatureRef
from hyped.data.processors.ops.noop import NoOp, NoOpInputRefs


# fixtures
@pytest.fixture
def setup_graph():
    # create the graph and add the source node
    graph = DataFlowGraph()
    src = graph.add_src_node(Features({"val": Value("int32")}))
    # add processors
    o1 = graph.add_processor(NoOp(), NoOpInputRefs(x=src.val))
    o2 = graph.add_processor(NoOp(), NoOpInputRefs(x=o1.y))
    # return setup
    return graph, o1, o2


@pytest.fixture
def setup_executor(setup_graph):
    graph, out1, out2 = setup_graph
    executor = DataFlowExecutor(graph, out2)
    return executor, graph, out1, out2


@pytest.fixture
def setup_state(setup_graph):
    graph, out1, out2 = setup_graph
    batch = {"val": [-1, -2, -3]}
    index = [0, 1, 2]
    rank = 0
    state = ExecutionState(graph, batch, index, rank)
    return state, graph, out1, out2


@pytest.fixture
def setup_flow(setup_graph):
    graph, o1, o2 = setup_graph
    # create data flow
    flow = DataFlow(Features({"val": Value("int32")}))
    flow._graph = graph
    return flow, o1, o2


class TestDataFlowGraph:
    def test_add_processor(self):
        features = Features({"val": Value("int32")})
        # create the graph and add the source node
        graph = DataFlowGraph()
        src = graph.add_src_node(features)
        assert SRC_NODE_ID in graph
        assert graph.nodes[SRC_NODE_ID]["processor"] is None
        assert graph.nodes[SRC_NODE_ID]["features"] == features
        assert src.feature_ == features

        # add processors
        p1, i1 = NoOp(), NoOpInputRefs(x=src.val)
        o1 = graph.add_processor(p1, i1)

        # check node
        assert o1.node_id_ in graph
        assert graph.nodes[o1.node_id_]["processor"] == p1
        assert graph.nodes[o1.node_id_]["features"] == Features(
            {"y": Value("int32")}
        )
        # check edge
        assert graph.has_edge(SRC_NODE_ID, o1.node_id_)
        for n, r in i1.named_refs.items():
            assert n in graph[SRC_NODE_ID][o1.node_id_]
            assert graph[SRC_NODE_ID][o1.node_id_][n]["feature_key"] == r.key_

        p2, i2 = NoOp(), NoOpInputRefs(x=o1.y)
        o2 = graph.add_processor(p2, i2)

        # check node
        assert o2.node_id_ in graph
        assert graph.nodes[o2.node_id_]["processor"] == p2
        assert graph.nodes[o2.node_id_]["features"] == Features(
            {"y": Value("int32")}
        )
        # check edge
        assert graph.has_edge(o1.node_id_, o2.node_id_)
        for n, r in i2.named_refs.items():
            assert n in graph[o1.node_id_][o2.node_id_]
            assert graph[o1.node_id_][o2.node_id_][n]["feature_key"] == r.key_

    def test_add_processor_invalid_input(self):
        g1 = DataFlowGraph()
        g2 = DataFlowGraph()
        # add source nodes
        x1 = g1.add_src_node(Features({"val": Value("int32")}))
        x2 = g2.add_src_node(Features({"val": Value("int32")}))
        # add valid nodes
        g1.add_processor(NoOp(), NoOpInputRefs(x=x1.val))
        g2.add_processor(NoOp(), NoOpInputRefs(x=x2.val))
        # try add invalid node
        with pytest.raises(RuntimeError):
            g1.add_processor(NoOp(), NoOpInputRefs(x=x2.val))

    def test_dependency_graph(self, setup_graph):
        graph, out1, out2 = setup_graph
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        subgraph = graph.dependency_graph(node_id2)
        assert set(subgraph.nodes) == {SRC_NODE_ID, node_id1, node_id2}
        subgraph = graph.dependency_graph(node_id1)
        assert set(subgraph.nodes) == {SRC_NODE_ID, node_id1}


class TestExecutionState:
    @pytest.mark.asyncio
    async def test_wait_for(self, setup_state):
        state, graph, out1, out2 = setup_state
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
        # create the graph and add the source node
        graph = DataFlowGraph()
        src = graph.add_src_node(Features({"val": {"x": Value("string")}}))
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

    def test_collect_inputs(self):
        # create the graph and add the source node
        graph = DataFlowGraph()
        src = graph.add_src_node(Features({"val": {"x": Value("string")}}))
        # create arguments for execution state
        rank = 0
        index = [0, 1, 2]
        batch = {"val": ["a", "b", "c"]}
        # create execution state
        state = ExecutionState(graph, batch, index, rank)

        # add processor and collect inputs for it
        out = graph.add_processor(NoOp(), NoOpInputRefs(x=src))
        collected, index = state.collect_inputs(out.node_id_)
        assert index == index
        assert collected == {"x": [{"val": "a"}, {"val": "b"}, {"val": "c"}]}

        # add processor and collect inputs for it
        out = graph.add_processor(NoOp(), NoOpInputRefs(x=src.val))
        collected, index = state.collect_inputs(out.node_id_)
        assert index == index
        assert collected == {"x": ["a", "b", "c"]}

    def test_collect_inputs_parent_not_ready(self, setup_state):
        state, graph, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        # Ensure the parent's output is not ready
        state.ready[node_id1] = asyncio.Event()

        with pytest.raises(AssertionError):
            state.collect_inputs(node_id2)

    def test_capture_output(self, setup_state):
        state, graph, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        output = {"x": [1, 2, 3]}
        new_index = [4, 5, 6]

        state.capture_output(node_id2, output, new_index)

        assert state.outputs[node_id2] == output
        assert state.index_hash[node_id2] == hash(tuple(new_index))
        assert state.index_lookup[hash(tuple(new_index))] == new_index
        assert state.ready[node_id2].is_set()


class TestDataFlowExecutor:
    @pytest.mark.asyncio
    async def test_execute_node(self, setup_executor, setup_state):
        executor, graph, out1, out2 = setup_executor
        state, graph, out1, out2 = setup_state
        node_id1, node_id2 = out1.node_id_, out2.node_id_

        await executor.execute_node(node_id1, state)

        assert node_id1 in state.outputs
        assert state.ready[node_id1].is_set()

    @pytest.mark.asyncio
    async def test_execute(self, setup_executor):
        executor, graph, node_id1, node_id2 = setup_executor
        batch = {"val": [1, 2, 3]}
        index = [0, 1, 2]
        rank = 0

        output_batch = await executor.execute(batch, index, rank)


class TestDataFlow:
    def test_build_flow(self, setup_flow):
        flow, out1, out2 = setup_flow

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
        flow, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create input
        batch = {"val": [0, 1, 2]}
        index = [0, 1, 2]
        rank = 0
        # batch process and check output
        out = flow.batch_process(batch, index, rank)
        assert out == {"y": [0, 1, 2]}

    def test_apply_to_dataset(self, setup_flow):
        # build flow
        flow, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"val": list(range(100))}, features=flow.src_features.feature_
        )
        # apply flow to dataset
        out_ds = flow.apply(ds, batch_size=10)
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["val"], out_ds["y"]))

    def test_apply_to_dataset_dict(self, setup_flow):
        # build flow
        flow, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"val": list(range(100))}, features=flow.src_features.feature_
        )
        ds_dict = datasets.DatasetDict({"train": ds})
        # apply flow to dataset
        out_ds = flow.apply(ds_dict, batch_size=10)["train"]
        # check output
        assert out_ds.features == flow.out_features.feature_
        assert all(i == j for i, j in zip(ds["val"], out_ds["y"]))

    def test_apply_to_iterable_dataset(self, setup_flow):
        # build flow
        flow, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"val": list(range(100))}, features=flow.src_features.feature_
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
        assert all(i == j for i, j in zip(ds["val"], out_ds["y"]))

    def test_apply_to_iterable_dataset_dict(self, setup_flow):
        # build flow
        flow, out1, out2 = setup_flow
        flow = flow.build(collect=out2)
        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"val": list(range(100))}, features=flow.src_features.feature_
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
        assert all(i == j for i, j in zip(ds["val"], out_ds["y"]))
