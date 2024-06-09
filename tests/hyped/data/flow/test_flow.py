import asyncio
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock, call, patch

import datasets
import matplotlib.pyplot as plt
import pytest
from datasets import Features, Value

from hyped.data.flow.aggregators.base import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    DataAggregationRef,
)
from hyped.data.flow.flow import (
    SRC_NODE_ID,
    DataFlow,
    DataFlowExecutor,
    DataFlowGraph,
    ExecutionState,
)
from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.data.flow.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.refs.outputs import OutputFeature, OutputRefs
from hyped.data.flow.refs.ref import FeatureRef


class MockInputRefs(InputRefs):
    a: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]
    b: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]


class MockOutputRefs(OutputRefs):
    y: Annotated[FeatureRef, OutputFeature(Value("int64"))]


class MockProcessorConfig(BaseDataProcessorConfig):
    i: int = 0


class MockProcessor(
    BaseDataProcessor[MockProcessorConfig, MockInputRefs, MockOutputRefs]
):
    # mock process function
    process = MagicMock(return_value={"y": 0})


class MockAggregatorConfig(BaseDataAggregatorConfig):
    i: int = 0


class MockAggregator(
    BaseDataAggregator[MockAggregatorConfig, MockInputRefs, int]
):
    # mock abstract functions
    initialize = MagicMock()
    extract = AsyncMock()
    update = AsyncMock()


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
    src_node_id = graph.add_source_node(src_features)
    # create processor
    p = MockProcessor()
    a = MockAggregator()

    # create input refs from source features
    i = MockInputRefs(
        a=graph.get_node_output_ref(src_node_id).x,
        b=graph.get_node_output_ref(src_node_id).x,
    )
    o = p._out_refs_type.build_features(p.config, i)

    # add nodes
    proc_node = graph.add_processor_node(p, i, o)
    agg_node = graph.add_processor_node(a, i, None)

    return graph, proc_node, agg_node


@pytest.fixture
def setup_state(setup_graph):
    graph, proc_node, agg_node = setup_graph
    # create state
    batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
    state = ExecutionState(graph, batch, index, rank)
    # return setup
    return state, graph, proc_node, agg_node


@pytest.fixture
def setup_flow(setup_graph):
    graph, proc_node, agg_node = setup_graph
    # create data flow
    flow = DataFlow(Features({"x": Value("int64")}))
    flow._graph = graph
    # return setup
    return flow, graph, proc_node, agg_node


class TestDataFlowGraph:
    def test_add_source_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # check source node was added
        assert SRC_NODE_ID in graph
        # check node properties
        node = graph.nodes[SRC_NODE_ID]
        assert node[DataFlowGraph.NodeProperty.DEPTH] == 0
        assert node[DataFlowGraph.NodeProperty.PROCESSOR] is None
        assert (
            node[DataFlowGraph.NodeProperty.PROCESSOR_TYPE]
            == DataFlowGraph.NodeType.SOURCE
        )
        assert node[DataFlowGraph.NodeProperty.IN_FEATURES] == src_features
        assert node[DataFlowGraph.NodeProperty.OUT_FEATURES] == src_features

    def test_add_processor_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)

        # check processor node was added
        assert node_id in graph
        # check node properties
        node = graph.nodes[node_id]
        assert node[DataFlowGraph.NodeProperty.DEPTH] == 1
        assert node[DataFlowGraph.NodeProperty.PROCESSOR] == p
        assert (
            node[DataFlowGraph.NodeProperty.PROCESSOR_TYPE]
            == DataFlowGraph.NodeType.DATA_PROCESSOR
        )
        assert node[DataFlowGraph.NodeProperty.IN_FEATURES] == i.features_
        assert node[DataFlowGraph.NodeProperty.OUT_FEATURES] == o
        # check edges
        assert graph.has_edge(SRC_NODE_ID, node_id)
        for n, r in i.named_refs.items():
            assert n in graph[SRC_NODE_ID][node_id]
            assert (
                graph[SRC_NODE_ID][node_id][n][DataFlowGraph.EdgeProperty.KEY]
                == r.key_
            )

    def test_add_aggregator_node(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        a = MockAggregator()
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        # add aggregator node
        node_id = graph.add_processor_node(a, i, None)

        # check processor node was added
        assert node_id in graph
        # check node properties
        node = graph.nodes[node_id]
        assert node[DataFlowGraph.NodeProperty.DEPTH] == 1
        assert node[DataFlowGraph.NodeProperty.PROCESSOR] == a
        assert (
            node[DataFlowGraph.NodeProperty.PROCESSOR_TYPE]
            == DataFlowGraph.NodeType.DATA_AGGREGATOR
        )
        assert node[DataFlowGraph.NodeProperty.IN_FEATURES] == i.features_
        assert node[DataFlowGraph.NodeProperty.OUT_FEATURES] is None
        # check edges
        assert graph.has_edge(SRC_NODE_ID, node_id)
        for n, r in i.named_refs.items():
            assert n in graph[SRC_NODE_ID][node_id]
            assert (
                graph[SRC_NODE_ID][node_id][n][DataFlowGraph.EdgeProperty.KEY]
                == r.key_
            )

    def test_depth_and_width(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)
        # check depth of source node
        assert graph.nodes[src_node_id][DataFlowGraph.NodeProperty.DEPTH] == 0
        # check graph properties
        assert graph.depth == 1
        assert graph.width == 1

        # create processor
        p = MockProcessor()

        # create input refs from source features
        i1 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i1)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i1, o)
        assert graph.nodes[node_id_1][DataFlowGraph.NodeProperty.DEPTH] == 1
        # check graph properties
        assert graph.depth == 2
        assert graph.width == 1

        # create input refs from first-level outputs
        i2 = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i2)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i2, o)
        assert graph.nodes[node_id_2][DataFlowGraph.NodeProperty.DEPTH] == 2
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 1

        # create in put refs from source and first level nodes
        i3 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i3)
        # add third level processor
        node_id_3 = graph.add_processor_node(p, i3, o)
        assert graph.nodes[node_id_3][DataFlowGraph.NodeProperty.DEPTH] == 2
        # check graph properties
        assert graph.depth == 3
        assert graph.width == 2

        # create in put refs from source and second level nodes
        i4 = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(node_id_2).y,
        )
        o = p._out_refs_type.build_features(p.config, i4)
        # add third level processor
        node_id_4 = graph.add_processor_node(p, i4, o)
        assert graph.nodes[node_id_4][DataFlowGraph.NodeProperty.DEPTH] == 3
        # check graph properties
        assert graph.depth == 4
        assert graph.width == 2

    def test_add_processor_invalid_input(self):
        g1 = DataFlowGraph()
        g2 = DataFlowGraph()
        # mock features
        src_features = Features({"x": Value("int64")})
        out_features = Features({"y": Value("int64")})
        # add source nodes
        g1_src_node_id = g1.add_source_node(src_features)
        g2_src_node_id = g2.add_source_node(src_features)
        # create processor instance
        p = MockProcessor()
        # add valid nodes
        g1.add_processor_node(
            p,
            MockInputRefs(
                a=g1.get_node_output_ref(g1_src_node_id).x,
                b=g1.get_node_output_ref(g1_src_node_id).x,
            ),
            out_features,
        )
        g2.add_processor_node(
            p,
            MockInputRefs(
                a=g2.get_node_output_ref(g2_src_node_id).x,
                b=g2.get_node_output_ref(g2_src_node_id).x,
            ),
            out_features,
        )
        # try add invalid node
        with pytest.raises(RuntimeError):
            g1.add_processor_node(
                p,
                MockInputRefs(
                    a=g2.get_node_output_ref(g2_src_node_id).x,
                    b=g1.get_node_output_ref(g1_src_node_id).x,
                ),
                out_features,
            )

    def test_get_node_output_ref(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor and aggregator
        p = MockProcessor()
        a = MockAggregator()

        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)

        # add processor and aggregator
        node_id_1 = graph.add_processor_node(p, i, o)
        node_id_2 = graph.add_processor_node(a, i, None)

        # test feature reference to source features
        ref = graph.get_node_output_ref(src_node_id)
        assert ref == FeatureRef(
            node_id_=src_node_id,
            key_=tuple(),
            flow_=graph,
            feature_=src_features,
        )
        # test feature reference to processor output
        ref = graph.get_node_output_ref(node_id_1)
        assert ref == MockOutputRefs(graph, node_id_1, o)

        # test feature reference to processor output
        ref = graph.get_node_output_ref(node_id_2)
        assert ref == DataAggregationRef(
            node_id_=node_id_2, flow_=graph, type_=int
        )

    def test_dependency_graph(self):
        # create graph
        graph = DataFlowGraph()
        # add source node
        src_features = Features({"x": Value("int64")})
        src_node_id = graph.add_source_node(src_features)

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(src_node_id).x,
            b=graph.get_node_output_ref(src_node_id).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        node_id_1 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add second level processor
        node_id_2 = graph.add_processor_node(p, i, o)

        # create input refs from first-level outputs
        i = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_2).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add third level processor
        node_id_3 = graph.add_processor_node(p, i, o)

        subgraph = graph.dependency_graph({src_node_id})
        assert set(subgraph.nodes) == {src_node_id}

        subgraph = graph.dependency_graph({node_id_1})
        assert set(subgraph.nodes) == {src_node_id, node_id_1}

        subgraph = graph.dependency_graph({node_id_2})
        assert set(subgraph.nodes) == {src_node_id, node_id_1, node_id_2}

        subgraph = graph.dependency_graph({node_id_3})
        assert set(subgraph.nodes) == {
            src_node_id,
            node_id_1,
            node_id_2,
            node_id_3,
        }


class TestExecutionState:
    def test_initial_state(self, setup_state):
        state, graph, proc_node, agg_node = setup_state
        # check initial state
        assert set(state.outputs.keys()) == {SRC_NODE_ID}
        assert set(state.ready.keys()) == {proc_node}

    @pytest.mark.asyncio
    async def test_wait_for(self, setup_state):
        state, graph, proc_node, _ = setup_state

        # make sure the node is not ready yet
        assert not state.ready[proc_node].is_set()

        # This coroutine should block until the event is set
        async def wait():
            await state.wait_for(proc_node)
            return True

        # schedule coroutine
        task = asyncio.create_task(wait())
        await asyncio.sleep(0.1)  # Ensure the task is waiting

        # set ready event
        state.ready[proc_node].set()
        result = await task
        assert result is True

    def test_collect_value(self):
        # nested input features
        src_features = Features({"val": {"x": Value("string")}})
        batch = {"val": [{"x": "a"}, {"x": "b"}, {"x": "c"}]}
        # build simple graph
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(src_features)
        src = graph.get_node_output_ref(src_node_id)
        # create execution state
        state = ExecutionState(graph, batch, [0, 1, 2], 0)
        # collect full node output
        collected = state.collect_value(src)
        assert collected == batch
        # collect sub-feature of node output
        collected = state.collect_value(src.val)
        assert collected == {"x": ["a", "b", "c"]}

    def test_collect_inputs(self, setup_state):
        state, graph, proc_node, _ = setup_state

        # collect inputs for processor
        collected = state.collect_inputs(proc_node)
        assert collected == {"a": [1, 2, 3], "b": [1, 2, 3]}

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(SRC_NODE_ID),
            b=graph.get_node_output_ref(SRC_NODE_ID).x,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add first level processor
        node_id = graph.add_processor_node(p, i, o)

        # collect nested inputs for processor
        collected = state.collect_inputs(node_id)
        assert collected == {
            "a": [{"x": 1}, {"x": 2}, {"x": 3}],
            "b": [1, 2, 3],
        }

    def test_collect_inputs_parent_not_ready(self, setup_state):
        state, graph, node_id_1, _ = setup_state

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(node_id_1).y,
            b=graph.get_node_output_ref(node_id_1).y,
        )
        o = p._out_refs_type.build_features(p.config, i)
        # add processor
        node_id_2 = graph.add_processor_node(p, i, o)

        # Ensure the parent's output is not ready
        state.ready[node_id_1] = asyncio.Event()

        with pytest.raises(AssertionError):
            state.collect_inputs(node_id_2)

    def test_capture_outputs(self, setup_state):
        state, graph, node_id, _ = setup_state
        # capture output
        output = {"y": [1, 2, 3]}
        state.capture_output(node_id, output)
        # check state
        assert state.outputs[node_id] == output
        assert state.ready[node_id].is_set()


class TestDataFlowExecutor:
    @pytest.mark.asyncio
    async def test_execute_processor(self, setup_state):
        state, graph, proc_node, _ = setup_state
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, None)
        # run processor node in executor
        await executor.execute_node(proc_node, state)

        # make sure the processor was called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 1}, 0, 0),
                call({"a": 2, "b": 2}, 1, 0),
                call({"a": 3, "b": 3}, 2, 0),
            ]
        )
        # check state after execution
        assert proc_node in state.outputs
        assert state.ready[proc_node].is_set()

    @pytest.mark.asyncio
    async def test_execute_aggregator(self, setup_state):
        state, graph, proc_node, agg_node = setup_state
        # create aggregation manager
        manager = MagicMock()
        manager.aggregate = AsyncMock()
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, manager)
        # run processor node in executor
        await executor.execute_node(agg_node, state)

        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]
        manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [1, 2, 3]}, [0, 1, 2], 0
        )

    @pytest.mark.asyncio
    async def test_execute_graph(self, setup_graph):
        graph, proc_node, agg_node = setup_graph
        # create aggregation manager
        mock_manager = MagicMock()
        mock_manager.aggregate = AsyncMock()
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, mock_manager)
        # execute graph
        batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
        await executor.execute(batch, index, rank)
        # make sure the processor is called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 1}, 0, 0),
                call({"a": 2, "b": 2}, 1, 0),
                call({"a": 3, "b": 3}, 2, 0),
            ]
        )
        # make sure the aggregator is called correctly
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]
        mock_manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [1, 2, 3]}, [0, 1, 2], 0
        )


class TestDataFlow:
    @pytest.fixture(autouse=True)
    def mock_manager(self):
        with patch(
            "hyped.data.flow.flow.DataAggregationManager"
        ) as mock_manager:
            mock_manager = mock_manager()
            mock_manager.aggregate = AsyncMock()
            mock_manager.values_proxy = MagicMock()
            yield mock_manager

    def test_build_flow(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow

        src_ref = graph.get_node_output_ref(SRC_NODE_ID)
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        # out features only set after build
        with pytest.raises(RuntimeError):
            flow.out_features

        # build subflow with processor and aggregator
        subflow, vals = flow.build(
            collect=out_ref, aggregators={"val": agg_ref}
        )
        assert len(subflow._graph) == 3
        assert subflow.out_features == out_ref
        assert vals == mock_manager.values_proxy
        # build subflow with processor only
        subflow, _ = flow.build(collect=out_ref)
        assert len(subflow._graph) == 2
        assert subflow.out_features == out_ref
        # build subflow with no processors
        subflow, _ = flow.build(collect=src_ref)
        assert len(subflow._graph) == 1
        assert subflow.out_features == src_ref

    def test_batch_process(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow

        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        flow, vals = flow.build(collect=out_ref, aggregators={"val": agg_ref})
        # check output types
        assert isinstance(flow, DataFlow)
        assert vals == mock_manager.values_proxy

        # run batch process
        batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
        out = flow.batch_process(batch, index, rank)

        # make sure the processor is called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 1}, 0, 0),
                call({"a": 2, "b": 2}, 1, 0),
                call({"a": 3, "b": 3}, 2, 0),
            ]
        )
        # make sure the aggregator is called correctly
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]
        mock_manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [1, 2, 3]}, [0, 1, 2], 0
        )

    def test_apply_overload(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]

        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"x": []}, features=flow.src_features.feature_
        )

        # no output features specified
        with pytest.raises(RuntimeError):
            flow.apply(ds)

        # apply flow to dataset
        out_ds, _ = flow.apply(
            ds,
            collect=out_ref,
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)

        # apply flow to dataset with aggregators
        out_ds, vals = flow.apply(
            ds,
            collect=out_ref,
            aggregators={"val": agg_ref},
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == mock_manager.values_proxy.copy()

        built_flow, _ = flow.build(collect=out_ref)
        # apply flow to dataset
        out_ds, _ = built_flow.apply(ds)
        assert isinstance(out_ds, datasets.Dataset)

        built_flow, vals = flow.build(
            collect=out_ref, aggregators={"val": agg_ref}
        )
        assert vals == mock_manager.values_proxy
        # apply flow to dataset
        out_ds, vals = built_flow.apply(ds)
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == mock_manager.values_proxy.copy()

    def test_apply_to_dataset(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]

        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"x": list(range(100))}, features=flow.src_features.feature_
        )

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == mock_manager.values_proxy.copy()
        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": i}, i, 0) for i in range(100)]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": list(range(i * 10, (i + 1) * 10)),
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_dataset_dict(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]

        # create dummy dataset
        ds = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"x": list(range(50))}, features=flow.src_features.feature_
                ),
                "test": datasets.Dataset.from_dict(
                    {"x": list(range(50, 100))},
                    features=flow.src_features.feature_,
                ),
            }
        )

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.DatasetDict)
        assert out_ds.keys() == ds.keys()
        assert vals == mock_manager.values_proxy.copy()
        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": i}, i % 50, 0) for i in range(100)]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": list(range(i * 10, (i + 1) * 10)),
                    },
                    list(range((i % 5) * 10, ((i % 5) + 1) * 10)),
                    0,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]

        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"x": list(range(100))}, features=flow.src_features.feature_
        ).to_iterable_dataset(num_shards=5)

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.IterableDataset)
        assert vals == mock_manager.values_proxy

        # at this point the processors shouldn't be called yet
        assert not p.process.called
        assert not mock_manager.aggregate.called

        # consume iterable dataset
        for _ in out_ds:
            pass

        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": i}, i, 0) for i in range(100)]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": list(range(i * 10, (i + 1) * 10)),
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset_dict(self, setup_flow, mock_manager):
        flow, graph, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeProperty.PROCESSOR]
        a = graph.nodes[agg_node][DataFlowGraph.NodeProperty.PROCESSOR]

        # create dummy dataset
        ds = datasets.IterableDatasetDict(
            {
                "train": datasets.Dataset.from_dict(
                    {"x": list(range(50))}, features=flow.src_features.feature_
                ).to_iterable_dataset(num_shards=5),
                "test": datasets.Dataset.from_dict(
                    {"x": list(range(50, 100))},
                    features=flow.src_features.feature_,
                ).to_iterable_dataset(num_shards=5),
            }
        )

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.IterableDatasetDict)
        assert out_ds.keys() == ds.keys()

        # at this point the processors shouldn't be called yet
        assert not p.process.called
        assert not mock_manager.aggregate.called

        # consume train dataset
        for _ in out_ds["train"]:
            pass

        # make sure processor is called for all samples in the train dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": i}, i % 50, 0) for i in range(50)]
        )
        # make sure the aggregator is called for all batches in the train dataset
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": list(range(i * 10, (i + 1) * 10)),
                    },
                    list(range((i % 5) * 10, ((i % 5) + 1) * 10)),
                    0,
                )
                for i in range(5)
            ]
        )

        # consume train dataset
        for _ in out_ds["test"]:
            pass

        # make sure processor is called for all samples in the train dataset
        p.process.assert_has_calls(
            [call({"a": 50 + i, "b": 50 + i}, i, 0) for i in range(50)]
        )
        # make sure the aggregator is called for all batches in the train dataset
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(50 + i * 10, 50 + (i + 1) * 10)),
                        "b": list(range(50 + i * 10, 50 + (i + 1) * 10)),
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                )
                for i in range(5)
            ]
        )

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
        flow, graph, proc_node, agg_node = setup_flow
        # Ensure the plot function runs without errors and returns an Axes object
        with patch(
            "matplotlib.pyplot.show"
        ):  # Mock plt.show to avoid displaying the plot during tests
            ax = flow.plot(
                src_node_label="[ROOT]",
                with_edge_labels=with_edge_labels,
                node_font_size=1e-5,
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
