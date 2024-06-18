import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import DataFlowExecutor, ExecutionState
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.processor import IOContext

from .mock import MockInputRefs, MockProcessor


class TestExecutionState:
    def test_initial_state(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state
        # check initial state
        assert set(state.outputs.keys()) == {graph.src_node_id}
        assert set(state.ready.keys()) == {const_node, proc_node}

    @pytest.mark.asyncio
    async def test_wait_for(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state

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
        state, graph, const_node, proc_node, agg_node = setup_state

        # capture output of the const node
        state.capture_output(const_node, {"value": [0, 0, 0]})

        # collect inputs for processor
        collected = state.collect_inputs(proc_node)
        assert collected == {"a": [1, 2, 3], "b": [0, 0, 0]}

        # create processor
        p = MockProcessor()
        # create input refs from source features
        i = MockInputRefs(
            a=graph.get_node_output_ref(graph.src_node_id),
            b=graph.get_node_output_ref(graph.src_node_id).x,
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
        state, graph, const_node, node_id_1, agg_node = setup_state

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
        state, graph, _, node_id, _ = setup_state
        # capture output
        output = {"y": [1, 2, 3]}
        state.capture_output(node_id, output)
        # check state
        assert state.outputs[node_id] == output
        assert state.ready[node_id].is_set()


class TestDataFlowExecutor:
    def test_error_on_invalid_init_args(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state
        out = graph.get_node_output_ref(proc_node)
        # collect (out.y) is not a feature mapping
        with pytest.raises(TypeError):
            DataFlowExecutor(graph, out.y, None)

    @pytest.mark.asyncio
    async def test_execute_const(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, None)
        # run processor node in executor
        await executor.execute_node(const_node, state)
        # check state after execution
        assert const_node in state.outputs
        assert state.ready[const_node].is_set()

    @pytest.mark.asyncio
    async def test_execute_processor(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state
        # capture output of const node
        state.capture_output(const_node, {"value": [0, 0, 0]})
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, None)
        # run processor node in executor
        await executor.execute_node(proc_node, state)

        # build io context for the processor
        io_ctx = IOContext(
            _IOContext__node_id=proc_node,
            inputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.IN_FEATURES
            ],
            outputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
        )

        # make sure the processor was called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 0}, 0, 0, io_ctx),
                call({"a": 2, "b": 0}, 1, 0, io_ctx),
                call({"a": 3, "b": 0}, 2, 0, io_ctx),
            ]
        )
        # check state after execution
        assert proc_node in state.outputs
        assert state.ready[proc_node].is_set()

    @pytest.mark.asyncio
    async def test_execute_aggregator(self, setup_state):
        state, graph, const_node, proc_node, agg_node = setup_state
        # capture output of const node
        state.capture_output(const_node, {"value": [0, 0, 0]})
        # create aggregation manager
        manager = MagicMock()
        manager.aggregate = AsyncMock()
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, manager)
        # run processor node in executor
        await executor.execute_node(agg_node, state)

        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [0, 0, 0]}, [0, 1, 2], 0
        )

    @pytest.mark.asyncio
    async def test_execute_graph(self, setup_graph):
        graph, const_node, proc_node, agg_node = setup_graph
        # create aggregation manager
        mock_manager = MagicMock()
        mock_manager.aggregate = AsyncMock()
        # build executor
        out = graph.get_node_output_ref(proc_node)
        executor = DataFlowExecutor(graph, out, mock_manager)
        # execute graph
        batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
        await executor.execute(batch, index, rank)

        # build io context for the processor
        io_ctx = IOContext(
            _IOContext__node_id=proc_node,
            inputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.IN_FEATURES
            ],
            outputs=graph.nodes[proc_node][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
        )
        # make sure the processor is called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 0}, 0, 0, io_ctx),
                call({"a": 2, "b": 0}, 1, 0, io_ctx),
                call({"a": 3, "b": 0}, 2, 0, io_ctx),
            ]
        )
        # make sure the aggregator is called correctly
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        mock_manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [0, 0, 0]}, [0, 1, 2], 0
        )
