"""Manages the execution of data flow graphs.

This module provides the `DataFlowExecutor` class, which is responsible for executing
a data flow graph. It manages the execution state, ensures that data processors
are executed in the correct order, and collects the results.

Classes:
    - :class:`ExecutionState`: Tracks the state during the execution of a data flow graph.
    - :class:`DataFlowExecutor`: Executes a data flow graph, managing the execution of each node and collecting results.
"""
import asyncio
from typing import Any, TypeAlias

import datasets

from .graph import DataFlowGraph
from .nodes.aggregator import BaseDataAggregator, DataAggregationManager
from .nodes.base import IOContext
from .refs.ref import FeatureRef

Batch: TypeAlias = dict[str, list[Any]]


class ExecutionState(object):
    """Tracks the state during the execution of a data flow graph.

    This class is used internally to manage the state of data processing as the
    data flow graph is executed, keeping track of outputs, indexes, and node readiness.
    """

    def __init__(
        self, graph: DataFlowGraph, batch: Batch, index: list[int], rank: int
    ):
        """Initialize the execution state.

        Args:
            graph (DataFlowGraph): The data flow graph being executed.
            batch (Batch): The initial batch of data.
            index (list[int]): The index of the batch.
            rank (int): The rank of the process in a distributed setting.
        """
        self.graph = graph
        self.index = index
        self.outputs = {graph.src_node_id: batch}

        self.rank = rank
        self.ready = {
            node_id: asyncio.Event()
            for node_id in graph.nodes()
            if (
                (node_id != graph.src_node_id)
                and not isinstance(
                    graph.nodes[node_id][DataFlowGraph.NodeAttribute.NODE_OBJ],
                    BaseDataAggregator,
                )
            )
        }

        self.graph = graph

    async def wait_for(self, node_id: str) -> None:
        """Wait until the specified node is ready.

        Args:
            node_id (str): The ID of the node to wait for.
        """
        if node_id in self.ready:
            await self.ready[node_id].wait()

    def collect_value(self, ref: FeatureRef) -> Batch:
        """Collect the values requested by the feature reference.

        Args:
            ref (FeatureRef): The feature reference indicating which
                values to collect.

        Returns:
            Batch: The collected batch of data.

        Raises:
            AssertionError: If the feature reference does not contain
                expected feature types.
        """
        assert isinstance(ref.feature_, (datasets.Features, dict)), (
            f"Expected features of type datasets.Features or dict, "
            f"but got {type(ref.feature_)}"
        )
        batch = ref.key_.index_batch(self.outputs[ref.node_id_])
        # in case the feature key is empty the collected values are already
        # in batch format, otherwise they need to be converted from a
        # list-of-dicts to a dict-of-lists
        if len(ref.key_) != 0:
            batch = (
                {key: [d[key] for d in batch] for key in batch[0].keys()}
                if len(batch) > 0
                else {}
            )

        return batch

    def collect_inputs(self, node_id: str) -> Batch:
        """Collect inputs for a given node.

        Args:
            node_id (str): The ID of the node for which to collect inputs.

        Returns:
            Batch: The collected inputs to the processor

        Raises:
            AssertionError: If inputs are collected from a node that is not ready.
            AssertionError: If the collected values are not of the expected type.
        """
        inputs = dict()
        for u, _, name, data in self.graph.in_edges(
            node_id, keys=True, data=True
        ):
            assert (u == self.graph.src_node_id) or self.ready[
                u
            ].is_set(), f"Node {u} is not ready."
            # get feature key from data
            key = data["feature_key"]
            # get the values requested by the batch
            values = key.index_batch(self.outputs[u])
            # this is always a list of values, except when the key is empty
            # in that case the values are the exact output of the source node u
            if len(key) == 0:
                assert isinstance(
                    values, (dict, datasets.formatting.formatting.LazyBatch)
                ), f"Expected values of type dict, but got {type(values)}"
                keys = values.keys()
                values = [
                    dict(zip(keys, vals)) for vals in zip(*values.values())
                ]
            assert isinstance(
                values, list
            ), f"Expected values to be a list, but got {type(values)}"
            # store the values in inputs and add keep track of the index hash
            inputs[name] = values

        return inputs

    def capture_output(self, node_id: str, output: Batch) -> None:
        """Capture the output of a node.

        Args:
            node_id (str): The ID of the node producing the output.
            output (Batch): The output batch of data.

        Raises:
            AssertionError: If the node is already set
        """
        assert not self.ready[
            node_id
        ].is_set(), f"Node {node_id} is already set."

        self.outputs[node_id] = output
        self.ready[node_id].set()


class DataFlowExecutor(object):
    """Executes a data flow graph.

    This class provides the low-level functionality for executing a data flow
    graph, managing the execution of each node and collecting results.
    """

    def __init__(
        self,
        graph: DataFlowGraph,
        collect: FeatureRef,
        aggregation_manager: None | DataAggregationManager,
    ) -> None:
        """Initialize the executor.

        Args:
            graph (DataFlowGraph): The data flow graph to execute.
            collect (FeatureRef): The feature reference to collect results.
            aggregation_manager (None | DataAggregationManager):
                The manager responsible for handling data aggregation. Can be
                None if the graph has no aggregator nodes.


        Raises:
            TypeError: If the collect feature is not of type datasets.Features.
        """
        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError(
                f"Expected collect feature of type datasets.Features or dict, "
                f"but got {type(collect.feature_)}"
            )

        self.graph = graph
        self.collect = collect
        self.aggregation_manager = aggregation_manager

    async def execute_node(self, node_id: str, state: ExecutionState) -> None:
        """Execute a single node in the data flow graph.

        Args:
            node_id (str): The ID of the node to execute.
            state (ExecutionState): The current execution state.

        Raises:
            AssertionError: If the output batch size doesn't match
                the input batch size.
        """
        if self.graph.in_degree(node_id) > 0:
            # wait for all dependencies of the current node
            deps = self.graph.predecessors(node_id)
            futures = map(state.wait_for, deps)
            await asyncio.gather(*futures)

        # collect inputs for processor execution
        inputs = state.collect_inputs(node_id)
        node_obj = self.graph.nodes[node_id][
            DataFlowGraph.NodeAttribute.NODE_OBJ
        ]
        node_type = self.graph.nodes[node_id][
            DataFlowGraph.NodeAttribute.NODE_TYPE
        ]

        if node_type == DataFlowGraph.NodeType.CONST:
            # get constants from node object
            consts = node_obj.get_const_batch(batch_size=len(state.index))
            state.capture_output(node_id, consts)
            # done
            return

        # build the io context
        io = IOContext(
            node_id=node_id,
            inputs=self.graph.nodes[node_id][
                DataFlowGraph.NodeAttribute.IN_FEATURES
            ],
            outputs=self.graph.nodes[node_id][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
        )

        if node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
            # run processor and check the output batch size
            out = await node_obj.batch_process(
                inputs, state.index, state.rank, io
            )
            assert all(
                len(vals) == len(state.index) for vals in out.values()
            ), "Output values length does not match index length."
            # capture output in execution state
            state.capture_output(node_id, out)

        elif node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # run aggregator
            await self.aggregation_manager.aggregate(
                node_obj, inputs, state.index, state.rank, io
            )

    async def execute(
        self, batch: Batch, index: list[int], rank: int
    ) -> Batch:
        """Execute the entire data flow graph.

        Args:
            batch (Batch): The initial batch of data.
            index (list[int]): The index of the batch.
            rank (int): The rank of the process in a multiprocessing setting.

        Returns:
            Batch: The final collected batch of data.
        """
        # create an execution state
        state = ExecutionState(self.graph, batch, index, rank)
        # execute all processors in the flow
        await asyncio.gather(
            *[
                self.execute_node(node_id, state)
                for node_id in self.graph.nodes()
                if node_id != self.graph.src_node_id
            ]
        )
        # collect output values
        return state.collect_value(self.collect)
