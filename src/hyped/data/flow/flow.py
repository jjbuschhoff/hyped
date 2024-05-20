"""Provides classes for defining and executing data flows as DAGs.

The data flow module facilitates the construction and execution of complex
data processing pipelines using a graph-based approach. It allows users to
define a set of data processors, where each processor takes input features
and produces output features. Processors can be connected to form a directed
acyclic graph, representing the flow of data through the processing pipeline.

Classes:
    - `DataFlow`: High-level interface for defining and executing data processing workflows.
    - `DataFlowGraph`: A multi-directed graph representing a data flow of data processors.
    - `ExecutionState`: Tracks the state during the execution of a data flow graph.
    - `DataFlowExecutor`: Executes a data flow graph, managing the execution of each node and collecting results.

The module also provides various utility functions and types to support data processing tasks, including:
    - Feature reference management
    - Dependency graph generation
    - State tracking during execution
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from itertools import groupby
from typing import Any, TypeVar

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
from torch.utils.data import get_worker_info
from typing_extensions import TypeAlias

from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.common.feature_checks import check_feature_equals
from hyped.data.processors.base import BaseDataProcessor
from hyped.data.processors.base.inputs import InputRefs
from hyped.data.processors.base.outputs import OutputRefs

from .ref import FeatureRef

Batch: TypeAlias = dict[str, list[Any]]
D = TypeVar(
    "D",
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
)

SRC_NODE_ID = 0

# patch asyncio if running in an async environment, such as jupyter notebook
# this fixes #26
nest_asyncio.apply()


class DataFlowGraph(nx.MultiDiGraph):
    """A multi-directed graph representing a data flow of data processors.

    This class is used internally to define a directed acyclic graph (DAG)
    where nodes represent data processors of `BaseDataProcessor` type, and
    edges define the data flow between these processors.
    """

    class NodeProperty(str, Enum):
        """Enum representing properties of a node in the data flow graph."""

        PROCESSOR = "processor"
        """
        Represents the data processor associated with the node.

        Type: BaseDataProcessor

        This property holds a reference to the data processor instance that the node
        represents within the data flow graph.
        """

        FEATURES = "features"
        """
        Represents the output features associated with the node.

        Type: datasets.Features

        This property contains the output features produced by the data processor,
        encapsulated in a HuggingFace `datasets.Features` instance. It defines the
        structure and types of data that are output by this node.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: int

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    class EdgeProperty(str, Enum):
        """Enum representing properties of an edge in the data flow graph."""

        NAME = "name"
        """
        Represents the name of the edge.

        Type: str

        This property corresponds to the keyword of the argument used as an input
        to the processor, linking the edge to a specific input parameter.

        The name is also used by NetworkX as an identifier to distinguish multiedges
        between a pair of nodes. It serves as a unique identifier for the edge.
        """

        KEY = "feature_key"
        """
        Represents the key of the feature associated with the edge.

        Type: FeatureKey

        This property specifies which subfeature of the output of the source node
        is flowing through the edge. It defines the particular feature that is being
        transmitted from one node to another in the data flow graph.
        """

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return (
            max(
                nx.get_node_attributes(
                    self, DataFlowGraph.NodeProperty.DEPTH
                ).values()
            )
            + 1
        )

    @property
    def width(self) -> int:
        """Computes the width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        # group nodes by their layer
        depths = nx.get_node_attributes(self, DataFlowGraph.NodeProperty.DEPTH)
        layers = groupby(sorted(self, key=depths.get), key=depths.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def add_source_node(self, features: datasets.Features) -> FeatureRef:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            features (datasets.Features): The features of the source node.

        Returns:
            FeatureRef: A reference to the input features.

        Raises:
            AssertionError: If the graph already contains a source node
        """
        assert SRC_NODE_ID not in self, "Graph already contains a source node"
        # add src node to graph
        self.add_node(
            SRC_NODE_ID,
            **{
                DataFlowGraph.NodeProperty.PROCESSOR: None,
                DataFlowGraph.NodeProperty.FEATURES: features,
                DataFlowGraph.NodeProperty.DEPTH: 0,
            },
        )
        # return feature ref over input features
        return FeatureRef(
            key_=tuple(), feature_=features, node_id_=SRC_NODE_ID, flow_=self
        )

    def add_processor_node(
        self, processor: BaseDataProcessor, inputs: InputRefs
    ) -> OutputRefs:
        """Add a processor node to the graph.

        This method adds a processor node to the graph and creates the
        necessary edges to define the data flow from input nodes to this
        processor.

        Args:
            processor (BaseDataProcessor): The processor to add.
            inputs (InputRefs): The input references for the processor.

        Returns:
            OutputRefs: The output references produced by the processor.

        Raises:
            RuntimeError: If input references are not from this data flow.
            AssertionError: If the graph does not contain a source node.
            AssertionError: If input references are not of the expected type
            AssertionError: If input features do not match the output features
                of the referred node.
        """
        # make sure the input refs match the processor
        assert SRC_NODE_ID in self, "No source node in graph."
        assert isinstance(inputs, processor._in_refs_type), (
            f"Expected input references of type {processor._in_refs_type}, "
            f"but got {type(inputs)}"
        )

        # add processor to graph
        depth = -1
        node_id = self.number_of_nodes()
        # TODO: seems unintuitive that the output refs are
        #       created here and not in the call method of
        #       the processor
        outputs = processor._out_refs_type(processor.config, inputs, node_id)
        self.add_node(
            node_id,
            **{
                DataFlowGraph.NodeProperty.PROCESSOR: processor,
                DataFlowGraph.NodeProperty.FEATURES: outputs.feature_,
                DataFlowGraph.NodeProperty.DEPTH: -1,
            },
        )
        # add dependency edges to graph
        for name, ref in inputs.named_refs.items():
            # make sure the inputs come from this flow
            if ref.flow_ != self:
                raise RuntimeError(
                    "Input reference does not belong to this data flow."
                )
            # make sure the input is a valid output of the referred node
            assert ref.node_id_ in self
            assert (
                ref.key_.index_features(self.nodes[ref.node_id_]["features"])
                is not None
            )
            # add edge to other nodes
            x = self.add_edge(
                ref.node_id_,
                node_id,
                key=name,
                **{
                    DataFlowGraph.EdgeProperty.NAME: name,
                    DataFlowGraph.EdgeProperty.KEY: ref.key_,
                },
            )
            # update the depth of the node based on the depth of the source node
            depth = max(
                depth,
                self.nodes[ref.node_id_][DataFlowGraph.NodeProperty.DEPTH] + 1,
            )

        # update the depth of the node
        self.nodes[node_id][DataFlowGraph.NodeProperty.DEPTH] = depth
        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return outputs

    def dependency_graph(self, node: int) -> DataFlowGraph:
        """Generate the dependency subgraph for a given node.

        This method generates a subgraph containing all nodes that the given
        node depends on directly or indirectly.

        Args:
            node (int): The node ID for which to generate the dependency graph.

        Returns:
            DataFlowGraph: A subgraph representing the dependencies.
        """
        visited = set()
        nodes = set([node])
        # search through dependency graph
        while len(nodes) > 0:
            node = nodes.pop()
            visited.add(node)
            nodes.update(self.predecessors(node))

        return self.subgraph(visited)


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
        src_index_hash = hash(tuple(index))

        self.outputs = {SRC_NODE_ID: batch}
        self.index_hash = {SRC_NODE_ID: src_index_hash}
        self.index_lookup = {src_index_hash: index}

        self.rank = rank
        self.ready = {
            node_id: asyncio.Event()
            for node_id in graph.nodes()
            if node_id != SRC_NODE_ID
        }

        self.graph = graph

    async def wait_for(self, node_id: int) -> None:
        """Wait until the specified node is ready.

        Args:
            node_id (int): The ID of the node to wait for.
        """
        if node_id != SRC_NODE_ID:
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

    def collect_inputs(self, node_id: int) -> tuple[Batch, list[int]]:
        """Collect inputs for a given node.

        Args:
            node_id (int): The ID of the node for which to collect inputs.

        Returns:
            tuple[Batch, list[int]]: A tuple containing the collected inputs
                and the corresponding index.

        Raises:
            AssertionError: If inputs are collected from a node that is not ready.
            AssertionError: If the collected values are not of the expected type.
            AssertionError: If the collected inputs have different indexes.
        """
        inputs, index = dict(), set()
        for u, _, name, data in self.graph.in_edges(
            node_id, keys=True, data=True
        ):
            assert (u == SRC_NODE_ID) or self.ready[
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
            index.add(self.index_hash[u])

        # make sure that all collected inputs have the same index
        assert len(index) == 1, "Collected inputs have different indexes."
        index = next(iter(index))

        return inputs, self.index_lookup[index]

    def capture_output(
        self, node_id: int, output: Batch, new_index: list[int]
    ) -> None:
        """Capture the output of a node.

        Args:
            node_id (int): The ID of the node producing the output.
            output (Batch): The output batch of data.
            new_index (list[int]): The new index of the output data.

        Raises:
            AssertionError: If the node is already set
            AssertionError: If the length of output values do not match
                the length of the new index.
        """
        assert not self.ready[
            node_id
        ].is_set(), f"Node {node_id} is already set."
        assert all(
            len(vals) == len(new_index) for vals in output.values()
        ), "Output values length does not match new index length."

        self.outputs[node_id] = output

        new_index_hash = hash(tuple(new_index))
        self.index_hash[node_id] = new_index_hash
        if new_index_hash not in self.index_lookup:
            self.index_lookup[new_index_hash] = new_index

        self.ready[node_id].set()


class DataFlowExecutor(object):
    """Executes a data flow graph.

    This class provides the low-level functionality for executing a data flow
    graph, managing the execution of each node and collecting results.
    """

    def __init__(self, graph: DataFlowGraph, collect: FeatureRef) -> None:
        """Initialize the executor.

        Args:
            graph (DataFlowGraph): The data flow graph to execute.
            collect (FeatureRef): The feature reference to collect results.

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

    async def execute_node(self, node_id: int, state: ExecutionState):
        """Execute a single node in the data flow graph.

        Args:
            node_id (int): The ID of the node to execute.
            state (ExecutionState): The current execution state.
        """
        if self.graph.in_degree(node_id) > 0:
            # wait for all dependencies of the current node
            deps = self.graph.predecessors(node_id)
            futures = map(state.wait_for, deps)
            await asyncio.gather(*futures)

        # collect inputs for processor execution
        inputs, index = state.collect_inputs(node_id)
        processor = self.graph.nodes[node_id]["processor"]
        # run processor and capture output in execution state
        out, out_index = await processor.batch_process(
            inputs, index, state.rank
        )
        state.capture_output(node_id, out, out_index)

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
                if node_id != SRC_NODE_ID
            ]
        )
        # collect output values
        return state.collect_value(self.collect)


class DataFlow(object):
    """High-level interface for defining and executing data processing workflows.

    The DataFlow class allows users to create and manage directed acyclic graphs
    (DAGs) of data processors, facilitating complex data transformations and
    processing pipelines. Users can easily define source features, build sub-flows
    for specific outputs, and apply these workflows to batches of data or entire
    HuggingFace datasets.

    This class integrates various components such as the data flow graph and the
    executor to provide a seamless experience for processing data. It handles the
    internal state management, execution scheduling, and data flow dependencies to
    ensure efficient and accurate data processing.

    Properties:
        width: Returns the width of the underlying graph structure.
        depth: Returns the depth of the underlying graph structure.
        src_features: Returns the reference to the source features.
        out_features: Returns the reference to the output features, raising an
            error if not set.

    Methods:
        build: Constructs a sub-data flow for specified output features.
        batch_process: Processes a single batch of data.
        apply: Applies the data flow to an entire dataset.
        plot: Plot the data flow to a `matplotlib.pyplot` axes.

    Example:
        Define a data flow for processing text data:

        .. code-block:: python

            text_features = datasets.Features({"text": datasets.Value("string")})
            data_flow = DataFlow(features=text_features)

            # Define a processing step to tokenize text
            tokenizer = TokenizerProcessor(model_name="bert-base-uncased")
            tokenized_features = tokenizer.output_features

            # Add the tokenizer processor to the data flow
            data_flow.add_processor_node(tokenizer, inputs=data_flow.src_features)

            # Apply the data flow to a dataset
            processed_dataset = data_flow.apply(dataset)
    """

    def __init__(self, features: datasets.Features) -> None:
        """Initialize the DataFlow.

        Args:
            features (datasets.Features): The features of the source node.
        """
        # create graph
        self._graph = DataFlowGraph()
        self._src_features = self._graph.add_source_node(features=features)
        self._out_features: None | FeatureRef = None

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return self._graph.depth

    @property
    def width(self) -> int:
        """Computes the maximum width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        return self._graph.width

    @property
    def src_features(self) -> FeatureRef:
        """Get the source features.

        Returns:
            FeatureRef: The reference to the source features.
        """
        return self._src_features

    @property
    def out_features(self) -> FeatureRef:
        """Get the output features.

        Returns:
            FeatureRef: The reference to the output features.

        Raises:
            RuntimeError: If the output features have not been set.
        """
        if self._out_features is None:
            raise RuntimeError("Output features have not been set.")
        return self._out_features

    def build(self, collect: FeatureRef) -> DataFlow:
        """Build a sub-data flow to compute the requested output features.

        Args:
            collect (FeatureRef): The feature reference to collect.

        Returns:
            DataFlow: The sub-data flow.

        Raises:
            TypeError: If the collect feature is not of type
                datasets.Features or dict.
        """
        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError(
                f"Expected collect feature of type datasets.Features or dict, "
                f"but got {type(collect.feature_)}"
            )

        sub_graph = self._graph.dependency_graph(collect.node_id_)
        # build sub_flow
        # TODO: restrict input features to only the
        #       ones required by the sub-graph
        flow = DataFlow(self.src_features.feature_)
        flow._out_features = collect
        flow._graph = sub_graph

        return flow

    def batch_process(
        self, batch: Batch, index: list[int], rank: None | int = None
    ) -> Batch:
        """Process a batch of data.

        Args:
            batch (Batch): The batch of data to process.
            index (list[int]): The index of the batch.
            rank (None | int): The rank of the process in a distributed setting.

        Returns:
            Batch: The processed batch of data.

        Raises:
            AssertionError: If the output features have not been set.
        """
        assert self._out_features is not None

        if rank is None:
            # try to get multiprocessing rank from pytorch worker info
            worker_info = get_worker_info()
            rank = 0 if worker_info is None else worker_info.id

        # create a data flow executor
        executor = DataFlowExecutor(self._graph, self._out_features)
        future = executor.execute(batch, index, rank)
        # schedule the execution for the current batch
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(future)

    def _batch_process_to_pyarrow(
        self,
        batch: Batch,
        index: list[int],
        rank: None | int = None,
    ) -> pa.Table:
        """Process a batch of data and convert to a PyArrow table.

        Args:
            batch (Batch): The batch of data to process.
            index (list[int]): The index of the batch.
            rank (None | int): The rank of the process in a
                multiprocessing setting.

        Returns:
            pa.Table: The processed data as a PyArrow table.
        """
        # convert to pyarrow table with correct schema
        return pa.table(
            data=self.batch_process(batch, index, rank),
            schema=convert_features_to_arrow_schema(
                self._out_features.feature_
            ),
        )

    def apply(self, ds: D, collect: None | FeatureRef = None, **kwargs) -> D:
        """Apply the data flow to a dataset.

        Args:
            ds (D): The dataset to process.
            collect (None | FeatureRef): The feature reference to collect.
                If None, uses current output features.
            **kwargs: Additional arguments for dataset mapping. For more
                information please refer to the HuggingFace documentation
                of the Datasets.map function for the respective dataset type.

        Returns:
            D: The processed dataset.

        Raises:
            ValueError: If the dataset type is not supported.
            TypeError: If the dataset features do not match the source features.
            ValueError: If the output features have not been set.
        """
        # get the dataset features
        if isinstance(ds, (datasets.Dataset, datasets.IterableDataset)):
            features = ds.features
        elif isinstance(
            ds, (datasets.DatasetDict, datasets.IterableDatasetDict)
        ):
            features = next(iter(ds.values())).features
        else:
            raise ValueError(
                "Expected one of `datasets.Dataset`, `datasets.DatasetDict`, "
                "`datasets.IterableDataset` or `datasets.IterableDatasetDict`,"  # noqa: E501
                "got %s" % type(ds)
            )

        if (features is not None) and not check_feature_equals(
            features, self.src_features.feature_
        ):
            # TODO: should only check whether the features are present
            #       i.e. they should be a subset and don't need to match exactly
            raise TypeError("Dataset features do not match source features.")

        # build the sub data flow required to compute the requested output features
        flow = self if collect is None else self.build(collect=collect)

        if flow._out_features is None:
            raise ValueError("Output features have not been set.")

        # run data flow
        ds = flow._internal_apply(ds, **kwargs)

        if isinstance(
            ds, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # set output features for lazy datasets manually
            if isinstance(ds, datasets.IterableDataset):
                ds.info.features = flow._out_features.feature_
            elif isinstance(ds, datasets.IterableDatasetDict):
                for split in ds.values():
                    split.info.features = flow._out_features.feature_

        return ds

    def _internal_apply(self, ds: D, **kwargs) -> D:
        """(Internal) Apply the data flow to a dataset.

        Args:
            ds (D): The dataset to process.
            **kwargs: Additional arguments for dataset mapping. For more
                information please refer to the HuggingFace documentation
                of the Datasets.map function for the respective dataset type.

        Returns:
            D: The processed dataset.
        """
        # required settings
        kwargs["batched"] = True
        kwargs["with_indices"] = True
        # for non-iterable datasets the map function provide the rank
        if isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            kwargs["with_rank"] = True

        if isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            # use pyarrow table as output format for in-memory
            # datasets that support caching since it includes
            # the output feature information
            return ds.map(self._batch_process_to_pyarrow, **kwargs)

        elif isinstance(
            ds, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # iterable dataset class doesn't support pyarrow
            # outputs in map function, but it also doesn't cache
            # and thus doesn't need the features while processing
            return ds.map(
                self.batch_process,
                remove_columns=set(self.src_features.feature_.keys())
                - set(self._out_features.feature_.keys()),
                **kwargs,
            )

    def plot(
        self,
        with_edge_labels: bool = True,
        edge_label_format: str = "{name}={key}",
        src_node_label: str = "[ROOT]",
        edge_font_size: int = 6,
        node_font_size: int = 6,
        node_size: int = 5_000,
        arrowsize: int = 25,
        ax: None | plt.Axes = None,
    ) -> plt.Axes:
        """Plot the data flow graph.

        Args:
            with_edge_labels (bool): Whether to include labels on the edges. Defaults to True.
            edge_label_format (str): Format string for edge labels. Defaults to "{name}={key}".
            src_node_label (str): Label for the source node. Defaults to "[ROOT]".
            edge_font_size (int): The font size for edge labels. Defaults to 6.
            node_font_size (int): The font size for node labels. Defaults to 6.
            node_size (int): The size of the nodes. Defaults to 5_000.
            arrowsize (int): The size of the arrows on the edges. Defaults to 25.
            ax (Optional[plt.Axes]): Matplotlib axes object to draw the plot on. Defaults to None.

        Returns:
            plt.Axes: The Matplotlib axes object with the plot.
        """
        # create a plot axes
        if ax is None:
            _, ax = plt.subplots(
                1, 1, figsize=(self.depth * 2, self.width * 2.5)
            )

        # compute the node positions
        pos = nx.multipartite_layout(
            self._graph, subset_key=DataFlowGraph.NodeProperty.DEPTH
        )

        # plot the raw graph
        nx.draw(
            self._graph,
            pos,
            with_labels=False,
            node_size=node_size,
            arrowsize=arrowsize,
            ax=ax,
        )

        # limit the maximum number of character in a single line in nodes
        max_line_length = node_size // (node_font_size * 65)

        node_labels = {}
        # build node labels
        for node, data in self._graph.nodes(data=True):
            if node == SRC_NODE_ID:
                # add root node label
                node_labels[node] = src_node_label

            else:
                # get the processor type name of the current node
                proc = data[DataFlowGraph.NodeProperty.PROCESSOR]
                node_label = type(proc).__name__
                # split string into words
                words = re.split(r"(?<=[a-z])(?=[A-Z])", node_label)
                # group words such that each group has a limited number of
                # characers
                lengths = np.cumsum(list(map(len, words))) // max_line_length
                groups = groupby(range(len(words)), key=lengths.__getitem__)
                # join groups with newlines inbetween
                node_labels[node] = "\n".join(
                    ["".join([words[i] for i in group]) for _, group in groups]
                )

        # add node labels
        nx.draw_networkx_labels(
            self._graph,
            pos,
            labels=node_labels,
            font_size=node_font_size,
            ax=ax,
        )

        # add edge labels
        if with_edge_labels:
            # group multi-edges by their source and target nodes
            grouped_edges = groupby(
                sorted(
                    self._graph.edges(data=True), key=lambda e: (e[0], e[1])
                ),
                key=lambda e: (e[0], e[1]),
            )

            edge_labels = {}
            # build edge labels
            for edge, group in grouped_edges:
                edge_labels[edge] = "\n".join(
                    [
                        edge_label_format.format(
                            name=data[DataFlowGraph.EdgeProperty.NAME],
                            key=str(data[DataFlowGraph.EdgeProperty.KEY]),
                        )
                        for _, _, data in group
                    ]
                )

            # draw the edge labels
            nx.draw_networkx_edge_labels(
                self._graph,
                pos,
                edge_labels=edge_labels,
                font_size=edge_font_size,
                ax=ax,
            )

        return ax
