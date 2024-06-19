"""Defines the structure of data flow graphs and their components.

This module provides the :class:`DataFlowGraph` class and related components, which represent
the structure of a data processing workflow. The graph consists of nodes (data processors)
and edges (data flow between processors).
"""


from __future__ import annotations

import uuid
from enum import Enum
from functools import wraps
from itertools import groupby
from typing import Any

import datasets
import networkx as nx
from datasets.features.features import FeatureType

from hyped.data.flow.core.nodes.aggregator import BaseDataAggregator
from hyped.data.flow.core.nodes.base import BaseNode
from hyped.data.flow.core.nodes.const import Const
from hyped.data.flow.core.nodes.processor import BaseDataProcessor
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.outputs import OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class DataFlowGraph(nx.MultiDiGraph):
    """A multi-directed graph representing a data flow of data processors.

    This class is used internally to define a directed acyclic graph (DAG)
    where nodes represent data processors of `BaseDataProcessor` type, and
    edges define the data flow between these processors.
    """

    class GraphProperty(str, Enum):
        """Enum representing properties of the data flow graph."""

        SRC_NODE_ID = "src_node_id"
        """
        Property representing the source node ID.

        This property identifies the source node ID of an edge in the graph.
        """

    class PredefinedPartition(str, Enum):
        """Enum representing predefined partitions in the data flow graph."""

        CONST = "CONSTANT"
        """
        Represents a partition containing all constant nodes.

        This partition is predefined to include nodes that hold constant
        values used in the data processing flow.
        """

        DEFAULT = "DEFAULT"
        """
        Represents the default partition for nodes.

        This partition is assigned to the source node and is inherited by
        its sub-graph.
        """

        AGGREGATED = "AGGREGATED"

    class NodeType(Enum):
        """Enum representing types of nodes in the data flow graph."""

        SOURCE = "SOURCE_NODE"
        """
        Represents a source node in the data flow graph.

        This type of node acts as the starting point of the data flow graph,
        typically representing raw input data sources.
        """

        CONST = "CONST_NODE"
        """
        Represents a constant node in the data flow graph.

        This type of node introduces constant values into the data flow, serving
        as fixed inputs to the subsequent processing stages.
        """

        DATA_PROCESSOR = "DATA_PROCESSOR_NODE"
        """
        Represents a data processor node in the data flow graph.

        This type of node represents a data processing component within the
        data flow graph. Data processors perform specific transformations
        on input data and produce output data based on defined processing logic.
        """

        DATA_AGGREGATOR = "DATA_AGGREGATOR_NODE"
        """
        Represents a data aggregator node in the data flow graph.

        This type of node is responsible for aggregating data from multiple
        sources or processing stages within the data flow graph. Aggregator
        nodes typically perform dataset-wide computations or combine data
        from different sources into a unified representation.
        """

    class NodeAttribute(str, Enum):
        """Enum representing properties of a node in the data flow graph."""

        NODE_OBJ = "node_object"
        """
        The object associated with the node.
        
        The value of this property is dependent on the type of node. For
        nodes of type :class:`NodeType.DATA_PROCESSOR`, this property
        refers to the processor instance of the node. 
        """

        NODE_TYPE = "node_type"
        """
        Indicates the type of node.

        Type: :class:`NodeType`

        This property indicates the type of data processor. It helps in categorizing
        and identifying the nature of the processor in the data flow graph.
        """

        IN_FEATURES = "in_features"
        """
        Represents the input features associated with the node.

        Type: :class:`datasets.Features`

        This property contains the input features required by the data processor,
        encapsulated in a HuggingFace `datasets.Features` instance. It defines the
        structure and types of data that is expected by this node.
        """

        OUT_FEATURES = "out_features"
        """
        Represents the output features associated with the node.

        Type: :class:`datasets.Features`

        This property contains the output features produced by the data processor,
        encapsulated in a HuggingFace `datasets.Features` instance. It defines the
        structure and types of data that are output by this node.
        """

        PARTITION = "partition"
        """
        Represents the partition to which the node belongs.

        Type: :code:`str`

        This property indicates the specific partition of the data flow graph that
        the node is part of, which can be used to group nodes by different semantics.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: :class:`int`

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    class EdgeAttribute(str, Enum):
        """Enum representing properties of an edge in the data flow graph."""

        NAME = "name"
        """
        Represents the name of the edge.

        Type: :class:`str`

        This property corresponds to the keyword of the argument used as an input
        to the processor, linking the edge to a specific input parameter.

        The name is also used by NetworkX as an identifier to distinguish multiedges
        between a pair of nodes. It serves as a unique identifier for the edge.
        """

        KEY = "feature_key"
        """
        Represents the key of the feature associated with the edge.

        Type: :class:`FeatureKey`

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
                    self, DataFlowGraph.NodeAttribute.DEPTH
                ).values()
            )
            + 1
        )

    @wraps(nx.MultiDiGraph)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DataFlowGraph.

        Args:
            *args: Positional arguments forwarded to the init of the :class:`MultiDiGraph`.
            **kwargs: Keyword arguments forwarded to the init of the :class:`MultiDiGraph`.
        """
        super(DataFlowGraph, self).__init__(*args, **kwargs)

        # set default source node id
        if DataFlowGraph.GraphProperty.SRC_NODE_ID not in self.graph:
            self.graph[DataFlowGraph.GraphProperty.SRC_NODE_ID] = None

        # reset source node id for subgraphs
        if (self.src_node_id is not None) and (self.src_node_id not in self):
            self.graph[DataFlowGraph.GraphProperty.SRC_NODE_ID] = None

    @property
    def src_node_id(self) -> str:
        """Get the source node ID of the data flow graph.

        This property returns the source node ID associated with the data flow graph.
        The source node is the entrypoint for inputs to the data flow.

        Returns:
            str: The uuid of the source node.
        """
        return self.graph[DataFlowGraph.GraphProperty.SRC_NODE_ID]

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
        depths = nx.get_node_attributes(
            self, DataFlowGraph.NodeAttribute.DEPTH
        )
        layers = groupby(sorted(self, key=depths.get), key=depths.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def add_source_node(
        self, features: datasets.Features, node_id: None | str = None
    ) -> int:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            features (datasets.Features): The features of the source node.
            node_id (None | str): The id of the node, defaults to a random uuid.

        Returns:
            FeatureRef: A reference to the input features.

        Raises:
            AssertionError: If the graph already contains a source node
        """
        # make sure the graph has no source node yet
        if self.src_node_id is not None:
            raise RuntimeError("Graph already contains a source node.")
        # add the source node and set the source node id in the graph properties
        node_id = self.add_processor_node(None, None, features, node_id)
        self.graph[DataFlowGraph.GraphProperty.SRC_NODE_ID] = node_id
        # return the source node id
        return node_id

    # TODO: rename to more generic 'add_node'
    def add_processor_node(
        self,
        obj: BaseNode,
        inputs: None | InputRefs,
        output_features: datasets.Features,
        node_id: None | str = None,
    ) -> str:
        """Add a processor node to the data flow graph.

        This method adds a processor node to the data flow graph and creates the necessary edges
        to define the data flow from input nodes to this processor.

        Args:
            obj (BaseNode): The node object.
            inputs (None | InputRefs): The input references to the node. If None, the node will be a source node.
            output_features (datasets.Features): The output features generated by the node.
            node_id (None | str): The id of the node, defaults to a random uuid.

        Returns:
            str: The uuid of the node within the data flow graph.

        Raises:
            AssertionError: If the processor type is invalid.
            AssertionError: If the graph is cyclic after adding the new node.
            AssertionError: If the partition cannot be inferred.
            RuntimeError: If any input reference do not belong to this data flow.
            RuntimeError: If the input references are a mix of aggregated and non-aggregated features.
            NotImplementedError: If the input to a non-data-processor node is an aggregated feature.
        """
        # get processor type
        node_type = (
            DataFlowGraph.NodeType.SOURCE
            if obj is None
            else DataFlowGraph.NodeType.CONST
            if isinstance(obj, Const)
            else DataFlowGraph.NodeType.DATA_PROCESSOR
            if isinstance(obj, BaseDataProcessor)
            else DataFlowGraph.NodeType.DATA_AGGREGATOR
            if isinstance(obj, BaseDataAggregator)
            else None
        )
        # make sure the object is valid
        assert node_type is not None, f"Invalid processor type {type(obj)}."

        # make sure all input references belong to this graph
        if (inputs is not None) and any(
            ref.flow_ is not self for ref in inputs.refs
        ):
            raise RuntimeError(
                "Input reference does not belong to this data flow."
            )

        # compute the depth of the node in the graph based
        # on it's input references
        depth = (
            0
            if inputs is None
            else max(
                (
                    self.nodes[ref.node_id_][DataFlowGraph.NodeAttribute.DEPTH]
                    + 1
                    for ref in inputs.refs
                ),
                default=0,
            )
        )

        partition = None
        # infer partition of the node
        if node_type == DataFlowGraph.NodeType.SOURCE:
            # source node is added to the default partition
            partition = DataFlowGraph.PredefinedPartition.DEFAULT.value

        elif node_type == DataFlowGraph.NodeType.CONST:
            # contants are added to the constant partition
            partition = DataFlowGraph.PredefinedPartition.CONST.value

        elif inputs is not None:
            # for other node types the partition is inferred from the inputs
            candidate_partitions = set(
                [
                    self.nodes[ref.node_id_][
                        DataFlowGraph.NodeAttribute.PARTITION
                    ]
                    if self.nodes[ref.node_id_][
                        DataFlowGraph.NodeAttribute.NODE_TYPE
                    ]
                    != DataFlowGraph.NodeType.DATA_AGGREGATOR
                    else DataFlowGraph.PredefinedPartition.AGGREGATED
                    for ref in inputs.refs
                ]
            )

            if candidate_partitions == {
                DataFlowGraph.PredefinedPartition.CONST
            }:
                # if all inputs come from the constant partition, then this node
                # is also part of the constant partition
                partition = DataFlowGraph.PredefinedPartition.CONST.value

            elif (
                DataFlowGraph.PredefinedPartition.AGGREGATED
                in candidate_partitions
            ) and (
                DataFlowGraph.PredefinedPartition.DEFAULT
                in candidate_partitions
            ):
                raise RuntimeError(
                    "Cannot mix aggregated and non-aggregated features."
                )

            elif (
                DataFlowGraph.PredefinedPartition.AGGREGATED
                in candidate_partitions
            ):
                # if the inputs come directly from an aggregator or from the
                # aggregated partition, then stay in the aggregated partition
                partition = DataFlowGraph.PredefinedPartition.AGGREGATED.value

            else:
                # if any of the inputs are not from the constant partition,
                # then the node is part of the default partition
                partition = DataFlowGraph.PredefinedPartition.DEFAULT.value

        # partition could not be inferred
        assert partition is not None, (
            "Partition cannot be inferred for source nodes, "
            "i.e. nodes without any input references."
        )

        # aggregated partition currently only supports processor type nodes
        if (node_type != DataFlowGraph.NodeType.DATA_PROCESSOR) and (
            partition == DataFlowGraph.PredefinedPartition.AGGREGATED
        ):
            raise NotImplementedError(
                f"Aggregator may only be processed by data processors, got {node_type}."
            )

        # create the node id if it was not provided
        if node_id is None:
            node_id = str(uuid.uuid4())

        # add the node to the graph
        self.add_node(
            node_id,
            **{
                DataFlowGraph.NodeAttribute.NODE_OBJ: obj,
                DataFlowGraph.NodeAttribute.NODE_TYPE: node_type,
                DataFlowGraph.NodeAttribute.IN_FEATURES: (
                    None if inputs is None else inputs.features_
                ),
                DataFlowGraph.NodeAttribute.OUT_FEATURES: output_features,
                DataFlowGraph.NodeAttribute.PARTITION: partition,
                DataFlowGraph.NodeAttribute.DEPTH: depth,
            },
        )

        if inputs is not None:
            # add dependency edges to graph
            for name, ref in inputs.named_refs.items():
                # make sure the input is a valid output of the referred node
                assert ref.node_id_ in self
                assert (
                    ref.key_.index_features(
                        self.nodes[ref.node_id_][
                            DataFlowGraph.NodeAttribute.OUT_FEATURES
                        ]
                    )
                    is not None
                )
                # add the edge
                self.add_edge(
                    ref.node_id_,
                    node_id,
                    key=name,
                    **{
                        DataFlowGraph.EdgeAttribute.NAME: name,
                        DataFlowGraph.EdgeAttribute.KEY: ref.key_,
                    },
                )

        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return node_id

    def get_node_output_ref(self, node_id: str) -> FeatureRef | OutputRefs:
        """Retrieves the output reference for a given node in the data flow graph.

        This method returns an appropriate output reference based on the type of the node specified by the
        given node ID. The method constructs the appropriate reference object based on the node type:

            - For source nodes, this method builds a feature reference using the output features of the node.
            - For data processor nodes, it retrieves the processor's output references type and constructs the full output reference.
            - For data aggregator nodes, it builds a data aggregation reference using the node's value type.

        Args:
            node_id (str): The ID of the node for which to retrieve the output reference.

        Returns:
            FeatureRef | OutputRefs: The output reference associated with the specified node.

        Raises:
            KeyError: If the node ID does not exist in the data flow graph.
            TypeError: If the node type is not recognized.
        """
        if node_id not in self:
            raise KeyError(
                f"Node ID {node_id} does not exist in the data flow graph."
            )

        # get node properties
        node = self.nodes[node_id]
        node_obj = node[DataFlowGraph.NodeAttribute.NODE_OBJ]
        node_type = node[DataFlowGraph.NodeAttribute.NODE_TYPE]
        features = node[DataFlowGraph.NodeAttribute.OUT_FEATURES]

        if node_type == DataFlowGraph.NodeType.SOURCE:
            # build a feature reference to the source features of the graph
            features = node[DataFlowGraph.NodeAttribute.OUT_FEATURES]
            return FeatureRef(
                key_=tuple(), node_id_=node_id, flow_=self, feature_=features
            )

        # build the output references object
        assert isinstance(node_obj, BaseNode)
        return node_obj._out_refs_type(self, node_id, features)

    def dependency_graph(self, nodes: set[int]) -> DataFlowGraph:
        """Generate the dependency subgraph for a given node.

        This method generates a subgraph containing all nodes that the given
        set of nodes depend on directly or indirectly.

        Args:
            nodes (set[int]): The node IDs for which to generate the dependency graph.

        Returns:
            DataFlowGraph: A subgraph representing the dependencies.
        """
        visited = set()
        nodes = nodes.copy()
        # search through dependency graph
        while len(nodes) > 0:
            node = nodes.pop()
            visited.add(node)
            nodes.update(self.predecessors(node))

        return self.subgraph(visited)

    def get_partition(self, partition: str) -> DataFlowGraph:
        """Extract a subgraph containing only nodes from a specific partition.

        This method creates a subgraph from the current graph by selecting
        nodes that belong to a specified partition.

        Args:
            partition (str): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph containing nodes of the specified partition.
        """
        # get all nodes to the provided partition
        partition = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] == partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(partition)

    def drop_partition(self, partition: str) -> DataFlowGraph:
        """Drop a specified partition from the graph.

        This method creates a subgraph from the current graph by excluding
        nodes that belong to a specified partition.

        Args:
            partition (str): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph excluding nodes of the specified partition.
        """
        # get all nodes to the provided partition
        remainder = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] != partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(remainder)

    def subgraph_in_edges(
        self, subgraph: DataFlowGraph, data: bool | EdgeAttribute = False
    ) -> list[tuple[int, int, str] | tuple[int, int, str, Any]]:
        """Get incoming edges to a subgraph from nodes outside the subgraph.

        This method returns a list of edges that point to nodes within the
        specified subgraph from nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.
            data (bool | EdgeAttribute): Whether to include edge data. If set to
                :code:`True` or an :class:`EdgeAttribute`, the method returns edges with data.

        Returns:
            list[tuple[int, int, str] | tuple[int, int, str, Any]]: The incoming edges
            to the subgraph.
        """
        return [
            e
            for e in self.in_edges(subgraph, keys=True, data=data)
            if e[0] not in subgraph
        ]

    def subgraph_out_edges(
        self, subgraph: DataFlowGraph, data: bool | EdgeAttribute = False
    ) -> list[tuple[int, int, str] | tuple[int, int, str, Any]]:
        """Get outgoing edges from a subgraph to nodes outside the subgraph.

        This method returns a list of edges that point from nodes within the
        specified subgraph to nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.
            data (bool | EdgeAttribute): Whether to include edge data. If set to
                :code:`True` or an :class:`EdgeAttribute`, the method returns edges with data.

        Returns:
            list[tuple[int, int, str] | tuple[int, int, str, Any]]: The outgoing edges
            from the subgraph.
        """
        return [
            e
            for e in self.out_edges(subgraph, keys=True, data=data)
            if e[1] not in subgraph
        ]

    def recompute_depths(self) -> None:
        """Recompute the depth of all nodes in the data flow graph.

        This method recalculates the depth of each node based on the topological
        order of the graph. The depth of a node is defined as the length of the
        longest path from the source node to the node.
        """
        for node_id in nx.topological_sort(self):
            self.nodes[node_id][DataFlowGraph.NodeAttribute.DEPTH] = max(
                (
                    self.nodes[in_node_id][DataFlowGraph.NodeAttribute.DEPTH]
                    + 1
                    for in_node_id, _ in self.in_edges(node_id)
                ),
                default=0,
            )
