"""Module for Data Flow Graph Optimization.

This module provides an optimizer for data flow graphs, which applies various optimization
techniques to improve program efficiency and reduce redundant computations.

The optimizer module includes methods for optimizing data flow graphs, such as:

1. **Prune Redundant Nodes**: The optimizer prunes the data flow graph, removing nodes that
   do not contribute to producing the desired output.

2. **Constant Expression Evaluation**: Pre-computes the constant partition of the graph, which consists
   solely of constant values and has no dependencies on other parts of the graph, and replaces
   these constants with their computed values.

3. **Common Subexpression Elimination (CSE)**: Identifies and eliminates redundant computations
   by recognizing and reusing common subexpressions in the data flow graph.

4. **Constant Folding/Propagation**: Evaluates constant expressions at compile time and replaces
   them with their computed values to simplify the data flow graph.

The DataFlowGraphOptimizer class within this module provides these optimization methods,
which can be applied individually or in combination to optimize a given data flow graph.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import groupby

from datasets import Features, Value

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.processors.ops.collect import (
    CollectFeatures,
    NestedContainer,
)

from .executor import DataFlowExecutor
from .graph import DataFlowGraph
from .nodes.const import Const
from .refs.inputs import InputRefs
from .refs.ref import FeatureRef


class DataFlowGraphOptimizer(object):
    """Optimizer for Data Flow Graphs.

    This optimizer applies several optimization techniques to the given data flow graph
    in order to improve its efficiency and reduce redundant computations. The optimization
    steps include:

    1. Prune Unnecessary Nodes
    2. Constant Expressions Evaluation
    3. Common Subexpression Elimination (CSE)
    4. Constant Folding/Propagation

    Note the difference between constant expression evaluation and constant folding. Constant expression evaluation
    focuses on evaluating the constant partition of the graph, which consists solely of constant values and has no
    dependencies on other parts of the graph. In contrast, constant folding aims to simplify expressions that include
    both constant and non-constant values by precomputing the constant parts of these expressions.
    """

    def cse(
        self, graph: DataFlowGraph
    ) -> tuple[DataFlowGraph, dict[int, int]]:
        """Performs Common Subexpression Elimination (CSE) on the data flow graph.

        This method performs Common Subexpression Elimination (CSE) on the given data flow graph.
        It optimizes the graph by identifying and eliminating redundant computations.

        Consider the following data flow graph:

        .. code-block:: python

            x = a + b
            y = c * d
            z = a + b

        After applying Common Subexpression Elimination (CSE), the redundant computation
        :code:`a + b` is eliminated, resulting in the following optimized graph:

        .. code-block:: python

            x = a + b
            y = c * d
            z = x

        The node IDs before and after optimization are mapped as follows:
        :code:`{0: 0, 1: 1, 2: 0}`

        Args:
            graph (DataFlowGraph): The data flow graph.

        Returns:
            tuple[DataFlowGraph, dict[int, int]]: The optimized data flow graph and a mapping
                of node IDs before and after optimization.
        """

        @dataclass
        class _CSE_NodeIdentifier(object):
            """Helper class to identify redundant nodes."""

            node_type: DataFlowGraph.NodeType
            node_config: str
            in_edge_identifiers: list[tuple[int, str, FeatureKey]]
            node_id: str = field(default=None, compare=False)

        cse_graph = DataFlowGraph()
        # maps nodes of the original graph to the nodes in the cse-graph
        # this is a non-injective function as multiple nodes in the original
        # graph can be mapped to the same target node during optimization
        node_mapping = dict()

        key = lambda n: graph.nodes[n][DataFlowGraph.NodeAttribute.DEPTH]
        for _, layer in groupby(sorted(graph, key=key), key=key):
            cse_layer = []

            for node_id in layer:
                # build identifiers for incoming edges with
                # source nodes mapped to nodes in optimized graph
                in_edge_identifiers = [
                    (
                        node_mapping[src_node_id],
                        edge_data[DataFlowGraph.EdgeAttribute.NAME],
                        edge_data[DataFlowGraph.EdgeAttribute.KEY],
                    )
                    for src_node_id, _, edge_data in graph.in_edges(
                        node_id, data=True
                    )
                ]

                node_data = graph.nodes[node_id]
                obj = node_data[DataFlowGraph.NodeAttribute.NODE_OBJ]
                # create cse node identifier
                identifier = _CSE_NodeIdentifier(
                    node_type=node_data[DataFlowGraph.NodeAttribute.NODE_TYPE],
                    node_config=getattr(obj, "config", None),
                    in_edge_identifiers=in_edge_identifiers,
                )

                # do not add a new node if the node is already present in the layer
                if identifier in cse_layer:
                    identifier = cse_layer[cse_layer.index(identifier)]
                    node_mapping[node_id] = identifier.node_id

                else:
                    # read node feature properties
                    in_features = node_data[
                        DataFlowGraph.NodeAttribute.IN_FEATURES
                    ]
                    out_features = node_data[
                        DataFlowGraph.NodeAttribute.OUT_FEATURES
                    ]

                    if obj is None:
                        # add source node to optimized graph
                        identifier.node_id = cse_graph.add_source_node(
                            out_features, node_id=node_id
                        )

                    else:
                        # build input references object if expected
                        if in_features is not None:
                            named_refs = {
                                name: cse_graph.get_node_output_ref(
                                    src_node_id
                                )[key]
                                for src_node_id, name, key in in_edge_identifiers
                            }

                            if isinstance(obj, CollectFeatures):
                                # collect features processor is a special case because
                                # the input reference object doesn't directly take the
                                # incoming edges as input arguments but requires them
                                # to be wrapped by a container
                                collection = NestedContainer[FeatureRef](
                                    data=named_refs
                                )
                                inputs = obj._in_refs_type(
                                    collection=collection
                                )
                            else:
                                # other nodes expect the incoming edge data as arguments
                                inputs = obj._in_refs_type(**named_refs)

                        else:
                            # the node doesn't expect any inputs, i.e. it is a source node
                            assert obj._in_refs_type is type(None)
                            inputs = None

                        # add the node to the optimized graph
                        identifier.node_id = cse_graph.add_processor_node(
                            obj, inputs, out_features, node_id=node_id
                        )

                    # update cse layer and node id mapping
                    cse_layer.append(identifier)
                    # the node keeps the same id
                    node_mapping[node_id] = node_id

        return cse_graph

    def constant_evaluation(self, graph: DataFlowGraph) -> DataFlowGraph:
        """Pre-computes the constant partition of the data flow graph.

        This method evaluates the constant partition of the data flow graph,
        which is self-contained and has no outside dependencies. It creates a
        new graph from this partition, collects all outputs, and executes them
        using a data flow executor. The evaluated constants are then re-inserted
        into the main graph with their computed values.

        Consider the following example:

        .. code-block:: python

            x, y = 1, 2
            z = x + y

        After applying constant evaluation, the above is expression is precomputed to

        .. code-block:: python

            z = 3

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.

        Returns:
            DataFlowGraph: The optimized data flow graph with evaluated constants.
        """
        # get the constant partition of the graph and make sure
        # the sub-flow is self-contained, i.e. has no outside dependencies
        const_graph = graph.get_partition(
            DataFlowGraph.PredefinedPartition.CONST
        )
        assert len(graph.subgraph_in_edges(const_graph)) == 0

        # check if there is anything to optimize in the constant partition
        # there are operations to collapse only if there are any edges within
        # the constant graph, otherwise the constant graph is either empty or
        # all nodes in the constant partition are source nodes which cannot
        # be optimized further
        if len(const_graph.edges) > 0:
            # create a new graph from the view
            # and add a source node with no features
            const_graph = DataFlowGraph(const_graph)
            const_graph.add_source_node(Features({"x": Value("int32")}))

            # collect the outputs of all nodes
            collect = NestedContainer[FeatureRef](
                data={
                    i: const_graph.get_node_output_ref(i)
                    for i in const_graph.nodes()
                }
            )
            collect = CollectFeatures().call(collection=collect)

            # create an executor for the constant partition
            executor = DataFlowExecutor(
                graph=const_graph, collect=collect, aggregation_manager=None
            )

            # execute the constant partition
            loop = asyncio.new_event_loop()
            future = executor.execute({"x": [0]}, index=[0], rank=0)
            out = loop.run_until_complete(future)["collected"][0]
            # close the event loop
            loop.close()

            # get usage of constants in the graph
            const_edges = graph.subgraph_out_edges(const_graph, data=True)

            # drop the constant partition in the original graph
            graph = graph.drop_partition(
                DataFlowGraph.PredefinedPartition.CONST
            )
            graph = DataFlowGraph(graph)

            const_lookup = dict()
            # add all required constants
            for const_node_id, tgt_node_id, key, data in const_edges:
                key = data.pop(DataFlowGraph.EdgeAttribute.KEY)

                if (const_node_id, key) not in const_lookup:
                    # get the feature type of the constant referenced by the edge
                    feature = const_graph.nodes[const_node_id][
                        DataFlowGraph.NodeAttribute.OUT_FEATURES
                    ]
                    dtype = key.index_features(feature)
                    # get the constant value referenced by the edge
                    value = key.index_example(out[const_node_id])
                    # create a new constant and add it to the data flow
                    const = Const(value=value, dtype=dtype)
                    ref = const.to(graph).value
                    # add the reference to the constants lookup
                    const_lookup[(const_node_id, key)] = ref

                # get the reference object from the lookup
                ref = const_lookup[(const_node_id, key)]
                # add the edge to the graph
                graph.add_edge(
                    ref.node_id_,
                    tgt_node_id,
                    key=key,
                    **{
                        DataFlowGraph.EdgeAttribute.NAME: data[
                            DataFlowGraph.EdgeAttribute.NAME
                        ],
                        DataFlowGraph.EdgeAttribute.KEY: ref.key_,
                    },
                )

        return graph

    def constant_folding(self, graph: DataFlowGraph) -> DataFlowGraph:
        """Performs constant folding optimization on the data flow graph.

        Constant folding is an optimization technique used to evaluate constant expressions
        at compile time and replace them with their computed values. This method traverses
        the data flow graph and identifies expressions involving constants that can be
        evaluated statically. It then replaces these expressions with their computed values,
        eliminating redundant computations and simplifying the graph.

        Consider the following example:

        .. code-block:: python

            z = x + (-y)

        After applying constant folding optimization, the expression is simpified to

        .. code-block:: python

            z = x - y

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.

        Returns:
            DataFlowGraph: The optimized data flow graph after constant folding.
        """
        return graph

    def optimize(
        self, graph: DataFlowGraph, leaf_nodes: set[str]
    ) -> DataFlowGraph:
        """Optimizes the data flow graph for a specified set of leaf nodes.

        Args:
            graph (DataFlowGraph): The data flow graph.
            leaf_nodes (set[str]): Set of leaf node IDs.

        Returns:
            DataFlowGraph: The optimized data flow graph.

        Raises:
            AssertionError: If not all leaf nodes are contained in the optimized graph.
        """
        # build dependency graph for the given set of nodes
        graph = graph.dependency_graph(leaf_nodes)

        # evaluate all constants
        graph = self.constant_evaluation(graph)

        # apply common sub-expresison elimination
        graph = self.cse(graph)

        # apply constant folding/propagation
        graph = self.constant_folding(graph)

        # recompute all depths after optimiztion
        graph.recompute_depths()

        # make sure all leaf nodes are present in the optimized graph
        assert all(node_id in graph for node_id in leaf_nodes)

        return graph
