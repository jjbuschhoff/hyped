"""Defines high-level interfaces for data processing workflows.

This module provides the :class:`DataFlow` class, which allows users to define and
execute complex data processing workflows. The workflows are represented as
directed acyclic graphs (DAGs) of data processors.
"""

from __future__ import annotations

import asyncio
import re
from functools import cached_property, partial
from itertools import groupby
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import datasets
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pyarrow as pa
from datasets.features.features import FeatureType
from matplotlib import colormaps
from torch.utils.data import get_worker_info
from typing_extensions import TypeAlias

from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.common.feature_checks import check_feature_equals
from hyped.common.lazy import LazyInstance

from .executor import DataFlowExecutor
from .graph import DataFlowGraph
from .nodes.aggregator import DataAggregationManager
from .nodes.const import Const
from .optim import DataFlowGraphOptimizer
from .refs.inputs import InputRefs
from .refs.ref import AggregationRef, FeatureRef

Batch: TypeAlias = dict[str, list[Any]]
D = TypeVar(
    "D",
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
)

# patch asyncio if running in an async environment, such as jupyter notebook
# this fixes #26
try:
    nest_asyncio._patch_asyncio()
    loop = asyncio.get_event_loop()
    nest_asyncio.apply(loop)
except ValueError:  # pragma: not covered
    # TODO: log warning
    pass


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
    """

    def __init__(self, features: datasets.Features) -> None:
        """Initialize the DataFlow.

        Args:
            features (datasets.Features): The features of the source node.
        """
        # create graph and add the source node
        self._graph = DataFlowGraph()
        self._graph.add_source_node(features)
        # lazy executor instance, set in build
        self._executor: None | LazyInstance[DataFlowExecutor] = None

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

    @cached_property
    def src_features(self) -> FeatureRef:
        """Get the source features.

        Returns:
            FeatureRef: The reference to the source features.
        """
        return FeatureRef(
            key_=tuple(),
            feature_=self._graph.nodes[self._graph.src_node_id][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ],
            node_id_=self._graph.src_node_id,
            flow_=self._graph,
        )

    @property
    def out_features(self) -> FeatureRef:
        """Get the output features.

        Returns:
            FeatureRef: The reference to the output features.

        Raises:
            RuntimeError: If the flow hasn't been build yet.
        """
        if self._executor is None:
            raise RuntimeError("Flow has not been build yet.")
        return self._executor.collect

    @property
    def aggregates(self) -> None | MappingProxyType[str, Any]:
        """Access the aggregated values computed by data aggregators.

        This property provides access to the aggregated values computed by data
        aggregators during the execution of the data flow. These aggregated values
        represent dataset-wide metrics or summary statistics calculated based on the
        input data.

        Returns:
            None | MappingProxyType[str, Any]: A read-only view of the aggregated
            values as a mapping from aggregation names to their respective values.

        Raises:
            RuntimeError: If the flow has not been built yet.
        """
        if self._executor is None:
            raise RuntimeError("Flow has not been build yet.")

        return (
            None
            if (self._executor.aggregation_manager is None)
            else self._executor.aggregation_manager.values_proxy
        )

    def const(
        self, value: Any, ftype: None | FeatureType = None
    ) -> FeatureRef:
        """Adds a constant node to the data flow graph.

        This function creates and adds a constant node to the data flow graph
        with the specified value and optionally specified feature type. It
        returns a feature reference to the constant value.

        Args:
            value (Any): The constant value to be introduced into the data flow.
            ftype (None | FeatureType, optional): The type of the feature. If not provided,
                it is inferred from the value. Defaults to None.

        Returns:
            FeatureRef: A feature reference to the constant value in the data flow.
        """
        return Const(value=value, ftype=ftype).to(self._graph).value

    def build(
        self,
        collect: FeatureRef,
        aggregators: None | dict[str, AggregationRef] = None,
    ) -> tuple[DataFlow, None | MappingProxyType[str, Any]]:
        """Build an optimized sub-data flow to compute the requested output features.

        This method constructs a sub-graph of the data flow to compute the specified
        output features. It can optionally include aggregators for dataset-wide computations.

        The constructed sub-graph undergoes optimization, including techniques such as
        common subexpression elimination (CSE) and other AST optimizations.

        Args:
            collect (FeatureRef): The feature reference to collect.
            aggregators (None | dict[str, AggregationRef], optional): Dictionary of
                aggregators for computing dataset-wide values. Defaults to None.

        Returns:
            tuple[DataFlow, None | MappingProxyType[str, Any]]: The sub-data flow and a proxy
                object of the aggregated values. The aggregated values object is None in case
                no aggregators were provided.

        Raises:
            TypeError: If the collect feature is not of type `datasets.Features` or `dict`.
            TypeError: If aggregators are provided but are not of the expected type.
            RuntimeError: If the collect feature does not belong to this flow.
            RuntimeError: If any of the aggregators does not belong to this flow.
        """
        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError(
                f"Expected collect feature of type `datasets.Features` or `dict`, "
                f"but got {type(collect.feature_)}"
            )

        if (aggregators is not None) and not (
            isinstance(aggregators, dict)
            and all(
                isinstance(ref, AggregationRef) for ref in aggregators.values()
            )
        ):
            raise TypeError(
                f"Expected aggregators to be a dictionary with values of type "
                f"`AggregationRef`, but got {type(aggregators)} with "
                f"values {aggregators.values()}"
            )

        if collect.flow_ != self._graph:
            raise RuntimeError(
                "The collect feature does not belong to the current graph."
            )

        if (aggregators is not None) and any(
            ref.flow_ != self._graph for ref in aggregators.values()
        ):
            raise RuntimeError(
                "One or more aggregators do not belong to the current graph."
            )

        # collect all requested leaf nodes
        leaf_nodes = set([collect.node_id_])
        if aggregators is not None:
            leaf_nodes.update([ref.node_id_ for ref in aggregators.values()])

        # optimize data flow graph
        optim = DataFlowGraphOptimizer()
        optim_graph = optim.optimize(self._graph, leaf_nodes)

        # update collect reference to optimized graph
        collect = collect.model_copy(update=dict(flow_=optim_graph))

        aggregation_manager = None
        # create the aggregation manager
        if aggregators is not None:
            in_features = {name: None for name in aggregators.keys()}  # TODO
            aggregation_manager = DataAggregationManager(
                aggregators, in_features
            )

        # build the sub-flow
        # TODO: restrict input features to only the
        #       ones required by the sub-graph
        flow = DataFlow(self.src_features.feature_)
        flow._graph = optim_graph
        flow._executor = LazyInstance(
            partial(
                DataFlowExecutor,
                graph=optim_graph,
                collect=collect,
                aggregation_manager=aggregation_manager,
            )
        )

        return flow, flow.aggregates

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
            AssertionError: If the flow has not been build yet.
        """
        assert self._executor is not None, "Flow has not been build yet."

        if rank is None:
            # try to get multiprocessing rank from pytorch worker info
            worker_info = get_worker_info()
            rank = 0 if worker_info is None else worker_info.id

        # create a new event loop to execute the flow in
        loop = asyncio.new_event_loop()
        # schedule the execution for the current batch
        future = self._executor.execute(batch, index, rank)
        out = loop.run_until_complete(future)
        # close the event loop
        loop.close()

        return out

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
        if isinstance(batch, datasets.formatting.formatting.LazyBatch):
            batch = dict(batch)
        # convert to pyarrow table with correct schema
        return pa.table(
            data=self.batch_process(batch, index, rank),
            schema=convert_features_to_arrow_schema(
                self._executor.collect.feature_
            ),
        )

    def apply(
        self,
        ds: D,
        collect: None | FeatureRef = None,
        aggregators: None | dict[str, AggregationRef] = None,
        **kwargs,
    ) -> tuple[D, None | dict[str, Any] | MappingProxyType[str, Any]]:
        """Apply the data flow to a dataset.

        This method applies the data flow to the given dataset, processing the data according to the defined
        processors and optionally including aggregators for dataset-wide computations.

        Args:
            ds (D): The dataset to process.
            collect (None | FeatureRef): The feature reference to collect. If None, uses current output features.
            aggregators (None | dict[str, AggregationRef]): Optional dictionary of aggregators for computing
                dataset-wide values. Defaults to None.
            **kwargs: Additional arguments for dataset mapping. Refer to the HuggingFace documentation for the
                Datasets.map function for the respective dataset type.

        Returns:
            tuple[D, None | dict[str, Any] | MappingProxyType[str, Any]]: The processed dataset and a snapshot
            of the aggregated values after processing the dataset. In case of iterable datasets, the aggregated
            values proxy object is returned instead of a snapshot.

        Raises:
            ValueError: If the dataset type is not supported.
            TypeError: If the dataset features do not match the source features.
            RuntimeError: If the flow has not been built yet.
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
        if (collect is None) and (aggregators is None):
            flow = self
        else:
            # default to output features of self
            if (collect is None) and (self._executor is not None):
                collect = self.out_features
            # build the flow
            flow, _ = self.build(collect=collect, aggregators=aggregators)

        if flow._executor is None:
            raise RuntimeError(
                "Flow has not been built yet. Please either build the flow "
                "manually using `.build()` or provide appropriate keyword "
                "arguments to the `.apply` call."
            )

        # run data flow
        ds = flow._internal_apply(ds, **kwargs)

        if isinstance(
            ds, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # set output features for lazy datasets manually
            if isinstance(ds, datasets.IterableDataset):
                ds.info.features = flow._executor.collect.feature_
            elif isinstance(ds, datasets.IterableDatasetDict):
                for split in ds.values():
                    split.info.features = flow._executor.collect.feature_

        # return the processed dataset and a snapshot of the aggregated values
        return ds, (
            None
            if flow.aggregates is None
            else flow.aggregates.copy()
            if isinstance(ds, (datasets.Dataset, datasets.DatasetDict))
            else flow.aggregates
        )

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
                - set(self._executor.collect.feature_.keys()),
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
        color_map: dict[
            Literal[
                DataFlowGraph.NodeType.SOURCE,
                DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeType.DATA_AGGREGATOR,
            ],
            str,
        ] = {},
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
            color_map (dict[None | type, str]): indicate custom color scheme based on the processor
                type. `None` refers to the source node.
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
            self._graph, subset_key=DataFlowGraph.NodeAttribute.DEPTH
        )

        # build color map
        cmap = colormaps.get_cmap("Pastel1")
        default_color_map = {
            DataFlowGraph.NodeType.SOURCE: cmap.colors[0],
            DataFlowGraph.NodeType.CONST: cmap.colors[1],
            DataFlowGraph.NodeType.DATA_PROCESSOR: cmap.colors[2],
            DataFlowGraph.NodeType.DATA_AGGREGATOR: cmap.colors[3],
        }
        color_map = default_color_map | color_map

        # apply color map
        node_colors = [
            color_map[data[DataFlowGraph.NodeAttribute.NODE_TYPE]]
            for _, data in self._graph.nodes(data=True)
        ]

        # plot the raw graph
        nx.draw(
            self._graph,
            pos,
            with_labels=False,
            node_size=node_size,
            node_color=node_colors,
            arrowsize=arrowsize,
            ax=ax,
        )

        # limit the maximum number of character in a single line in nodes
        max_line_length = node_size // (node_font_size * 65)

        node_labels = {}
        # build node labels
        for node, data in self._graph.nodes(data=True):
            if node == self._graph.src_node_id:
                # add root node label
                node_labels[node] = src_node_label

            else:
                # get the processor type name of the current node
                proc = data[DataFlowGraph.NodeAttribute.NODE_OBJ]
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
                            name=data[DataFlowGraph.EdgeAttribute.NAME],
                            key=str(data[DataFlowGraph.EdgeAttribute.KEY]),
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
