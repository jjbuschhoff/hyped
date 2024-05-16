from __future__ import annotations

import asyncio
import nest_asyncio
from typing import TypeVar, Any
from typing_extensions import TypeAlias

import datasets
import pyarrow as pa
import networkx as nx

from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.common.feature_ref import FeatureRef, FeatureCollection, Feature, FeatureKey
from hyped.common.feature_checks import check_feature_equals
from hyped.data.processors.base import BaseDataProcessor

Batch: TypeAlias = dict[str, list[Any]]
D = TypeVar(
    "D",
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict
)
P = TypeVar("P", bound=BaseDataProcessor)

SRC_FEATURES_NODE_ID = -1

# patch asyncio if running in an async environment, such as jupyter notebook
# this fixes #26
nest_asyncio.apply()


class DataFlow(object):

    def __init__(self, features: datasets.Features) -> None:

        self._graph = nx.MultiDiGraph()
        self._src_features = FeatureRef(
            key=FeatureKey(),
            feature=features,
            node_id=SRC_FEATURES_NODE_ID
        )
        self._processors: dict[int, BaseDataProcessor] = {}
        self._out_collection: None | FeatureCollection = None

    @property
    def src_features(self) -> FeatureRef:
        return self._src_features

    def add_node(self, processor: P) -> P:
        # get the node id and set it in the processor
        node_id = self._graph.number_of_nodes()
        processor.node_id = node_id
        # add processor node to graph
        self._graph.add_node(node_id)
        self._processors[node_id] = processor
        # add dependency edges to graph
        for ref in processor.config.inputs.refs:
            if isinstance(ref, FeatureRef):
                if ref.node_id != SRC_FEATURES_NODE_ID:
                    self._graph.add_edge(ref.node_id, node_id)
            elif isinstance(ref, FeatureCollection):
                for r in ref.refs:
                    if r.node_id != SRC_FEATURES_NODE_ID:
                        self._graph.add_edge(r.node_id, node_id)

        return processor

    def build(
        self,
        collect: FeatureCollection
    ) -> DataFlow:

        if not isinstance(collect.collection, dict):
            raise TypeError()

        def dependency_graph(graph, nodes):

            visited = set()
            nodes = set(nodes)
            # search through dependency graph
            while len(nodes) > 0:
                node = nodes.pop()
                visited.add(node)
                nodes.update(graph.predecessors(node))
            
            return graph.subgraph(visited)

        sub_graph = dependency_graph(
            self._graph, [ref.node_id for ref in collect.refs]
        )

        # build sub_flow
        # TODO: restrict input features to only the
        #       ones required by the sub-graph
        flow = DataFlow(self.src_features.feature)
        flow._out_collection = collect
        flow._graph = sub_graph
        flow._processors = {
            n: self._processors[n] for n in sub_graph.nodes()
        }

        return flow

    def batch_process(self, batch: Batch, index: list[int], rank: int) -> Batch:
        
        # create a data flow executor
        executor = DataFlowExecutor(self._graph, self._processors, self._out_collection)
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
        # convert to pyarrow table with correct schema
        return pa.table(
            data=self.batch_process(batch, index, rank),
            schema=convert_features_to_arrow_schema(self._out_collection.feature),
        )
    
    def apply(self, ds: D, collect: None | FeatureCollection = None, **kwargs) -> D:

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
                "got %s" % type(data)
            )

        if (
            (features is not None)
            and not check_feature_equals(features, self.src_features.feature)
        ):
            # TODO: should only check whether the features are present
            #       i.e. they should be a subset and don't need to match exactly
            raise TypeError()

        # build the sub data flow required to compute the requested output features
        flow = self if collect is None else self.build(collect=collect)

        if flow._out_collection is None:
            raise ValueError()

        # run data flow
        ds = flow._internal_apply(ds, **kwargs)

        if isinstance(
            ds, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            # set output features for lazy datasets manually
            if isinstance(ds, datasets.IterableDataset):
                ds.info.features = flow._out_collection.feature
            elif isinstance(ds, datasets.IterableDatasetDict):
                for split in ds.values():
                    split.info.features = flow._out_collection.feature

        return ds

    def _internal_apply(self, ds: D, **kwargs) -> D:

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
                remove_columns=set(self.src_features.feature.keys())
                - set(self._out_collection.collection.keys()),
                **kwargs,
            )


class ExecutionState(object):

    def __init__(self, graph: nx.MultiDiGraph, batch: Batch, index: list[int], rank: int):
       
        src_index_hash = hash(tuple(index))

        self.outputs = {SRC_FEATURES_NODE_ID: batch}
        self.index_hash = {SRC_FEATURES_NODE_ID: src_index_hash}
        self.index_lookup = {src_index_hash: index}

        self.rank = rank
        self.ready = {
            node_id: asyncio.Event()
            for node_id in graph.nodes()
        }

    async def wait_for(self, node_id: int) -> None:
        await self.ready[node_id].wait()

    def _collect_values(self, ref: Feature) -> tuple[list[Any], str]:

        if isinstance(ref, FeatureRef):
            assert ref.node_id in self.outputs
            # get index id and values
            index = self.index_hash[ref.node_id]
            values = ref.key.index_batch(self.outputs[ref.node_id])
            return values, index

        if isinstance(ref, FeatureCollection):

            if isinstance(ref.collection, dict):
                data = {k: self._collect_values(v) for k, v in ref.collection.items()}
                # all index ids must be equal, otherwise we cannot merge
                index = set(i for _, i in data.values())
                assert len(index) == 1
                # get values only and convert from dict of lists to list of dicts
                data = {k: v for k, (v, _) in data.items()}
                data = [
                    dict(zip(data.keys(), values)) for values in zip(*data.values())
                ]

                return data, next(iter(index))

            if isinstance(ref.collection, list):
                data = map(self._collect_values, ref.collection)
                # all index ids must be equal, otherwise we cannot merge
                index = set(i for i, _ in data)
                assert len(index) == 1
                # get only values and transpose
                data = (v for _, v in data)
                data = [list(row) for row in zip(*data)]

                return data, next(iter(index))

            raise TypeError()

    def collect_values(self, ref: Feature) -> tuple[list[Any], list[int]]:

        values, index = self._collect_values(ref)
        index = self.index_lookup[index]

        return values, index

    def collect_inputs(self, processor: BaseDataProcessor) -> tuple[dict[FeatureRef, list[Any]], list[int]]:

        inputs = {
            ref: self._collect_values(ref)
            for ref in processor.config.inputs.refs
        }
        # all index ids must be equal, otherwise we cannot merge
        index = set(i for _, i in inputs.values())
        assert len(index) == 1
        # collect only input values
        inputs = {k: v for k, (v, _) in inputs.items()}
        # get the index from the index id
        index = self.index_lookup[next(iter(index))]
        return inputs, index

    def capture_output(
        self,
        node_id: int,
        output: dict[FeatureKey, list[Any]],
        new_index: list[int]
    ) -> None:
        assert not self.ready[node_id].is_set()
        assert all(len(ref.key) == 1 for ref in output.keys())
        # unpack keys and store output in state
        output = {ref.key[0]: val for ref, val in output.items()}
        self.outputs[node_id] = output

        new_index_hash = hash(tuple(new_index))
        self.index_hash[node_id] = new_index_hash
        if new_index_hash not in self.index_lookup:
            self.index_lookup[new_index_hash] = new_index
        
        self.ready[node_id].set()


class DataFlowExecutor(object):

    def __init__(
        self,
        flow_graph: nx.MultiDiGraph,
        processors: dict[int, BaseDataProcessor],
        collect: FeatureCollection
    ) -> None:

        assert isinstance(collect.collection, dict)

        self.processors = processors
        self.graph = flow_graph
        self.collect = collect

    async def execute_node(
        self, node_id: int, state: ExecutionState
    ):

        if self.graph.in_degree(node_id) > 0:
            # wait for all dependencies of the current node
            deps = self.graph.predecessors(node_id)
            futures = map(state.wait_for, deps)
            await asyncio.gather(*futures)

        # get the processor for the current node
        processor = self.processors[node_id]
        # collect inputs for processor execution
        inputs, index = state.collect_inputs(processor)
        # run processor and capture output in execution state
        out, out_index = await processor.batch_process(inputs, index, state.rank)
        state.capture_output(node_id, out, out_index)

    async def execute(self, batch: Batch, index: list[int], rank: int) -> Batch:

        # create an execution state
        state = ExecutionState(self.graph, batch, index, rank)

        # execute all nodes in the flow
        await asyncio.gather(
            *[self.execute_node(node_id, state) for node_id in self.graph.nodes()]
        )

        # collect output values
        return {
            key: state.collect_values(ref)[0]
            for key, ref in self.collect.collection.items()
        }
