from __future__ import annotations

import asyncio
import nest_asyncio
from typing import TypeVar, Any
from typing_extensions import TypeAlias
from types import SimpleNamespace

import datasets
import pyarrow as pa
import networkx as nx
from torch.utils.data import get_worker_info

from .ref import FeatureRef
from hyped.common.feature_key import FeatureKey
from hyped.common.arrow import convert_features_to_arrow_schema
from hyped.common.feature_checks import check_feature_equals
from hyped.data.processors.base import BaseDataProcessor
from hyped.data.processors.inputs import InputRefs
from hyped.data.processors.outputs import OutputRefs

Batch: TypeAlias = dict[str, list[Any]]
D = TypeVar(
    "D",
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict
)

SRC_NODE_ID = 0

# patch asyncio if running in an async environment, such as jupyter notebook
# this fixes #26
nest_asyncio.apply()


class DataFlowGraph(nx.MultiDiGraph):

    def add_src_node(self, features: datasets.Features) -> FeatureRef:
        # add src node to graph
        self.add_node(
            SRC_NODE_ID,
            processor=None,
            features=features
        )
        # return feature ref over input features
        return FeatureRef(
            key_=tuple(),
            feature_=features,
            node_id_=SRC_NODE_ID,
            flow_=self
        )

    def add_processor(
        self,
        processor: BaseDataProcessor,
        inputs: InputRefs
    ) -> OutputRefs:
        # make sure the input refs match the processor
        assert isinstance(inputs, processor._in_refs_type)

        # add processor to graph
        # TODO: seems unintuitive that the output refs are
        #       created here and not in the call method of
        #       the processor
        node_id = self.number_of_nodes()
        outputs = processor._out_refs_type(
            processor.config, inputs, node_id
        )
        self.add_node(
            node_id,
            processor=processor,
            features=outputs.feature_
        )
        # add dependency edges to graph
        for name, ref in inputs.named_refs.items():
            # make sure the inputs come from this flow
            if ref.flow_ != self:
                raise RuntimeError()
            # make sure the input is a valid output of the referred node
            assert ref.node_id_ in self
            assert ref.key_.index_features(
                self.nodes[ref.node_id_]["features"]
            ) is not None
            # add edge to other nodes
            x = self.add_edge(
                ref.node_id_, node_id, key=name, feature_key=ref.key_
            )

        return outputs

    def dependency_graph(self, node: int):

        visited = set()
        nodes = set([node])
        # search through dependency graph
        while len(nodes) > 0:
            node = nodes.pop()
            visited.add(node)
            nodes.update(self.predecessors(node))
        
        return self.subgraph(visited)
        

class ExecutionState(object):

    def __init__(self, graph: DataFlowGraph, batch: Batch, index: list[int], rank: int):
       
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
        if node_id != SRC_NODE_ID:
            await self.ready[node_id].wait()

    def collect_value(self, ref: FeatureRef) -> Batch:
        # collect values requested by the feature reference
        assert isinstance(ref.feature_, (datasets.Features, dict))
        batch = ref.key_.index_batch(self.outputs[ref.node_id_])
        # in case the feature key is empty the collected values are already
        # in batch format, otherwise they need to be converted from a
        # list-of-dicts to a dict-of-lists
        if len(ref.key_) != 0:
            batch = {
                key: [d[key] for d in batch]
                for key in batch[0].keys()
            } if len(batch) > 0 else {}

        return batch

    def collect_inputs(self, node_id: int) -> tuple[Batch, list[int]]:

        inputs, index = dict(), set()
        for u, v, name, data in self.graph.in_edges(node_id, keys=True, data=True):
            assert (u == SRC_NODE_ID) or self.ready[u].is_set()
            # get feature key from data
            key = data["feature_key"]
            # get the values requested by the batch
            values = key.index_batch(self.outputs[u])
            # this is always a list of values, except when the key is empty
            # in that case the values are the exact output of the source node u
            if len(key) == 0:
                assert isinstance(
                    values, (dict, datasets.formatting.formatting.LazyBatch)
                )
                keys = values.keys()
                values = [
                    dict(zip(keys, vals)) for vals in zip(*values.values())
                ]
            assert isinstance(values, list)
            # store the values in inputs and add keep track of the index hash
            inputs[name] = values
            index.add(self.index_hash[u])

        # make sure that all collected inputs have the same index
        assert len(index) == 1
        index = next(iter(index))

        return inputs, self.index_lookup[index]

    def capture_output(
        self,
        node_id: int,
        output: Batch,
        new_index: list[int]
    ) -> None:
        assert not self.ready[node_id].is_set()
        assert all(len(vals) == len(new_index) for vals in output.values())

        self.outputs[node_id] = output

        new_index_hash = hash(tuple(new_index))
        self.index_hash[node_id] = new_index_hash
        if new_index_hash not in self.index_lookup:
            self.index_lookup[new_index_hash] = new_index
        
        self.ready[node_id].set()


class DataFlowExecutor(object):

    def __init__(
        self,
        graph: DataFlowGraph,
        collect: FeatureRef
    ) -> None:

        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError()

        self.graph = graph
        self.collect = collect

    async def execute_node(
        self, node_id: int, state: ExecutionState
    ):

        if self.graph.in_degree(node_id) > 0:
            # wait for all dependencies of the current node
            deps = self.graph.predecessors(node_id)
            futures = map(state.wait_for, deps)
            await asyncio.gather(*futures)

        # collect inputs for processor execution
        inputs, index = state.collect_inputs(node_id)
        processor = self.graph.nodes[node_id]["processor"]
        # run processor and capture output in execution state
        out, out_index = await processor.batch_process(inputs, index, state.rank)
        state.capture_output(node_id, out, out_index)

    async def execute(self, batch: Batch, index: list[int], rank: int) -> Batch:

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

    def __init__(self, features: datasets.Features) -> None:
        # create graph
        self._graph = DataFlowGraph()
        self._src_features = self._graph.add_src_node(features=features)
        self._out_features: None | FeatureRef = None

    @property
    def src_features(self) -> FeatureRef:
        return self._src_features

    @property
    def out_features(self) -> FeatureRef:
        if self._out_features is None:
            raise RuntimeError()
        return self._out_features

    def build(
        self, collect: FeatureRef
    ) -> DataFlow:

        if not isinstance(collect.feature_, (datasets.Features, dict)):
            raise TypeError()

        sub_graph = self._graph.dependency_graph(collect.node_id_)
        # build sub_flow
        # TODO: restrict input features to only the
        #       ones required by the sub-graph
        flow = DataFlow(self.src_features.feature_)
        flow._out_features = collect
        flow._graph = sub_graph

        return flow

    def batch_process(self, batch: Batch, index: list[int], rank: None | int = None) -> Batch:
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
        # convert to pyarrow table with correct schema
        return pa.table(
            data=self.batch_process(batch, index, rank),
            schema=convert_features_to_arrow_schema(self._out_features.feature_),
        )
    
    def apply(self, ds: D, collect: None | FeatureRef = None, **kwargs) -> D:

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
            and not check_feature_equals(features, self.src_features.feature_)
        ):
            # TODO: should only check whether the features are present
            #       i.e. they should be a subset and don't need to match exactly
            raise TypeError()

        # build the sub data flow required to compute the requested output features
        flow = self if collect is None else self.build(collect=collect)

        if flow._out_features is None:
            raise ValueError()

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

