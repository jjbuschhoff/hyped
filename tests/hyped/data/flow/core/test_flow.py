from collections.abc import Mapping
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, call, patch

import datasets
import matplotlib.pyplot as plt
import pytest

from hyped.common.feature_checks import check_object_matches_feature
from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph

from .mock import MockInputRefs, MockProcessor


class TestDataFlow:
    @pytest.fixture(autouse=True)
    def mock_manager(self):
        with patch(
            "hyped.data.flow.core.flow.DataAggregationManager"
        ) as mock_manager:
            mock_manager = mock_manager()
            mock_manager.aggregate = AsyncMock()
            mock_manager.values_proxy = MagicMock()
            yield mock_manager

    @pytest.fixture(autouse=True)
    def mock_lazy_flow_output(self):
        with patch(
            "hyped.data.flow.core.flow.LazyFlowOutput"
        ) as mock_lazy_vals:
            mock_lazy_vals.return_value = {
                "y": 0
            }  # matches the output of the mock aggregator
            yield mock_lazy_vals

    def test_build_flow(self, setup_flow, mock_manager, mock_lazy_flow_output):
        flow, graph, const_node, proc_node, agg_node = setup_flow

        src_ref = graph.get_node_output_ref(graph.src_node_id)
        cst_ref = graph.get_node_output_ref(const_node)
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        # out features only set after build
        with pytest.raises(RuntimeError):
            flow.out_features
        # aggregates only set after build
        with pytest.raises(RuntimeError):
            flow.aggregates

        # build subflow with processor and aggregator
        subflow, vals = flow.build(collect=out_ref, aggregate=agg_ref)
        assert len(subflow._graph) == 4
        assert subflow.out_features.key_ is out_ref.key_
        assert subflow.out_features.feature_ is out_ref.feature_
        assert vals == subflow.aggregates

        # build subflow with processor only
        subflow, _ = flow.build(collect=out_ref)
        assert len(subflow._graph) == 3
        assert subflow.out_features.key_ is out_ref.key_
        assert subflow.out_features.feature_ is out_ref.feature_
        # build subflow with no processors
        subflow, _ = flow.build(collect=src_ref)
        assert len(subflow._graph) == 1
        assert subflow.out_features.key_ is src_ref.key_
        assert subflow.out_features.feature_ is src_ref.feature_

    def test_extract_lazy_flow(
        self, setup_flow, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow

        cst_ref = graph.get_node_output_ref(const_node)
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        # create processor
        p = MockProcessor()
        i = MockInputRefs(a=agg_ref.y, b=agg_ref.y)
        i = p._in_refs_validator.validate(i)
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)
        val_ref = graph.get_node_output_ref(node_id)

        i = MockInputRefs(a=agg_ref.y, b=cst_ref.value)
        i = p._in_refs_validator.validate(i)
        o = p._out_refs_type.build_features(p.config, i)
        # add processor to graph
        node_id = graph.add_processor_node(p, i, o)
        val_ref_w_const = graph.get_node_output_ref(node_id)

        def get_feature(graph, ref):
            return ref.key_.index_features(
                graph.nodes[ref.node_id_][
                    DataFlowGraph.NodeAttribute.OUT_FEATURES
                ]
            )

        mock_lazy_flow_output.reset_mock()
        # case A: aggregate is direct output of an aggregator
        flow.build(collect=out_ref, aggregate=agg_ref)
        # make sure the lazy flow output object is created correctly
        mock_lazy_flow_output.assert_called_once()
        assert (
            mock_lazy_flow_output.call_args.kwargs["input_proxy"]
            is mock_manager.values_proxy
        )
        assert get_feature(graph, agg_ref) == get_feature(
            mock_lazy_flow_output.call_args.kwargs["executor"].graph,
            mock_lazy_flow_output.call_args.kwargs["executor"].collect,
        )
        # make sure the lazy graph only contains of a single source node
        lazy_graph = mock_lazy_flow_output.call_args.kwargs["executor"].graph
        assert lazy_graph.src_node_id in lazy_graph
        assert len(lazy_graph.nodes) == 1
        # make sure the source node contains the output of the aggregator node
        assert (
            agg_ref.node_id_
            in lazy_graph.nodes[lazy_graph.src_node_id][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ]
        )

        mock_lazy_flow_output.reset_mock()
        # case B: aggregate is part of aggregated partition without constants
        flow.build(collect=out_ref, aggregate=val_ref)
        # make sure the lazy flow output object is created correctly
        mock_lazy_flow_output.assert_called_once()
        assert (
            mock_lazy_flow_output.call_args.kwargs["input_proxy"]
            is mock_manager.values_proxy
        )
        assert get_feature(graph, val_ref) == get_feature(
            mock_lazy_flow_output.call_args.kwargs["executor"].graph,
            mock_lazy_flow_output.call_args.kwargs["executor"].collect,
        )
        # make sure the lazy graph contains only the source node and the processor node
        lazy_graph = mock_lazy_flow_output.call_args.kwargs["executor"].graph
        assert lazy_graph.src_node_id in lazy_graph
        assert val_ref.node_id_ in lazy_graph
        assert len(lazy_graph.nodes) == 2
        # make sure the two nodes are connected correctly
        assert lazy_graph.has_edge(
            lazy_graph.src_node_id, val_ref.node_id_, key="a"
        )
        assert lazy_graph.has_edge(
            lazy_graph.src_node_id, val_ref.node_id_, key="b"
        )
        # make sure the source node contains the output of the aggregator node
        assert (
            agg_ref.node_id_
            in lazy_graph.nodes[lazy_graph.src_node_id][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ]
        )

        mock_lazy_flow_output.reset_mock()
        # case C: aggregate is part of aggregated partition with constants
        flow.build(collect=out_ref, aggregate=val_ref_w_const)
        # make sure the lazy flow output object is created correctly
        mock_lazy_flow_output.assert_called_once()
        assert (
            mock_lazy_flow_output.call_args.kwargs["input_proxy"]
            is mock_manager.values_proxy
        )
        assert get_feature(graph, val_ref_w_const) == get_feature(
            mock_lazy_flow_output.call_args.kwargs["executor"].graph,
            mock_lazy_flow_output.call_args.kwargs["executor"].collect,
        )
        # make sure the lazy graph contains only the source node, the constant and the processor node
        lazy_graph = mock_lazy_flow_output.call_args.kwargs["executor"].graph
        assert lazy_graph.src_node_id in lazy_graph
        assert cst_ref.node_id_ in lazy_graph
        assert val_ref_w_const.node_id_ in lazy_graph
        assert len(lazy_graph.nodes) == 3
        # make sure the two nodes are connected correctly
        assert lazy_graph.has_edge(
            lazy_graph.src_node_id, val_ref_w_const.node_id_, key="a"
        )
        assert lazy_graph.has_edge(
            cst_ref.node_id_, val_ref_w_const.node_id_, key="b"
        )
        # make sure the source node contains the output of the aggregator node
        assert (
            agg_ref.node_id_
            in lazy_graph.nodes[lazy_graph.src_node_id][
                DataFlowGraph.NodeAttribute.OUT_FEATURES
            ]
        )

    def test_batch_process(self, setup_flow, io_contexts, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        proc_io_ctx, agg_io_ctx = io_contexts

        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        flow, vals = flow.build(collect=out_ref, aggregate=agg_ref)
        assert isinstance(flow, DataFlow)
        assert isinstance(vals, Mapping)

        # run batch process
        batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
        out = flow.batch_process(batch, index, rank)

        # make sure the processor is called correctly
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        p.process.assert_has_calls(
            [
                call({"a": 1, "b": 0}, 0, 0, proc_io_ctx),
                call({"a": 2, "b": 0}, 1, 0, proc_io_ctx),
                call({"a": 3, "b": 0}, 2, 0, proc_io_ctx),
            ]
        )
        # make sure the aggregator is called correctly
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        mock_manager.aggregate.assert_called_with(
            a, {"a": [1, 2, 3], "b": [0, 0, 0]}, [0, 1, 2], 0, agg_io_ctx
        )

    def test_apply_overload(
        self, setup_flow, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

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
            aggregate=agg_ref,
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == dict(mock_lazy_flow_output())

        built_flow, _ = flow.build(collect=out_ref)
        # apply flow to dataset
        out_ds, _ = built_flow.apply(ds)
        assert isinstance(out_ds, datasets.Dataset)

        built_flow, vals = flow.build(collect=out_ref, aggregate=agg_ref)
        assert vals == mock_lazy_flow_output()
        # apply flow to dataset
        out_ds, vals = built_flow.apply(ds)
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == MappingProxyType(mock_lazy_flow_output())

    def test_apply_to_dataset(
        self, setup_flow, io_contexts, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        proc_io_ctx, agg_io_ctx = io_contexts
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]

        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"x": list(range(100))}, features=flow.src_features.feature_
        )

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregate=agg_ref, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == mock_lazy_flow_output()

        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i, 0, proc_io_ctx) for i in range(100)]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": [0] * 10,
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                    agg_io_ctx,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_dataset_dict(
        self, setup_flow, io_contexts, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        proc_io_ctx, agg_io_ctx = io_contexts
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]

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
            ds, collect=out_ref, aggregate=agg_ref, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.DatasetDict)
        assert out_ds.keys() == ds.keys()
        assert vals == mock_lazy_flow_output()

        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [
                call({"a": i, "b": 0}, i % 50, 0, proc_io_ctx)
                for i in range(100)
            ]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": [0] * 10,
                    },
                    list(range((i % 5) * 10, ((i % 5) + 1) * 10)),
                    0,
                    agg_io_ctx,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset(
        self, setup_flow, io_contexts, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        proc_io_ctx, agg_io_ctx = io_contexts
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]

        # create dummy dataset
        ds = datasets.Dataset.from_dict(
            {"x": list(range(100))}, features=flow.src_features.feature_
        ).to_iterable_dataset(num_shards=5)

        # apply flow to dataset
        out_ds, vals = flow.apply(
            ds, collect=out_ref, aggregate=agg_ref, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.IterableDataset)
        assert vals == mock_lazy_flow_output()

        # at this point the processors shouldn't be called yet
        assert not p.process.called
        assert not mock_manager.aggregate.called

        # consume iterable dataset
        for _ in out_ds:
            pass

        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i, 0, proc_io_ctx) for i in range(100)]
        )
        # make sure the aggregator is called for all batches
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(i * 10, (i + 1) * 10)),
                        "b": [0] * 10,
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                    agg_io_ctx,
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset_dict(
        self, setup_flow, io_contexts, mock_manager, mock_lazy_flow_output
    ):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        proc_io_ctx, agg_io_ctx = io_contexts
        # get references
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)
        # get the processor and aggregator instance
        p = graph.nodes[proc_node][DataFlowGraph.NodeAttribute.NODE_OBJ]
        a = graph.nodes[agg_node][DataFlowGraph.NodeAttribute.NODE_OBJ]

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
            ds, collect=out_ref, aggregate=agg_ref, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.IterableDatasetDict)
        assert vals == mock_lazy_flow_output()
        assert out_ds.keys() == ds.keys()

        # at this point the processors shouldn't be called yet
        assert not p.process.called
        assert not mock_manager.aggregate.called

        # consume train dataset
        for _ in out_ds["train"]:
            pass

        # make sure processor is called for all samples in the train dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i % 50, 0, proc_io_ctx) for i in range(50)]
        )
        # make sure the aggregator is called for all batches in the train dataset
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {"a": list(range(i * 10, (i + 1) * 10)), "b": [0] * 10},
                    list(range((i % 5) * 10, ((i % 5) + 1) * 10)),
                    0,
                    agg_io_ctx,
                )
                for i in range(5)
            ]
        )

        # consume train dataset
        for _ in out_ds["test"]:
            pass

        # make sure processor is called for all samples in the train dataset
        p.process.assert_has_calls(
            [call({"a": 50 + i, "b": 0}, i, 0, proc_io_ctx) for i in range(50)]
        )
        # make sure the aggregator is called for all batches in the train dataset
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {
                        "a": list(range(50 + i * 10, 50 + (i + 1) * 10)),
                        "b": [0] * 10,
                    },
                    list(range(i * 10, (i + 1) * 10)),
                    0,
                    agg_io_ctx,
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
        flow, graph, const_node, proc_node, agg_node = setup_flow
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
                if node == graph.src_node_id
                else type(data[DataFlowGraph.NodeAttribute.NODE_OBJ]).__name__
            )
            assert any(
                node_label in text.get_text() for text in ax.texts
            ), f"Node label {node_label} is missing in the plot."

        # Check if edge labels are correct
        for edge in flow._graph.edges(data=True):
            _, _, data = edge
            edge_label = edge_label_format.format(
                name=data[DataFlowGraph.EdgeAttribute.NAME],
                key=data[DataFlowGraph.EdgeAttribute.KEY],
            )

            if with_edge_labels:
                assert any(
                    edge_label in text.get_text() for text in ax.texts
                ), f"Edge label {edge_label} is missing in the plot."
            else:
                assert all(
                    edge_label not in text.get_text() for text in ax.texts
                ), f"Edge label {edge_label} in the plot but shouldn't be included."
