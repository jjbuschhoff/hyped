from unittest.mock import AsyncMock, MagicMock, call, patch

import datasets
import matplotlib.pyplot as plt
import pytest

from hyped.data.flow.core.flow import DataFlow
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.processor import IOContext


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

    def test_build_flow(self, setup_flow, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow

        src_ref = graph.get_node_output_ref(graph.src_node_id)
        cst_ref = graph.get_node_output_ref(const_node)
        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        # out features only set after build
        with pytest.raises(RuntimeError):
            flow.out_features

        # build subflow with processor and aggregator
        subflow, vals = flow.build(
            collect=out_ref, aggregators={"val": agg_ref}
        )
        assert len(subflow._graph) == 4
        assert subflow.out_features.key_ is out_ref.key_
        assert subflow.out_features.feature_ is out_ref.feature_
        assert vals == mock_manager.values_proxy
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

    def test_batch_process(self, setup_flow, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow
        print("+FLOW", id(flow._graph))
        print("+INIT", id(graph))

        out_ref = graph.get_node_output_ref(proc_node)
        agg_ref = graph.get_node_output_ref(agg_node)

        flow, vals = flow.build(collect=out_ref, aggregators={"val": agg_ref})
        # check output types
        assert isinstance(flow, DataFlow)
        assert vals == mock_manager.values_proxy

        # run batch process
        batch, index, rank = {"x": [1, 2, 3]}, [0, 1, 2], 0
        out = flow.batch_process(batch, index, rank)

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

        print(proc_node)
        print(out_ref.node_id_)

        print("+FLOW", id(flow._graph))
        print("+INIT", id(graph))

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

    def test_apply_overload(self, setup_flow, mock_manager):
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
        flow, graph, const_node, proc_node, agg_node = setup_flow
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
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.Dataset)
        assert vals == mock_manager.values_proxy.copy()

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
        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i, 0, io_ctx) for i in range(100)]
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
                )
                for i in range(10)
            ]
        )

    def test_apply_to_dataset_dict(self, setup_flow, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow
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
            ds, collect=out_ref, aggregators={"val": agg_ref}, batch_size=10
        )
        # check output types
        assert isinstance(out_ds, datasets.DatasetDict)
        assert out_ds.keys() == ds.keys()
        assert vals == mock_manager.values_proxy.copy()

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
        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i % 50, 0, io_ctx) for i in range(100)]
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
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset(self, setup_flow, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow
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
        # make sure processor is called for all samples in the dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i, 0, io_ctx) for i in range(100)]
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
                )
                for i in range(10)
            ]
        )

    def test_apply_to_iterable_dataset_dict(self, setup_flow, mock_manager):
        flow, graph, const_node, proc_node, agg_node = setup_flow
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
        # make sure processor is called for all samples in the train dataset
        p.process.assert_has_calls(
            [call({"a": i, "b": 0}, i % 50, 0, io_ctx) for i in range(50)]
        )
        # make sure the aggregator is called for all batches in the train dataset
        mock_manager.aggregate.assert_has_calls(
            [
                call(
                    a,
                    {"a": list(range(i * 10, (i + 1) * 10)), "b": [0] * 10},
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
            [call({"a": 50 + i, "b": 0}, i, 0, io_ctx) for i in range(50)]
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
