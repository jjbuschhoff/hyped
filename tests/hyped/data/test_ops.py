import pytest
from datasets import Features, Value

from hyped.data.flow import DataFlow
from hyped.data.ops import collect
from hyped.data.processors.ops.collect import CollectFeatures


def test_collect():
    flow = DataFlow(
        Features(
            {
                "x": Value("string"),
                "y": Value("string"),
            }
        )
    )

    out = collect([flow.src_features.x, flow.src_features.y])
    # make sure the node has been added
    assert out.node_id_ in flow._graph
    assert isinstance(
        flow._graph.nodes[out.node_id_]["processor"], CollectFeatures
    )
    # check the connections
    assert flow._graph.has_edge(flow.src_features.x.node_id_, out.node_id_)
    assert flow._graph.has_edge(flow.src_features.y.node_id_, out.node_id_)
