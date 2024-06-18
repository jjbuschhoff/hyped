from unittest.mock import MagicMock, patch

from datasets import Features, Sequence, Value

from hyped.data.flow.core.nodes.const import Const, ConstConfig


def test_const_config():
    # test infer feature type
    config = ConstConfig(value=5)
    assert isinstance(
        config.feature, Value
    ) and config.feature.dtype.startswith("int")
    # test infer sequence feature
    config = ConstConfig(value=[4, 5, 6])
    assert (
        isinstance(config.feature, Sequence)
        and isinstance(config.feature.feature, Value)
        and config.feature.feature.dtype.startswith("int")
    )
    # test infer nested feature
    config = ConstConfig(value={"a": 3, "b": {"x": "x", "y": "y"}})
    assert config.feature == Features(
        {
            "a": Value("int64"),
            "b": {
                "x": Value("string"),
                "y": Value("string"),
            },
        }
    )

    with patch(
        "hyped.data.flow.core.nodes.const.raise_object_matches_feature"
    ) as mock:
        # create a mock config
        mock_value, feature = MagicMock(), Value("int64")
        ConstConfig(value=mock_value, feature=feature)
        # make sure the check fired
        mock.assert_called_once_with(mock_value, feature)


def test_const():
    # create a mock config
    const = Const(value=0)
    #
    assert const.get_const_batch(3) == {"value": [0, 0, 0]}
    assert const.get_const_batch(4) == {"value": [0, 0, 0, 0]}
    # create a mock flow
    mock_flow = MagicMock()
    mock_flow.add_processor_node = MagicMock(return_value="node_id")
    # add constant value to mock flow
    out = const.to(mock_flow)
    # make sure the node was added to the mock flow
    mock_flow.add_processor_node.assert_called_with(
        const, None, Features({"value": const.config.feature})
    )
    # check the output reference
    assert out.node_id_ == "node_id"
    assert out.flow_ == mock_flow
