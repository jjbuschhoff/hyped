from unittest.mock import MagicMock

import pytest
from datasets import Features, Value
from typing_extensions import Annotated

from hyped.data.flow.core.refs.outputs import (
    ConditionalOutputFeature,
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef


def test_lambda_output_feature():
    x_feature = MagicMock(return_value=Value("string"))
    y_feature = MagicMock(return_value=Value("int32"))

    class CustomOutputRefs(OutputRefs):
        x: Annotated[FeatureRef, LambdaOutputFeature(x_feature)]
        y: Annotated[FeatureRef, LambdaOutputFeature(y_feature)]

    # create config and inputs mock
    config = MagicMock()
    inputs = MagicMock()
    # create output refs instance
    features = CustomOutputRefs.build_features(config, inputs)
    # make sure the feature generators were called with the correct inputs
    x_feature.assert_called_with(config, inputs)
    y_feature.assert_called_with(config, inputs)
    # create instance
    inst = CustomOutputRefs(MagicMock(), "", features)
    # check features
    assert inst.x.feature_ == Value("string")
    assert inst.y.feature_ == Value("int32")


def test_conditional_output_feature():
    class CustomOutputRefs(OutputRefs):
        x: Annotated[
            FeatureRef,
            ConditionalOutputFeature(Value("int32"), lambda *args: True),
        ]
        y: Annotated[
            FeatureRef,
            ConditionalOutputFeature(Value("int32"), lambda *args: False),
        ]

    # create config and inputs mock
    config = MagicMock()
    inputs = MagicMock()

    # build output features and create output refs instance
    features = CustomOutputRefs.build_features(config, inputs)
    inst = CustomOutputRefs(inputs.flow, "", features)

    # x should be propagated, y should not be propagated
    assert hasattr(inst, "x")
    assert not hasattr(inst, "y")
    # check the feature type
    assert inst.x.feature_ == Value("int32")

    # raise error when accessing non-propagated feature
    with pytest.raises(AttributeError):
        inst.y


def test_output_refs():
    x_feature = OutputFeature(Value("int32"))
    y_feature = OutputFeature(Value("string"))

    with pytest.raises(TypeError):

        class CustomOutputRefs(OutputRefs):
            x: Annotated[FeatureRef, y_feature]
            y: FeatureRef  # invalid type annotation

    class CustomOutputRefs(OutputRefs):
        x: Annotated[FeatureRef, x_feature]
        y: Annotated[FeatureRef, y_feature]

    # check class vars
    assert CustomOutputRefs._feature_generators == {
        "x": x_feature,
        "y": y_feature,
    }
    assert CustomOutputRefs._feature_names == {"x", "y"}

    # create output refs instance
    features = CustomOutputRefs.build_features(MagicMock(), MagicMock())
    inst = CustomOutputRefs(MagicMock(), "", features)

    # check feature type
    assert inst.x.feature_ == Value("int32")
    assert inst.y.feature_ == Value("string")
    assert inst.feature_ == Features(
        {"x": Value("int32"), "y": Value("string")}
    )

    # check refs
    assert inst.refs == {inst.x, inst.y}

    with pytest.raises(AttributeError):
        # test accessing non-existing attributes
        # should raise the standard attribute error
        inst.invalid_attribute
