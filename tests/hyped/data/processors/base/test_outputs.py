from unittest.mock import MagicMock

import pytest
from datasets import Features, Value
from typing_extensions import Annotated

from hyped.data.flow.ref import FeatureRef
from hyped.data.processors.base.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)


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
    CustomOutputRefs(config, inputs, -1)
    # make sure the feature generators were called with the correct inputs
    x_feature.assert_called_with(config, inputs)
    y_feature.assert_called_with(config, inputs)


def test_output_refs():
    x_feature = OutputFeature(Value("int32"))
    y_feature = OutputFeature(Value("string"))

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
    inst = CustomOutputRefs(MagicMock(), MagicMock(), -1)

    # check feature type
    assert inst.x.feature_ == Value("int32")
    assert inst.y.feature_ == Value("string")
    assert inst.feature_ == Features(
        {"x": Value("int32"), "y": Value("string")}
    )

    # check refs
    assert inst.refs == {inst.x, inst.y}
