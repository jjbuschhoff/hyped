from unittest.mock import MagicMock

import pytest
from datasets import Sequence, Value
from typing_extensions import Annotated

# import hyped.data.processors.base
from hyped.data.flow.ref import FeatureRef
from hyped.data.processors.base.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    FeatureValidator,
    InputRefs,
)


def test_feature_validator():
    # create mock validators
    x_validator = MagicMock()
    y_validator = MagicMock()

    # create custom input refs class using mock validators
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(x_validator)]
        y: Annotated[FeatureRef, FeatureValidator(y_validator)]

    k, n, f = tuple(), -1, None
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # instantiate the input refs and make sure the
    # validators were called with the correct arguments
    CustomInputRefs(x=x_ref, y=y_ref)
    x_validator.assert_called_with(x_ref, Value("int32"))
    y_validator.assert_called_with(y_ref, Value("string"))


def test_check_feature_equals():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureEquals(Value("int32"))]
        y: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]

    k, n, f = tuple(), -1, None
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # create input refs, all types match
    CustomInputRefs(x=x_ref, y=y_ref)

    # create input refs but one type doesn't match the expectation
    with pytest.raises(TypeError):
        CustomInputRefs(x=x_ref, y=x_ref)


def test_check_feature_is_sequence():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureIsSequence(Value("int32"))]
        y: Annotated[FeatureRef, CheckFeatureIsSequence(Value("string"))]

    k, n, f = tuple(), -1, None
    # create dummy input refs
    x_ref = FeatureRef(
        key_=k, node_id_=n, flow_=f, feature_=Sequence(Value("int32"))
    )
    y_ref = FeatureRef(
        key_=k,
        node_id_=n,
        flow_=f,
        feature_=Sequence(Value("string"), length=2),
    )
    # create input refs, all types match
    CustomInputRefs(x=x_ref, y=y_ref)

    # create input refs but one type doesn't match the expectation
    with pytest.raises(TypeError):
        CustomInputRefs(x=x_ref, y=x_ref)

    # test with specified length
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, CheckFeatureIsSequence(Value("int32"))]
        y: Annotated[
            FeatureRef, CheckFeatureIsSequence(Value("string"), length=2)
        ]

    # create input refs, all types match
    CustomInputRefs(x=x_ref, y=y_ref)
    # create input refs but one type doesn't match the expectation
    with pytest.raises(TypeError):
        CustomInputRefs(x=x_ref, y=x_ref)


def test_input_refs():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(lambda r, f: None)]
        y: Annotated[FeatureRef, FeatureValidator(lambda r, f: None)]

    k, n, f = tuple(), -1, MagicMock
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))
    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)

    # check properties
    assert input_refs.keys == {"x", "y"}
    assert input_refs.refs == {x_ref, y_ref}
    assert input_refs.named_refs == {"x": x_ref, "y": y_ref}
    assert input_refs.flow == f
