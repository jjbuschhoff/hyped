from typing import Any, Hashable, Iterable
from unittest.mock import MagicMock

import pytest
from datasets import Features, Sequence, Value
from typing_extensions import Annotated

from hyped.data.flow.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    FeatureValidator,
    InputRefs,
)

# import hyped.data.processors.base
from hyped.data.flow.refs.ref import NONE_REF, FeaturePointer, FeatureRef


def ptr_set(refs: Iterable[FeatureRef]) -> set[FeaturePointer]:
    return {r.ptr for r in refs}


def ptr_dict(d: dict[Hashable, FeatureRef]) -> dict[Hashable, FeaturePointer]:
    return {k: r.ptr for k, r in d.items()}


def test_error_on_invalid_annotation():
    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: FeatureRef

    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: Annotated[FeatureRef, object]

    with pytest.raises(TypeError):

        class CustomInputRefs(InputRefs):
            x: Annotated[object, FeatureValidator(MagicMock())]

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock())]


def test_error_on_invalid_value():
    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(MagicMock)]

    with pytest.raises(ValueError):
        CustomInputRefs(x=int)


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
    z_ref = FeatureRef(
        key_=k,
        node_id_=n,
        flow_=f,
        feature_=Sequence(Value("string"), length=4),
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
    # create input refs but sequence length doesn't match the expectation
    with pytest.raises(TypeError):
        CustomInputRefs(x=x_ref, y=z_ref)


def test_required_input_refs():
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
    assert input_refs.required_keys == {"x", "y"}
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )
    assert input_refs.flow == f
    assert input_refs.features_ == Features(
        {"x": x_ref.feature_, "y": y_ref.feature_}
    )


def test_optional_input_refs():
    k, n, f = tuple(), -1, MagicMock
    # create dummy input refs
    x_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("int32"))
    y_ref = FeatureRef(key_=k, node_id_=n, flow_=f, feature_=Value("string"))

    class CustomInputRefs(InputRefs):
        x: Annotated[
            FeatureRef, FeatureValidator(lambda r, f: None)
        ] = NONE_REF
        y: Annotated[
            FeatureRef, FeatureValidator(lambda r, f: None)
        ] = NONE_REF

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)
    # check properties
    assert input_refs.required_keys == set()
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )
    assert input_refs.flow == f

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref)
    # check properties
    assert input_refs.required_keys == set()
    assert ptr_set(input_refs.refs) == ptr_set([x_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"x": x_ref})
    assert input_refs.flow == f

    # create input refs instance
    input_refs = CustomInputRefs(y=y_ref)
    # check properties
    assert input_refs.required_keys == set()
    assert ptr_set(input_refs.refs) == ptr_set([y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"y": y_ref})
    assert input_refs.flow == f

    class CustomInputRefs(InputRefs):
        x: Annotated[FeatureRef, FeatureValidator(lambda r, f: None)]
        y: Annotated[
            FeatureRef, FeatureValidator(lambda r, f: None)
        ] = NONE_REF

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref, y=y_ref)
    # check properties
    assert input_refs.required_keys == {"x"}
    assert ptr_set(input_refs.refs) == ptr_set([x_ref, y_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict(
        {"x": x_ref, "y": y_ref}
    )
    assert input_refs.flow == f

    # create input refs instance
    input_refs = CustomInputRefs(x=x_ref)
    # check properties
    assert input_refs.required_keys == {"x"}
    assert ptr_set(input_refs.refs) == ptr_set([x_ref])
    assert ptr_dict(input_refs.named_refs) == ptr_dict({"x": x_ref})
    assert input_refs.flow == f
