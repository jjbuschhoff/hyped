"""Pydantic helper functionality."""
import datetime
from functools import partial

import datasets
import pydantic
from typing_extensions import Annotated

# map datasets value dtype to
DATASETS_VALUE_TYPE_MAPPING = {
    "bool": bool,
    "int8": int,
    "int16": int,
    "int32": int,
    "int64": int,
    "uint8": int,
    "uint16": int,
    "uint32": int,
    "uint64": int,
    "float16": float,
    "float32": float,
    "float64": float,
    "string": str,
    "large_string": str,
    "date32": datetime.datetime,
    "date64": datetime.datetime,
    "time32": datetime.time,
    "time64": datetime.time,
}


def _class_label_validator(
    v: str | int | None,
    info: pydantic.ValidationInfo,
    feature: datasets.ClassLabel,
) -> int | None:
    """Class Label validator.

    Converts string labels to their integer representations.

    Arguments:
        v (str | int | None): value to validate
        info (pydantic.ValidationInfo): validation info
        feature (datasets.ClassLabel): class label feature

    Returns:
        validated_v (int | None): validated value
    """
    if (v is None) or isinstance(v, int):
        return v
    if isinstance(v, str):
        return feature.str2int(v)

    raise TypeError(
        "Invalid type for class label feature, expected str or int, got %s" % v
    )


def pydantic_model_from_features(
    features: datasets.Features,
) -> pydantic.BaseModel:
    """Create a pydantic model from dataset features.

    Arguments:
        features (Features): datasets features to build the pydantic model for

    Returns:
        model (pydantic.BaseModel):
            pydantic model matching the structure of the dataset features.
    """
    fields = {}
    for k, field_type in features.items():
        if isinstance(field_type, datasets.Value):
            # get data type for the given field
            dtype = DATASETS_VALUE_TYPE_MAPPING.get(
                field_type.dtype, field_type.pa_type.to_pandas_dtype()
            )
            # set field
            fields[k] = (
                dtype | None,
                None,
            )

        elif isinstance(field_type, datasets.ClassLabel):
            fields[k] = (
                Annotated[
                    int,
                    pydantic.BeforeValidator(
                        partial(_class_label_validator, feature=field_type)
                    ),
                ]
                | None,
                None,
            )
            # fields[k] = (Literal[tuple(field_type.names)] | None, None)

        elif isinstance(field_type, datasets.Sequence):
            # infer dtype for sequence values
            dtype = (
                pydantic_model_from_features({"field": field_type.feature})
                .model_fields["field"]
                .annotation
            )
            # set field
            fields[k] = (list[dtype], pydantic.Field(default_factory=list))

        elif isinstance(field_type, (dict, datasets.Features)):
            model = pydantic_model_from_features(field_type)
            # set field
            fields[k] = (
                model,
                pydantic.Field(default_factory=model),
            )

    return pydantic.create_model(
        "Model",
        **fields,
        __config__=pydantic.ConfigDict(
            arbitrary_types_allowed=True, validate_assignment=True
        ),
    )
