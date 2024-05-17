from typing import Callable
from functools import partial

from pydantic import AfterValidator
from datasets.features.features import FeatureType

from hyped.data.ref import FeatureRef
from hyped.common.pydantic import BaseModelWithTypeValidation
from hyped.common.feature_key import FeatureKey
from hyped.common.feature_checks import (
    raise_feature_equals,
    raise_feature_is_sequence,
    get_sequence_length
)

class FeatureValidator(AfterValidator):

    def __init__(self, f: Callable[[FeatureRef, FeatureType], None]) -> None:
        
        def check(ref: FeatureRef) -> FeatureRef:
            if not isinstance(ref, FeatureRef):
                raise TypeError()

            try:
                f(ref, ref.feature_)
            except TypeError as e:
                raise TypeError(ref) from e
            
            return ref

        super(FeatureValidator, self).__init__(check)


class CheckFeatureEquals(FeatureValidator):

    def __init__(self, feature_type: FeatureType | list[FeatureType]) -> None:
                
        super(CheckFeatureEquals, self).__init__(
            partial(
                raise_feature_equals,
                target=feature_type
            )
        )


class CheckFeatureIsSequence(FeatureValidator):

    def __init__(
        self,
        value_type: None | FeatureType | list[FeatureType],
        length: int = -1
    ) -> None:
        
        def check(key: FeatureKey, feature: FeatureType) -> None:
            raise_feature_is_sequence(key, feature, value_type)

            if -1 != length != get_sequence_length(feature):
                raise TypeError(
                    "Expected `%s` to be a sequence of length %i, got %i"
                    % (str(key), length, get_sequence_length(feature))
                )
            return ref

        super(CheckFeatureEquals, self).__init__(check)


class InputRefs(BaseModelWithTypeValidation):

    @classmethod
    def type_validator(cls) -> None:

        for name, field in cls.model_fields.items():
            # each field should be a feature ref with
            # an feature validator annotation
            if not (
                issubclass(field.annotation, FeatureRef)
                and len(field.metadata) == 1
                and isinstance(field.metadata[0], FeatureValidator)
            ):
                raise TypeError(name)

    @property
    def refs(self) -> set[FeatureRef]:
        return set(self.named_refs.values())

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        return {key: getattr(self, key) for key in self.model_fields.keys()}

    @property
    def flow(self) -> "DataFlowGraph":
        # assumes that all feature refs refer to the same flow
        # this is checked later when a processor is added to the flow
        return getattr(self, next(iter(self.model_fields.keys()))).flow_

