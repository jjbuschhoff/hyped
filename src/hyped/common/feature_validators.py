from functools import partial
from typing import Callable

from pydantic import AfterValidator
from datasets.features.features import FeatureType

from .feature_ref import Feature, FeatureRef, FeatureKey
from .feature_checks import (
    raise_feature_equals,
    raise_feature_is_sequence,
    get_sequence_length
)


class FeatureValidator(AfterValidator):

    def __init__(self, f: Callable[[Feature, FeatureType], None]) -> None:
        
        def check(ref: Feature) -> FeatureRef:
            if not isinstance(ref, Feature):
                raise TypeError()

            try:
                f(ref, ref.feature)
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
