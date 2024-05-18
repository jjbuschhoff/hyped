from __future__ import annotations

import json

from datasets.features.features import Features, FeatureType, Sequence
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
)
from typing_extensions import Annotated

from hyped.common.feature_key import FeatureKey


class FeatureRef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key_: FeatureKey

    feature_: None | Annotated[
        Features | FeatureType,
        # custom serialization
        PlainSerializer(
            lambda f: json.dumps(
                Features({"feature": f}).to_dict()["feature"]
            ),
            return_type=str,
            when_used="unless-none",
        ),
        # custom deserialization
        BeforeValidator(
            lambda v: (
                Features(v)
                if isinstance(v, dict)
                else Sequence(v)
                if isinstance(v, list)
                else v
                if isinstance(v, FeatureType)
                else Features.from_dict({"feature": json.loads(v)})["feature"]
            )
        ),
    ]

    node_id_: int
    flow_: object

    def __hash__(self) -> str:
        return hash((self.flow_, self.node_id_, self.key_))

    def __getattr__(self, key: str) -> FeatureRef:
        if key.startswith("_"):
            return object.__getitem__(self, key)

        return self.__getitem__(key)

    def __getitem__(self, key: str | int | slice | FeatureKey) -> FeatureRef:
        key = key if isinstance(key, tuple) else (key,)
        key = tuple.__new__(FeatureKey, key)
        return FeatureRef(
            key_=self.key_ + key,
            feature_=key.index_features(self.feature_),
            node_id_=self.node_id_,
            flow_=self.flow_,
        )
