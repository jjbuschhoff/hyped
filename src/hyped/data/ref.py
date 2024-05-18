"""Provides a class for referencing specific features within a data flow graph.

The FeatureRef module defines the `FeatureRef` class, which represents references to features
within a data processing flow. These references are used to specify and retrieve nested features
within the data flow graph.

The `FeatureRef` class supports both attribute-style and index-style access to sub-features,
enabling convenient navigation and manipulation of complex data structures within the data flow.
"""

from __future__ import annotations

import json

from datasets.features.features import Features, FeatureType, Sequence
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing_extensions import Annotated

from hyped.common.feature_key import FeatureKey


class FeatureRef(BaseModel):
    """A reference to a specific feature within a data flow graph.

    FeatureRef objects represent references to features within a data processing flow.
    These objects are used when defining a data flow but are not instantiated manually.
    Instead instances are provided by the data flow system.

    The class supports dynamic access to sub-features, enabling the specification and
    retrieval of nested features using both attribute-style and index-style access.

    Example:
        Assuming `ref` is an instance of FeatureRef:

        .. code-block: python

            sub_ref_attr = ref.some_feature
            sub_ref_index = ref['some_feature']

        Both approaches will return a new FeatureRef instance pointing to the sub-feature
        within the data flow.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key_: FeatureKey
    """
    The key identifying the specific feature within the data flow.

    It is part of the tuple (flow_, node_id_, key_), which indexes a
    specific feature uniquely in the context of a data flow. The key
    is used to locate and access the feature within a single node of
    a data flow.
    """

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
    """
    The feature or features represented by this FeatureRef instance.
    """

    node_id_: int
    """
    The identifier of the node within the data flow graph.

    This attribute represents the identifier of the node within the data flow graph
    to which the referenced feature belongs. It is part of the tuple (flow_, node_id_, key_),
    which uniquely indexes a specific feature in the context of a data flow.
    """

    flow_: object
    """
    The data flow graph to which the feature reference belongs.

    This attribute represents the data flow graph to which the feature reference belongs.
    It provides context for the feature reference within the overall data processing flow.
    """

    def __hash__(self) -> str:
        """Compute the hash value of the FeatureRef instance.

        Note that the hash value of a FeatureRef instance is independent of
        the feature type, it only considers the index (flow_, node_id_, key_)
        of the feature.

        Returns:
            str: The hash value of the FeatureRef instance, computed
                based on its attributes.
        """
        return hash((self.flow_, self.node_id_, self.key_))

    def __getattr__(self, key: str) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via attribute-style access.

        Args:
            key (str): The name of the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.
        """
        if key.startswith("_"):
            return object.__getitem__(self, key)

        return self.__getitem__(key)

    def __getitem__(self, key: str | int | slice | FeatureKey) -> FeatureRef:
        """Access a sub-feature within the FeatureRef instance via index-style access.

        Args:
            key (str | int | slice | FeatureKey): The index or key of the sub-feature to access.

        Returns:
            FeatureRef: A new FeatureRef instance representing the accessed sub-feature.
        """
        key = key if isinstance(key, tuple) else (key,)
        key = tuple.__new__(FeatureKey, key)
        return FeatureRef(
            key_=self.key_ + key,
            feature_=key.index_features(self.feature_),
            node_id_=self.node_id_,
            flow_=self.flow_,
        )
