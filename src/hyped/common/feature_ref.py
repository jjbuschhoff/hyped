from __future__ import annotations

import json
from enum import Enum
from functools import cached_property
from typing_extensions import Annotated, TypeAlias

from pydantic import (
    Field,
    BaseModel,
    AfterValidator,
    BeforeValidator,
    PlainSerializer,
    GetCoreSchemaHandler,
)
from pydantic_core import CoreSchema, core_schema
from datasets.features.features import Features, FeatureType, Sequence
from datasets.iterable_dataset import _batch_to_examples, _examples_to_batch

from .feature_checks import (
    get_sequence_feature,
    get_sequence_length,
    check_feature_equals,
    raise_feature_equals,
    raise_feature_is_sequence,
)


class FeatureKey(tuple[str | int | slice]):
    """Feature Key used to index features and examples.

    Arguments:
        *key (str | int | slice): key entries
    """

    @classmethod
    def from_tuple(cls, key: tuple[str | int | slice]) -> FeatureKey:
        """Generate a feature key from a tuple.

        Arguments:
            key (tuple[str|int|slice]): key entries

        Returns:
            key (FeatureKey): feature key
        """
        return cls(*key)

    def __new__(cls, *key: str | int | slice) -> None:
        """Instantiate a new feature key.

        Arguments:
            *key (tuple[str|int|slice]): key entries
        """
        if len(key) == 1 and isinstance(key[0], tuple):
            return FeatureKey.from_tuple(key[0])

        if len(key) > 0 and not isinstance(key[0], str):
            raise ValueError(
                "First entry of a feature key must be a string, got %s."
                % repr(key[0])
            )

        # unpack enums
        key = tuple(k.value if isinstance(k, Enum) else k for k in key)

        for key_entry in key:
            if not isinstance(key_entry, (str, int, slice)):
                raise TypeError(
                    "Feature key entries must be either str, int or slice "
                    "objects, got %s" % key_entry
                )

        return tuple.__new__(cls, key)
    
    def __getitem__(self, idx) -> FeatureKey | str | int | slice:
        """Get specific key entries of the feature key."""
        if isinstance(idx, slice) and (
            (idx.start == 0) or (idx.start is None)
        ):
            return FeatureKey(*super(FeatureKey, self).__getitem__(idx))
        return super(FeatureKey, self).__getitem__(idx)
    
    def __str__(self) -> str:
        """String representation of the feature key."""
        return "FeatureKey(%s)" % "->".join(map(repr, self))

    def __repr__(self) -> str:
        """String representation of the feature key."""
        return str(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Integrate feature key with pydantic."""
        return core_schema.no_info_after_validator_function(
            cls, handler(str | tuple)
        )
    
    def index_features(self, features: Features) -> FeatureType:
        """Get the feature type of the feature indexed by the key.

        Arguments:
            features (Features):
                The feature mapping to index with the given key.

        Returns:
            feature (FeatureType):
                the extracted feature type at the given key.
        """
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, str):
                # check feature type
                raise_feature_equals(self[:i], features, [Features, dict])
                # check key entry is present in features
                if key_entry not in features.keys():
                    raise KeyError(
                        "Key `%s` not present in features at `%s`, "
                        "valid keys are %s"
                        % (key_entry, self[:i], list(features.keys()))
                    )
                # get the feature at the key entry
                features = features[key_entry]

            elif isinstance(key_entry, (int, slice)):
                # check feature type
                raise_feature_is_sequence(self[:i], features)
                # get sequence feature and length
                length = get_sequence_length(features)
                features = get_sequence_feature(features)

                if isinstance(key_entry, int) and (
                    (length == 0) or ((length > 0) and (key_entry >= length))
                ):
                    raise IndexError(
                        "Index `%i` out of bounds for sequence of "
                        "length `%i` of feature at key %s"
                        % (key_entry, length, self[:i])
                    )

                if isinstance(key_entry, slice):
                    if length >= 0:
                        # get length of reminaing sequence after slicing
                        start, stop, step = key_entry.indices(length)
                        length = (stop - start) // step

                    # get features and pack them into a sequence of
                    # appropriate length
                    key = tuple.__new__(FeatureKey, self[i + 1 :])
                    return Sequence(
                        key.index_features(features), length=length
                    )

        return features

    def index_example(self, example: dict[str, Any]) -> Any:
        """Index the example with the key and retrieve the value.

        Arguments:
            example (dict[str, Any]): The example to index.

        Returns:
            value (Any): the value of the example at the given key.
        """
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, slice):
                assert isinstance(example, list)
                # recurse on all examples indexed by the slice
                # create a new subkey for recursion while avoiding
                # key checks asserting that the key must start with
                # a string entry
                key = tuple.__new__(FeatureKey, self[i + 1 :])
                return list(map(key.index_example, example[key_entry]))

            # index the example
            example = example[key_entry]

        return example

    def index_batch(self, batch: dict[str, list[Any]]) -> list[Any]:
        """Index batch.

        Index a batch of examples with the given key and retrieve
        the batch of values.

        Arguments:
            batch (dict[str, list[Any]]): Batch of example to index.

        Returns:
            values (list[Any]):
                the batch of values of the example at the given key.
        """
        return FeatureKey(self[0], slice(None), *self[1:]).index_example(batch)


    def __hash__(self) -> str:
        return hash(
            tuple(
                (k.start, k.stop, k.step) if isinstance(k, slice) else k
                for k in self
            )
        )


class FeatureRef(BaseModel):

    key: FeatureKey

    feature: Annotated[
        FeatureType,
        # custom serialization
        PlainSerializer(
            lambda f: json.dumps(
                Features({"feature": f}).to_dict()["feature"]
            ),
            return_type=str,
            when_used="unless-none"
        ),
        # custom deserialization
        BeforeValidator(
            lambda v: (
                v if isinstance(v, FeatureType)
                else Features.from_dict(
                    {"feature": json.loads(v)}
                )["feature"]
            )
        )
    ]

    node_id: int

    def __getattr__(self, key: str) -> FeatureRef:
        return self.__getitem__(key)

    def __getitem__(self, key: str | int | slice | FeatureKey) -> FeatureRef:
        key = FeatureKey(key)
        return FeatureRef(
            key=self.key + key,
            feature=key.index_features(self.feature),
            node_id=self.node_id
        )

    def __hash__(self) -> str:
        return hash((self.node_id, self.key))



class FeatureCollection(BaseModel):

    collection: Annotated[
        (
            dict[str, FeatureCollection | FeatureRef]
            | Annotated[list[FeatureCollection], Field(min_items=1)]
            | Annotated[list[FeatureRef], Field(min_items=1)]
        ),
        BeforeValidator(
            lambda x: (
                {
                    k: (
                        v if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for k, v in x.items()
                } if isinstance(x, dict) else
                [
                    (
                        v if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for v in x
                ] if isinstance(x, list) else
                x
            )
        )
    ]

    def __init__(
        self,
        collection: (
            None
            | dict[str, FeatureCollection | FeatureRef]
            | list[FeatureCollection]
            | list[FeatureRef]
        ) = None,
        **kwargs
    ) -> None:

        if collection is None and len(kwargs) == 0:
            raise TypeError()

        if collection is not None and len(kwargs) != 0:
            raise TypeError()

        super(FeatureCollection, self).__init__(
            collection=collection if collection is not None else kwargs
        )
    
    def __post_init__(self) -> None:
        # check valid feature collection types
        self.feature

    def __hash__(self) -> str:
        if isinstance(self.collection, dict):
            return hash(tuple(map(tuple, self.collection.items())))
        if isinstance(self.collection, list):
            return hash(tuple(self.collection))

    @cached_property
    def feature(self) -> FeatureType:
        
        if isinstance(self.collection, dict):
            return Features(
                {
                    k: v.feature
                    for k, v in self.collection.items()
                }
            )

        if isinstance(self.collection, list):
            # collect all features in specified in the list
            collected_features = (
                item.feature for item in self.collection
            )
            f = next(collected_features)
            # make sure the feature types match
            for ff in collected_features:
                if not check_feature_equals(f, ff):
                    raise TypeError(
                        "Expected all items of a sequence to be of the "
                        "same feature type, got %s != %s" % (str(f), str(ff))
                    )
            return Sequence(f, length=len(self.collection))

        raise TypeError()

    @property
    def refs(self) -> set[FeatureRef]:

        feature_refs = set()

        for v in (
            self.collection.values() if isinstance(self.collection, dict) else self.collection
        ):
            if isinstance(v, FeatureCollection):
                feature_refs.update(v.refs)
            elif isinstance(v, FeatureRef):
                feature_refs.add(v)

        return feature_refs


Feature: TypeAlias = FeatureRef | FeatureCollection
