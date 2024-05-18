from __future__ import annotations

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from datasets.features.features import Features, FeatureType, Sequence

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
        if len(self) == 0:
            return batch

        return FeatureKey(self[0], slice(None), *self[1:]).index_example(batch)


    def __hash__(self) -> int:
        return hash(
            tuple(
                (k.start, k.stop, k.step) if isinstance(k, slice) else k
                for k in self
            )
        )