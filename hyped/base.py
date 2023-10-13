from __future__ import annotations
import json
from abc import ABC, ABCMeta
from dataclasses import dataclass, asdict
from types import MappingProxyType
from typing import Literal, Any


class __TypeRegisterMeta(ABCMeta):
    _hash_register = dict()  # maps type-ids to type-hashes
    _type_register = dict()  # maps type-hashes to types

    def register_type(cls, T):
        h = hash(T)
        cls._hash_register[T.t] = h
        cls._type_register[h] = T

    def __new__(cls, name, bases, attrs) -> None:
        # create new type and register it
        T = super().__new__(cls, name, bases, attrs)
        cls.register_type(cls, T)
        # return new type
        return T


class TypeRegister(ABC, metaclass=__TypeRegisterMeta):
    """Type Register Base Class

    Tracks all types that inherit from a class.

    Attributes:
        t (ClassVar[str]): type identifier
    """

    t: Literal["base"] = "base"

    @classmethod
    @property
    def hash_register(cls) -> dict[int, type]:
        """Immutable hash register mapping type id to the
        corresponding type hash
        """
        return MappingProxyType(cls._hash_register)

    @classmethod
    @property
    def type_register(cls) -> dict[int, type]:
        """Immutable type register mapping type hash to the
        corresponding type
        """
        return MappingProxyType(cls._type_register)

    @classmethod
    @property
    def type_ids(cls) -> list[type]:
        """List of all registered type identifiers"""
        return list(cls.hash_register.keys())

    @classmethod
    @property
    def types(cls) -> list[type]:
        """List of all registered types"""
        return list(cls.type_register.values())

    @classmethod
    def get_type_by_t(cls, t: str) -> type:
        """Get registered type by type id

        Arguments:
            t (int): type identifier

        Returns:
            T (type): type corresponding to `t`
        """
        # check if type id is present in register
        if t not in cls.hash_register:
            raise ValueError(
                "Type id '%s' not registered, registered type ids: %s"
                % (t, ", ".join(cls.type_ids))
            )
        # get type corresponding to id
        return cls.get_type_by_hash(cls.hash_register[t])

    @classmethod
    def get_type_by_hash(cls, h: int) -> type:
        """Get registered type by type hash

        Arguments:
            h (int): type hash

        Returns:
            T (type): registered type corresponding to `h`
        """
        # check if hash is present in register
        if h not in cls.type_register:
            raise TypeError(
                "No type found matching hash %s, registered types: %s"
                % (str(h), ", ".join(list(map(str, cls.types))))
            )
        # get type corresponding to hash
        return cls.type_register[h]


@dataclass
class BaseConfig(TypeRegister):
    t: Literal["base.config"] = "base.config"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration object to dictionary"""
        return asdict(self) | {"__type_hash__": hash(type(self))}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> BaseConfig:
        if "__type_hash__" in dct:
            # create a copy to avoid removing entries from the original
            # pop hash from dict as its meta data and not a valid field
            dct = dct.copy()
            h = dct.pop("__type_hash__")
            # get type from registry
            T = cls.get_type_by_hash(h)

        elif "t" in dct:
            # similar to hash above, pop type id from dict as its
            # a class variable and not a field of the config
            dct = dct.copy()
            t = dct.pop("t")
            # get type from type id
            T = cls.get_type_by_t(t)

        else:
            # fallback to class on which the function is called
            T = cls

        # create instance
        return T(**dct)

    def serialize(self, **kwargs) -> str:
        """Serialize config object into json format

        Arguments:
            **kwargs: arguments forwarded to `json.dumps`

        Returns:
            serialized_config (str): the serialized configuration string
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def deserialize(cls, serialized: str) -> BaseConfig:
        return cls.from_dict(json.loads(serialized))
