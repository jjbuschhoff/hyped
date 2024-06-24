"""Nested Container."""

from __future__ import annotations

from functools import cache
from typing import Any, Callable, Generic, Hashable, Mapping, TypeVar

from pydantic import BaseModel, field_validator

T = TypeVar("T")
U = TypeVar("U")


class NestedContainer(BaseModel, Generic[T]):
    """A container for nested data structures.

    This class is used to represent nested data structures such as dictionaries and lists.
    It provides methods to map over its elements and flatten them into a dictionary.
    """

    data: T | dict[Hashable, NestedContainer[T]] | list[NestedContainer[T]]
    """The nested data structure of the container."""

    @field_validator("data", mode="before")
    def _validate_data(
        cls, data: Any
    ) -> T | dict[Hashable, NestedContainer[T]] | list[NestedContainer[T]]:
        """Pre-validation method for the data attribute.

        Args:
            data (Any): The raw data to parse.

        Returns:
            T | dict[Hashable, NestedContainer[T]] | list[NestedContainer[T]]:
            The parsed nested data structure.
        """
        return (
            {
                k: v if isinstance(v, NestedContainer) else cls(data=v)
                for k, v in data.items()
            }
            if isinstance(data, Mapping)
            else [
                v if isinstance(v, NestedContainer) else cls(data=v)
                for v in data
            ]
            if isinstance(data, list)
            else data
        )

    def map(
        self,
        f: Callable[[tuple[Hashable | int], T], U],
        target_type: type[U],
        _path: tuple[Hashable | int] = tuple(),
    ) -> NestedContainer[U]:
        """Map a function over the container's elements.

        Args:
            f (Callable[[tuple[Hashable | int], T], U]): The function to apply.
            target_type (type[U]): The type of the resulting elements.
            _path (tuple[Hashable | int], optional): The prefix path of the
                container container. Defaults to tuple().

        Returns:
            NestedContainer[U]: The container with the mapped elements.
        """
        if isinstance(self.data, dict):
            return NestedContainer[target_type](
                data={
                    k: v.map(f, target_type, _path=_path + (k,))
                    for k, v in self.data.items()
                }
            )

        if isinstance(self.data, list):
            return NestedContainer[target_type](
                data=[
                    v.map(f, target_type, _path=_path + (i,))
                    for i, v in enumerate(self.data)
                ]
            )

        return NestedContainer[target_type](data=f(_path, self.data))

    def flatten(self) -> dict[tuple[Hashable | int], T]:
        """Flatten the nested container into a dictionary.

        Returns:
            dict[tuple[Hashable | int], T]: The flattened dictionary.
        """
        # collect all values in the flattened dictionary with
        # the key being the corresponding path
        flattened = {}
        self.map(flattened.__setitem__, None)
        # return the flat dictionary
        return flattened

    def unpack(self) -> dict | list | T:
        """Unpack the nested container into its raw form.

        Returns:
            dict | list | T: The unpacked data.
        """
        if isinstance(self.data, dict):
            return {k: v.unpack() for k, v in self.data.items()}

        if isinstance(self.data, list):
            return [v.unpack() for v in self.data]

        return self.data
