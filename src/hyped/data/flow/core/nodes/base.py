"""Provides base classes for nodes in a data flow graph.

This module defines base classes for nodes in a data flow graph. It
includes a base configuration class (:class:`BaseNodeConfig`) and a
generic base class (:class:`BaseNode`) for defining nodes with
configurable input and output types.

Classes:
    - :class:`BaseNodeConfig`: Base configuration class for nodes in a data flow graph.
    - :class:`BaseNode`: Base class for nodes in a data flow graph.
"""
from typing import Generic, TypeVar

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.base.generic import solve_typevar

from ..refs.inputs import InputRefs
from ..refs.outputs import OutputRefs


class BaseNodeConfig(BaseConfig):
    """Base configuration class for nodes in a data flow graph."""


C = TypeVar("C", bound=BaseNodeConfig)
I = TypeVar("I", bound=None | InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseNode(BaseConfigurable[C], Generic[C, I, O]):
    """Base class for nodes in a data flow graph.

    Attributes:
        _in_refs_type (Type[I]): The type of input references expected by the node.
        _out_refs_type (Type[O]): The type of output references produced by the node.
    """

    def __init__(self, config: None | C = None, **kwargs) -> None:
        """Initialize the node with the given configuration.

        Args:
            config (None | C, optional): The configuration object for the node.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(BaseNode, self).__init__(config, **kwargs)
        # get input and output reference types from typevars
        self._in_refs_type = solve_typevar(type(self), I)
        self._out_refs_type = solve_typevar(type(self), O)
