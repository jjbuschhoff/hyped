"""Provides the aggregation reference class."""

from pydantic import BaseModel


class DataAggregationRef(BaseModel):
    """Reference to a data aggregation node in the data flow.

    Attributes:
        node_id_ (int): The ID of the aggregation node.
        flow_ (object): The data flow object.
        type_ (type): The type of the aggregation value.
    """

    node_id_: int
    """The ID of the aggregation node."""

    flow_: object
    """The data flow object."""

    type_: type
    """The type of the aggregation value."""
