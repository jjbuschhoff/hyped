"""The core package for defining and executing data flows as directed acyclic graphs (DAGs).

This package provides the necessary classes and methods to construct, manage, and execute
complex data processing workflows. It uses a graph-based approach where each node
represents a data processor, and edges represent the flow of data between processors.

Modules:
    - :class:`executor`: Manages the execution of the data flow graph.
    - :class:`flow`: Provides the high-level interface for defining data processing workflows.
    - :class:`graph`: Defines the structure of the data flow graph and its components.
    - :class:`nodes`: Defines the base classes for nodes of the DAG.
    - :class:`refs`: Defines reference objects as well as input/output helpers for processors.

While these modules are crucial for processor development, they are not intended for
direct use by end users interacting with the high-level data flow interface.
"""
