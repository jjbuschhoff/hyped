"""Provides base classes and helpers for defining data processors in a data flow graph.

The Base module defines fundamental classes and helper functions for building
data processors, configuring them, and managing their inputs and outputs in a
data flow graph.

Classes:
    - `BaseDataProcessor`: Base class for data processors in a data flow graph.
    - `BaseDataProcessorConfig`: Base configuration class for data processors.

Modules:
    - `inputs`: Contains classes and functions for managing input references.
    - `outputs`: Contains classes and functions for managing output references.
"""

from . import inputs, outputs
from .config import BaseDataProcessorConfig
from .processor import BaseDataProcessor, Batch, Sample
