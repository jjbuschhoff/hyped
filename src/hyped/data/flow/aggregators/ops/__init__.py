"""Provides operator-like data aggregators for performing common statistical operations.

This module includes data aggregators that perform basic statistical operations such as
:code:`sum`, :code:`mean`, and :code:`standard deviation` on input features. These aggregators are
designed to be used within data processing pipelines to compute dataset-wide statistics. Each
aggregator follows a standard interface for initialization, extraction of values from batches,
and updating the aggregated results.
"""
