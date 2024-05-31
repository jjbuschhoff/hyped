"""Module for various data statistics.

This module provides a comprehensive collection of data statistics, which enable
the computation of aggregate values over the entire dataset. Unlike data processors,
which operate on individual examples, data statistics allow for the calculation of
statistics over the entire dataset. Additionally, this module ensures process-safe
computation, enabling statistics to be safely reduced across all processes.
"""
