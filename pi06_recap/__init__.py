"""Utilities for a pi0.6-style RECAP reproduction.

This package intentionally avoids depending on openpi internals. It prepares
value-function data, trains/scores a lightweight stitched VF, and exports an
advantage lookup that a small openpi patch can consume during policy training.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
