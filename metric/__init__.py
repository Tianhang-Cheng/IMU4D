"""Canonical quantitative metrics for IMU4D experiments.

The package intentionally keeps the numerical metrics independent of model code.
See :mod:`metric.motion`, :mod:`metric.text`, and :mod:`metric.scene` for the
evaluation groups used by the paper.
"""

from .core.common import summarize_records

__all__ = ["summarize_records"]
