"""Aggregators: MIL pooling over a slide's tile features."""

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.mean import MeanAggregator


__all__ = ["Aggregator", "AttentionMIL", "MeanAggregator"]
