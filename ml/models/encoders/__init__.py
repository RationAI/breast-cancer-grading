"""Encoders: per-tile feature extractors for the MIL pipeline."""

from ml.models.encoders.base import Encoder
from ml.models.encoders.identity import IdentityEncoder


__all__ = ["Encoder", "IdentityEncoder"]
