"""Shared type aliases for breast-cancer grading training."""

from typing import TypedDict

from torch import Tensor


class TileMetadata(TypedDict):
    """Metadata for a tile, including its slide ID and coordinates."""

    slide_id: str
    x: int
    y: int


type TilesPredictSample = tuple[Tensor, TileMetadata]


class SlideMetadata(TypedDict):
    """Metadata for a slide-level MIL sample."""

    slide_id: str


type Bag = Tensor
"""A bag of per-slide items: tile features ``(N, D)`` or raw tiles ``(N, C, H, W)``."""

type MILSample = tuple[Bag, Tensor, SlideMetadata]
"""A slide-level training sample: ``(bag, label, metadata)``.

``label`` is the Nottingham grade encoded as a class index: ``0`` = no tumour
epithelium (negative slide), ``1`` = G1, ``2`` = G2, ``3`` = G3.
"""
