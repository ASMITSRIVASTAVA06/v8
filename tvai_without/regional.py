"""
TerraVision AI — regional brightening correction.

Rural electrification makes whole districts glow a little brighter every year, so
"current minus 5-year median" is positive almost everywhere and floods the map with
false alerts. A real encroachment is different: a small bright patch on a background
that is NOT brightening. This module estimates the smooth regional trend of the
radiance change and removes it, so detection looks for patches that stand out from
their own surroundings.

The trend is a median over coarse blocks (default 16 x 16 pixels, about 8 km), which a
bright cluster cannot drag upward unless it covers more than half of a block. Only
brightening is removed: a region getting darker never creates extra alerts.
Needs NumPy and SciPy only.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage


def regional_trend(delta: np.ndarray, block: int = 16) -> np.ndarray:
    """Smooth, non-negative estimate of the regional brightening, same shape as `delta`."""
    h, w = delta.shape
    if h < block or w < block:                       # tiny raster: one global median
        valid = delta[np.isfinite(delta)]
        return np.full(delta.shape, max(float(np.median(valid)), 0.0) if valid.size else 0.0)

    ph, pw = -h % block, -w % block                  # pad up to whole blocks with NaN
    padded = np.pad(delta.astype(float), ((0, ph), (0, pw)), constant_values=np.nan)
    H, W = padded.shape
    blocks = padded.reshape(H // block, block, W // block, block).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(H // block, W // block, block * block)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN blocks are handled just below
        finite_count = np.isfinite(blocks).sum(axis=2)
        coarse = np.nanmedian(np.where(np.isfinite(blocks), blocks, np.nan), axis=2)
    coarse[finite_count < (block * block) // 4] = np.nan      # mostly empty block: no opinion

    valid = np.isfinite(coarse)
    if not valid.any():
        return np.zeros(delta.shape)
    fill = float(np.median(coarse[valid]))
    coarse = np.where(valid, coarse, fill)
    coarse = ndimage.uniform_filter(coarse, size=3, mode="nearest")

    # Upsample smoothly back to pixel resolution, then crop the padding
    zoom = (H / coarse.shape[0], W / coarse.shape[1])
    trend = ndimage.zoom(coarse, zoom, order=1, mode="nearest")[:h, :w]
    return np.maximum(trend, 0.0)


def correct_delta(delta: np.ndarray, block: int = 16) -> np.ndarray:
    """Radiance change with the regional brightening trend removed."""
    return delta - regional_trend(delta, block)
