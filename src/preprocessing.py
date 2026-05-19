"""Wstępne przetwarzanie: kanał zielony, wypełnienie poza FOV, CLAHE."""
from __future__ import annotations

import numpy as np
import cv2


def to_green_channel(rgb: np.ndarray) -> np.ndarray:
    """Kanał G z obrazu RGB jako 2D uint8."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Oczekiwano RGB (H,W,3), otrzymano {rgb.shape}")
    return rgb[..., 1].copy()


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE — lokalne wyrównanie histogramu z ograniczeniem wzmocnienia."""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Korekcja gamma przez LUT."""
    if gamma == 1.0:
        return gray.copy()
    lut = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(gray.astype(np.uint8), lut)


def fill_outside_fov(
    gray: np.ndarray,
    fov: np.ndarray,
    fill_value: int | None = None,
) -> np.ndarray:
    """Zastępuje piksele poza FOV medianą wewnątrz — usuwa krawędź ramki."""
    out = gray.copy()
    inside = fov > 0
    if fill_value is None:
        fill_value = int(np.median(out[inside]))
    out[~inside] = np.array(fill_value).astype(out.dtype)
    return out


def preprocess(
    rgb: np.ndarray,
    fov: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """green → fill_outside_fov → CLAHE."""
    g = to_green_channel(rgb)
    g = fill_outside_fov(g, fov)
    return apply_clahe(g, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
