"""Baseline: preprocessing -> Frangi -> Otsu -> morfologia. Bez uczenia."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.exposure import rescale_intensity

from .preprocessing import preprocess


@dataclass
class BaselineResult:
    """Wszystkie kroki pipeline'u — do wizualizacji w notebooku."""

    preprocessed: np.ndarray
    frangi_response: np.ndarray
    binary_raw: np.ndarray
    mask: np.ndarray


def frangi_response(
    gray: np.ndarray,
    sigmas: tuple[float, ...] = (1, 2, 3, 4, 5),
    scale: float = 1.0,
) -> np.ndarray:
    """Filtr Frangiego z opcjonalnym downsamplingiem. Zwraca float32 znormalizowany do [0,1]."""
    h, w = gray.shape

    if scale != 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Skalujemy sigmy, żeby analizować naczynia tej samej fizycznej grubości.
        sigmas_eff = tuple(s * scale for s in sigmas)
        response_small = frangi(small, sigmas=sigmas_eff, black_ridges=True).astype(
            np.float32
        )
        response = cv2.resize(response_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        response = frangi(gray, sigmas=sigmas, black_ridges=True).astype(np.float32)

    return rescale_intensity(response, out_range=(0.0, 1.0)).astype(np.float32)


def threshold_in_fov(
    response: np.ndarray,
    fov: np.ndarray,
    method: str = "otsu",
    fixed_value: float = 0.1,
) -> np.ndarray:
    """Binaryzacja progiem Otsu liczonym TYLKO w FOV (albo progiem stałym)."""
    f_mask = fov > 0
    if method == "otsu":
        inside_values = response[f_mask]
        if inside_values.min() == inside_values.max():
            thr = 0.0
        else:
            thr = float(threshold_otsu(inside_values))
    elif method == "fixed":
        thr = float(fixed_value)
    else:
        raise ValueError(f"Nieznana metoda progowania: {method!r}")
    return ((response > thr) & f_mask).astype(np.uint8)


def clean_mask(
    binary: np.ndarray,
    min_size: int = 60,
    closing_radius: int = 1,
) -> np.ndarray:
    """Usunięcie małych komponentów, potem zamknięcie. Kolejność istotna."""
    m = binary.astype(bool)
    # max_size=min_size-1 — równoważne staremu min_size (usuwa komponenty < min_size)
    m = remove_small_objects(m, max_size=min_size - 1)
    if closing_radius > 0:
        m = closing(
            m, footprint=disk(closing_radius)
        )  # zamyka małe przerwy w naczyniach
    return m.astype(np.uint8)


def predict(
    rgb: np.ndarray,
    fov: np.ndarray,
    sigmas: tuple[float, ...] = (1, 2, 3, 4, 5),
    frangi_scale: float = 0.5,
    threshold_method: str = "otsu",
    fixed_threshold: float = 0.1,
    min_size: int = 60,
    closing_radius: int = 1,
) -> BaselineResult:
    """Pełny pipeline baseline. Zwraca BaselineResult z krokami pośrednimi."""
    pre = preprocess(rgb, fov)
    resp = frangi_response(pre, sigmas=sigmas, scale=frangi_scale)
    raw = threshold_in_fov(
        resp, fov, method=threshold_method, fixed_value=fixed_threshold
    )
    cleaned = clean_mask(raw, min_size=min_size, closing_radius=closing_radius)
    cleaned = (cleaned & (fov > 0)).astype(np.uint8)
    return BaselineResult(
        preprocessed=pre, frangi_response=resp, binary_raw=raw, mask=cleaned
    )


def predict_mask(rgb: np.ndarray, fov: np.ndarray, **kwargs) -> np.ndarray:
    """Sama maska — wspólna sygnatura dla wszystkich metod w app.ipynb."""
    return predict(rgb, fov, **kwargs).mask
