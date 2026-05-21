"""Overlay maski na obraz i siatka panelów do porównań."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """Koloruje piksele maski na obrazie RGB z mieszaniem α."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Oczekiwano RGB, otrzymano {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Rozmiar maski {mask.shape} != obraz {image.shape[:2]}")

    result = image.copy()
    m = mask > 0
    color_arr = np.array(color, dtype=np.float32)
    result[m] = (
        (1.0 - alpha) * result[m].astype(np.float32) + alpha * color_arr
    ).clip(0, 255).astype(np.uint8)
    return result


def compare_grid(
    panels: Sequence[tuple[str, np.ndarray]],
    ncols: int = 4,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    suptitle: str | None = None,
) -> Figure:
    """Siatka panelów (tytuł, obraz). cmap='gray' dobierany automatycznie dla 2D."""
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (title, img) in zip(axes_flat, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def metrics_to_markdown(
    rows: dict[str, dict[str, float]],
    columns: Sequence[str] = ("accuracy", "sensitivity", "specificity", "g_mean", "arith_mean"),
) -> str:
    """Tabela markdown z metryk - do wstawienia w raport."""
    return pd.DataFrame(rows).T.reindex(columns=list(columns)).round(4).to_markdown()
