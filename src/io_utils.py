"""Wczytywanie obrazów HRF - RGB + maska eksperta + maska FOV."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from .config import IMAGES_DIR, MANUAL_DIR, FOV_DIR


def _find_first(directory: Path, candidates: list[str]) -> Path:
    """Zwraca pierwszą istniejącą ścieżkę z listy."""
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    available = sorted(p.name for p in directory.iterdir()) if directory.exists() else []
    raise FileNotFoundError(
        f"Nie znaleziono żadnego z {candidates} w {directory}. Zawartość: {available[:10]}..."
    )


def load_image(image_id: str) -> np.ndarray:
    """RGB obraz dna oka, (H, W, 3) uint8."""
    # W HRF wielkość rozszerzenia jest niestała.
    path = _find_first(IMAGES_DIR, [f"{image_id}.jpg", f"{image_id}.JPG"])
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_manual(image_id: str) -> np.ndarray:
    """Maska ekspercka naczyń, (H, W) uint8 w {0, 1}."""
    path = _find_first(MANUAL_DIR, [
        f"{image_id}.tif", f"{image_id}.tiff",
        f"{image_id}.png", f"{image_id}.gif",
    ])
    return _to_binary(np.asarray(Image.open(path).convert("L")))


def load_fov(image_id: str) -> np.ndarray:
    """Maska pola widzenia, (H, W) uint8 w {0, 1}."""
    path = _find_first(FOV_DIR, [
        f"{image_id}_mask.tif", f"{image_id}_mask.tiff", f"{image_id}_mask.png",
        f"{image_id}.tif", f"{image_id}.tiff", f"{image_id}.png",
    ])
    return _to_binary(np.asarray(Image.open(path).convert("L")))


def load_triplet(image_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image, manual, fov) ze sprawdzeniem rozmiarów."""
    image = load_image(image_id)
    manual = load_manual(image_id)
    fov = load_fov(image_id)
    h, w = image.shape[:2]
    if manual.shape != (h, w) or fov.shape != (h, w):
        raise ValueError(
            f"Rozmiary nie zgadzają się dla {image_id}: "
            f"image={image.shape}, manual={manual.shape}, fov={fov.shape}"
        )
    return image, manual, fov


def _to_binary(mask: np.ndarray) -> np.ndarray:
    # Maski HRF są {0, 255}; próg > 0 działa też dla {0, 1}.
    return (mask > 0).astype(np.uint8)
