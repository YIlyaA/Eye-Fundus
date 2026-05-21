"""Ekstrakcja cech z patchów 5×5: statystyki RGB/G, momenty Hu, gradient Sobela."""
from __future__ import annotations

import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from .preprocessing import preprocess


# 3 (RGB mean) + 3 (RGB var) + 2 (G center, G center CLAHE)
# + 4 (G stats) + 7 (Hu) + 3 (mu20, mu02, mu11) + 2 (Sobel) = 24
N_FEATURES: int = 24


def _compute_hu_moments_batch(binary_patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """7 momentów Hu + [mu20, mu02, mu11] dla batcha patchy binarnych."""
    n = len(binary_patches)
    hu = np.empty((n, 7), dtype=np.float32)
    central = np.empty((n, 3), dtype=np.float32)
    for i, patch in enumerate(binary_patches):
        m = cv2.moments(patch.astype(np.uint8))
        hu[i] = cv2.HuMoments(m).ravel()
        central[i] = [m["mu20"], m["mu02"], m["mu11"]]
    return hu, central


def _features_from_batches(
    patches_rgb: np.ndarray,
    patches_g: np.ndarray,
    patches_gc: np.ndarray,
    patches_mag: np.ndarray,
) -> np.ndarray:
    """Składa pełny wektor cech (N, N_FEATURES) z patchy."""
    n, h, w = patches_g.shape
    cy, cx = h // 2, w // 2

    rgb_f = patches_rgb.astype(np.float32).reshape(n, h * w, 3)
    rgb_means = rgb_f.mean(axis=1)
    rgb_vars = rgb_f.var(axis=1)

    g_f = patches_g.astype(np.float32).reshape(n, h * w)
    g_center = patches_g[:, cy, cx].astype(np.float32)
    g_center_clahe = patches_gc[:, cy, cx].astype(np.float32)
    g_mean = g_f.mean(axis=1)
    g_var = g_f.var(axis=1)
    g_min = g_f.min(axis=1)
    g_max = g_f.max(axis=1)

    # Binaryzacja po średniej jasności - niezmiennicza względem oświetlenia.
    thr = g_mean[:, None]
    binary = (patches_g.reshape(n, h * w) > thr).reshape(n, h, w)
    hu, central = _compute_hu_moments_batch(binary)
    # log na Hu ze względu na ich ekstremalny zakres dynamiczny.
    hu_log = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    mag_f = patches_mag.reshape(n, h * w)
    mag_mean = mag_f.mean(axis=1)
    mag_var = mag_f.var(axis=1)

    X = np.empty((n, N_FEATURES), dtype=np.float32)
    i = 0
    X[:, i:i+3] = rgb_means; i += 3
    X[:, i:i+3] = rgb_vars;  i += 3
    X[:, i] = g_center;       i += 1
    X[:, i] = g_center_clahe; i += 1
    X[:, i] = g_mean;         i += 1
    X[:, i] = g_var;          i += 1
    X[:, i] = g_min;          i += 1
    X[:, i] = g_max;          i += 1
    X[:, i:i+7] = hu_log;     i += 7
    X[:, i:i+3] = central;    i += 3
    X[:, i] = mag_mean;       i += 1
    X[:, i] = mag_var;        i += 1
    assert i == N_FEATURES
    return X


def _prepare_image_maps(rgb: np.ndarray, fov: np.ndarray):
    """(green, green_clahe, sobel_magnitude). Liczone raz na obraz."""
    green = rgb[..., 1]
    green_clahe = preprocess(rgb, fov)
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return green, green_clahe, mag


def _windowed(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Sliding window view, bez kopiowania."""
    if arr.ndim == 2:
        return sliding_window_view(arr, (patch_size, patch_size))
    if arr.ndim == 3:
        return sliding_window_view(arr, (patch_size, patch_size, arr.shape[-1]))[..., 0, :, :, :]
    raise ValueError(f"Unsupported ndim={arr.ndim}")


def extract_features_for_points(
    rgb: np.ndarray,
    fov: np.ndarray,
    points: np.ndarray,
    patch_size: int = 5,
) -> np.ndarray:
    """Cechy dla wybranych środków patchy. Zwraca (N, N_FEATURES) float32."""
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size musi być nieparzysty, otrzymano {patch_size}")
    half = patch_size // 2

    green, green_clahe, mag = _prepare_image_maps(rgb, fov)
    win_rgb = _windowed(rgb, patch_size)
    win_g = _windowed(green, patch_size)
    win_gc = _windowed(green_clahe, patch_size)
    win_mag = _windowed(mag, patch_size)

    iy = points[:, 0].astype(np.int64) - half
    ix = points[:, 1].astype(np.int64) - half

    return _features_from_batches(
        win_rgb[iy, ix], win_g[iy, ix], win_gc[iy, ix], win_mag[iy, ix],
    )


def extract_features_grid(
    rgb: np.ndarray,
    fov: np.ndarray,
    patch_size: int = 5,
    stride: int = 1,
    batch_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Cechy dla wszystkich środków na siatce stride wewnątrz FOV. Batchowanie dla pamięci."""
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size musi być nieparzysty, otrzymano {patch_size}")
    half = patch_size // 2
    h, w = fov.shape

    green, green_clahe, mag = _prepare_image_maps(rgb, fov)
    win_rgb = _windowed(rgb, patch_size)
    win_g = _windowed(green, patch_size)
    win_gc = _windowed(green_clahe, patch_size)
    win_mag = _windowed(mag, patch_size)

    yy = np.arange(half, h - half, stride)
    xx = np.arange(half, w - half, stride)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    fov_at_grid = fov[grid_y, grid_x] > 0
    grid_y = grid_y[fov_at_grid].astype(np.int32)
    grid_x = grid_x[fov_at_grid].astype(np.int32)
    coords = np.column_stack([grid_y, grid_x])

    n = coords.shape[0]
    X = np.empty((n, N_FEATURES), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        iy = coords[start:end, 0].astype(np.int64) - half
        ix = coords[start:end, 1].astype(np.int64) - half
        X[start:end] = _features_from_batches(
            win_rgb[iy, ix], win_g[iy, ix], win_gc[iy, ix], win_mag[iy, ix],
        )
    return X, coords


def sample_points(
    fov: np.ndarray,
    manual: np.ndarray,
    n_samples: int,
    patch_size: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Zbalansowany sampling (50/50 naczynie/tło) wewnątrz FOV przez RandomUnderSampler."""
    if rng is None:
        rng = np.random.default_rng()
    half = patch_size // 2

    # Obcięcie FOV o margines pod patch 5×5 - specyfika projektu, biblioteka tego nie zrobi
    valid = fov > 0
    valid[:half, :] = False
    valid[-half:, :] = False
    valid[:, :half] = False
    valid[:, -half:] = False

    ys, xs = np.where(valid)
    coords = np.column_stack([ys, xs])
    labels = (manual[ys, xs] > 0).astype(np.uint8)

    n_per_class = n_samples // 2
    rus = RandomUnderSampler(
        sampling_strategy={0: n_per_class, 1: n_per_class},
        random_state=int(rng.integers(0, 2**31)),
    )
    return rus.fit_resample(coords, labels)


def build_dataset(
    image_ids: list[str],
    loader,
    n_samples_per_image: int = 5000,
    patch_size: int = 5,
    seed: int = 42,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Zbiera (X, y) z listy obrazów."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    iterator = tqdm(image_ids, desc="Build dataset") if progress else image_ids
    for image_id in iterator:
        rgb, manual, fov = loader(image_id)
        points, labels = sample_points(
            fov, manual, n_samples=n_samples_per_image,
            patch_size=patch_size, rng=rng,
        )
        X = extract_features_for_points(rgb, fov, points, patch_size=patch_size)
        Xs.append(X)
        ys.append(labels)
    return np.vstack(Xs), np.concatenate(ys)
