"""Random Forest na patchach 5×5 (etap 2, ocena 4)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import joblib
from tqdm import tqdm

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

from .features import extract_features_grid


def make_pipeline(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
    sampling_strategy: str | float = "auto",
) -> ImbPipeline:
    """RandomUnderSampler → RandomForestClassifier. Pipeline z imblearn, nie sklearn."""
    return ImbPipeline(steps=[
        ("undersample", RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])


def predict_mask(
    model,
    rgb: np.ndarray,
    fov: np.ndarray,
    patch_size: int = 5,
    stride: int = 1,
    batch_size: int = 200_000,
    progress: bool = True,
) -> np.ndarray:
    """Sliding window po FOV, predykcja batchami, opcjonalny stride+dylacja."""
    h, w = fov.shape

    X, coords = extract_features_grid(
        rgb, fov, patch_size=patch_size, stride=stride, batch_size=batch_size,
    )

    n = X.shape[0]
    preds = np.empty(n, dtype=np.uint8)
    iterator = range(0, n, batch_size)
    if progress:
        iterator = tqdm(list(iterator), desc=f"predict (n={n:,})")
    for start in iterator:
        end = min(start + batch_size, n)
        preds[start:end] = model.predict(X[start:end]).astype(np.uint8)

    if stride == 1:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[coords[:, 0], coords[:, 1]] = preds
    else:
        # Rzadka mapa + dylacja zastępują kazdą komórkę stride×stride wartością środka.
        sparse = np.zeros((h, w), dtype=np.uint8)
        sparse[coords[:, 0], coords[:, 1]] = preds
        kernel = np.ones((stride, stride), dtype=np.uint8)
        mask = cv2.dilate(sparse, kernel, anchor=(0, 0), iterations=1)

    return (mask & (fov > 0)).astype(np.uint8)


def save_model(model, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: Path | str):
    return joblib.load(path)
