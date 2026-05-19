from __future__ import annotations
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data" / "HRF"
IMAGES_DIR: Path = DATA_DIR / "images"
MANUAL_DIR: Path = DATA_DIR / "manual1"
FOV_DIR: Path = DATA_DIR / "mask"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
REPORT_DIR: Path = PROJECT_ROOT / "report"

HEALTHY_IDS: list[str] = [f"{i:02d}_h" for i in range(1, 16)]
DR_IDS: list[str] = [f"{i:02d}_dr" for i in range(1, 16)]
GLAUCOMA_IDS: list[str] = [f"{i:02d}_g" for i in range(1, 16)]

ALL_IMAGES: list[str] = HEALTHY_IDS + DR_IDS + GLAUCOMA_IDS


TEST_IMAGES: list[str] = [
    "14_h",
    "15_h",
    "14_dr",
    "15_dr",
    "14_g",
    "15_g",
]
TRAIN_IMAGES: list[str] = [i for i in ALL_IMAGES if i not in TEST_IMAGES]

PATCH_SIZE: int = 5  # 5 pix obok
RANDOM_SEED: int = 42

DEFAULT_SAMPLES_PER_IMAGE: int = 5000
