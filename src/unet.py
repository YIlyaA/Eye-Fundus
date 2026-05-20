"""U-Net (segmentation_models_pytorch) — etap 3, ocena 5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from torchmetrics.classification import BinaryRecall, BinarySpecificity


# ---- Model -----------------------------------------------------------------
def UNet(
    in_channels: int = 3,
    out_channels: int = 1,
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """U-Net z biblioteki segmentation_models_pytorch (encoder pretrained na ImageNet)."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
    )


# ---- Dataset ---------------------------------------------------------------
class RetinaPatchDataset(Dataset):
    """Losowe kropy 256×256 z augmentacją (rot90, flip), sampling z biasem na naczynia."""
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        image_ids: list[str],
        loader,
        crop_size: int = 256,
        crops_per_image: int = 64,
        vessel_prob: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.image_ids = image_ids
        self.loader = loader
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.vessel_prob = vessel_prob
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, tuple] = {}
        self.augment = A.Compose([
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'fov': 'mask'}, seed=seed)

    def __len__(self) -> int:
        return len(self.image_ids) * self.crops_per_image

    def _load(self, image_id: str):
        # Cache + prekomputacja dozwolonych środków (inaczej np.where co getitem).
        if image_id not in self._cache:
            rgb, manual, fov = self.loader(image_id)
            half = self.crop_size // 2
            valid = fov > 0
            valid[:half, :] = False
            valid[-half:, :] = False
            valid[:, :half] = False
            valid[:, -half:] = False
            vy, vx = np.where(valid & (manual > 0))
            ay, ax = np.where(valid)
            self._cache[image_id] = (rgb, manual, fov, vy, vx, ay, ax)
        return self._cache[image_id]

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx % len(self.image_ids)]
        rgb, manual, fov, vy, vx, ay, ax = self._load(image_id)
        half = self.crop_size // 2

        if self.rng.random() < self.vessel_prob and len(vy) > 0:
            k = int(self.rng.integers(0, len(vy)))
            cy, cx = vy[k], vx[k]
        else:
            k = int(self.rng.integers(0, len(ay)))
            cy, cx = ay[k], ax[k]
        y, x = cy - half, cx - half
        cs = self.crop_size
        rgb_c = rgb[y:y+cs, x:x+cs]
        manual_c = manual[y:y+cs, x:x+cs]
        fov_c = fov[y:y+cs, x:x+cs]

        out = self.augment(image=rgb_c, mask=manual_c, fov=fov_c)
        rgb_c, manual_c, fov_c = out['image'], out['mask'], out['fov']

        x_tensor = (rgb_c.astype(np.float32) / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
        x_tensor = torch.from_numpy(np.ascontiguousarray(x_tensor.transpose(2, 0, 1)))
        y_tensor = torch.from_numpy(np.ascontiguousarray(manual_c)).float().unsqueeze(0)
        fov_tensor = torch.from_numpy(np.ascontiguousarray(fov_c)).float().unsqueeze(0)
        return x_tensor, y_tensor, fov_tensor


# ---- Loss ------------------------------------------------------------------
# DiceLoss z smp + BCE z PyTorch, oba maskowane przez FOV
_smp_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)


def combined_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """BCE + Dice (z smp), oba maskowane przez FOV."""
    bce_per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_per_pixel * fov).sum() / fov.sum().clamp(min=1.0)
    dice = _smp_dice(logits * fov, (target * fov).long())
    return bce + dice


@torch.no_grad()
def gmean_torch(probs: torch.Tensor, target: torch.Tensor, fov: torch.Tensor, thr: float = 0.5) -> float:
    """G-mean = sqrt(sens·spec) w FOV — z torchmetrics."""
    # Maskowanie przez FOV: zostawiamy tylko piksele wewnątrz pola widzenia
    mask = (fov > 0).flatten()
    pred = (probs > thr).long().flatten()[mask]
    gt = (target > 0).long().flatten()[mask]
    sens = BinaryRecall().to(probs.device)(pred, gt).item()
    spec = BinarySpecificity().to(probs.device)(pred, gt).item()
    return math.sqrt(max(0.0, sens * spec))


def get_device() -> torch.device:
    """MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---- Trening ---------------------------------------------------------------
@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_gmean: list[float]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_best_to: Path | str | None = None,
    progress: bool = True,
) -> TrainHistory:
    """Adam + ReduceLROnPlateau. Zapisuje best-by-G-mean checkpoint."""
    if device is None:
        device = get_device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    history = TrainHistory(train_loss=[], val_loss=[], val_gmean=[])
    best_gmean = -1.0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it   # noqa: E731

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        iterator = train_loader
        if progress:
            iterator = tqdm(iterator, desc=f"epoch {epoch}/{epochs} [train]", leave=False)
        for x, y, fov in iterator:
            x, y, fov = x.to(device), y.to(device), fov.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y, fov)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_gmeans = [], []
        with torch.no_grad():
            iterator = val_loader
            if progress:
                iterator = tqdm(iterator, desc=f"epoch {epoch}/{epochs} [val]", leave=False)
            for x, y, fov in iterator:
                x, y, fov = x.to(device), y.to(device), fov.to(device)
                logits = model(x)
                loss = combined_loss(logits, y, fov)
                val_losses.append(loss.item())
                val_gmeans.append(gmean_torch(torch.sigmoid(logits), y, fov))

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        vg = float(np.mean(val_gmeans))
        history.train_loss.append(tl)
        history.val_loss.append(vl)
        history.val_gmean.append(vg)
        scheduler.step(vl)
        print(f"epoch {epoch:3d}: train_loss={tl:.4f}  val_loss={vl:.4f}  val_gmean={vg:.4f}")

        if vg > best_gmean and save_best_to is not None:
            best_gmean = vg
            Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_best_to)

    return history


# ---- Predykcja na pełnym obrazie -------------------------------------------
@torch.no_grad()
def predict_mask(
    model: nn.Module,
    rgb: np.ndarray,
    fov: np.ndarray,
    device: torch.device | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """Tiling z overlapem, średnia po nakładkach, próg na sigmoidzie."""
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    h, w = rgb.shape[:2]
    x = (rgb.astype(np.float32) / 255.0 - RetinaPatchDataset.IMAGENET_MEAN) / RetinaPatchDataset.IMAGENET_STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    x_t = torch.from_numpy(x).to(device)

    step = tile_size - overlap
    pad_h = (math.ceil((h - overlap) / step) * step + overlap) - h
    pad_w = (math.ceil((w - overlap) / step) * step + overlap) - w
    x_padded = F.pad(x_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    ph, pw = x_padded.shape[1:]

    coords = [
        (ty, tx)
        for ty in range(0, ph - overlap, step)
        for tx in range(0, pw - overlap, step)
    ]

    prob_sum = torch.zeros((ph, pw), dtype=torch.float32, device=device)
    weight = torch.zeros((ph, pw), dtype=torch.float32, device=device)

    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        tiles = torch.stack(
            [x_padded[:, ty:ty+tile_size, tx:tx+tile_size] for ty, tx in chunk],
            dim=0,
        )
        logits = model(tiles)
        probs = torch.sigmoid(logits).squeeze(1)
        for (ty, tx), prob in zip(chunk, probs):
            prob_sum[ty:ty+tile_size, tx:tx+tile_size] += prob
            weight[ty:ty+tile_size, tx:tx+tile_size] += 1.0

    avg = (prob_sum / weight.clamp(min=1.0)).cpu().numpy()[:h, :w]
    return (avg > threshold).astype(np.uint8) & (fov > 0).astype(np.uint8)


# ---- Zapis / wczytanie -----------------------------------------------------
def save_model(model: nn.Module, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: Path | str,
    in_channels: int = 3,
    out_channels: int = 1,
    encoder: str = "resnet34",
    device: torch.device | None = None,
) -> nn.Module:
    """Tworzy U-Net, wczytuje wagi, .to(device) + eval()."""
    if device is None:
        device = get_device()
    model = UNet(in_channels=in_channels, out_channels=out_channels,
                 encoder=encoder, encoder_weights=None)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
