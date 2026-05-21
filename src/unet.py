"""U-Net (segmentation_models_pytorch) - etap 3, ocena 5."""
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
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics.classification import BinaryRecall, BinarySpecificity
from monai.inferers import sliding_window_inference


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
    # Stałe ImageNet - wymagane przez encoder pretrained na ImageNet.
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
        # Pełny pipeline: aug -> normalize -> HWC->CHW tensor. Wszystko z albumentations.
        self.transform = A.Compose([
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=self.IMAGENET_MEAN.tolist(),
                        std=self.IMAGENET_STD.tolist(),
                        max_pixel_value=255.0),
            ToTensorV2(),
        ], additional_targets={'fov': 'mask'}, seed=seed)

    def __len__(self) -> int:
        return len(self.image_ids) * self.crops_per_image

    def _load(self, image_id: str):
        # Cache + prekomputacja dozwolonych środków kropu (inaczej np.where co getitem).
        if image_id not in self._cache:
            rgb, manual, fov = self.loader(image_id)
            half = self.crop_size // 2
            valid = fov > 0
            valid[:half, :] = valid[-half:, :] = False
            valid[:, :half] = valid[:, -half:] = False
            vy, vx = np.where(valid & (manual > 0))
            ay, ax = np.where(valid)
            self._cache[image_id] = (rgb, manual, fov, vy, vx, ay, ax)
        return self._cache[image_id]

    def __getitem__(self, idx: int):
        rgb, manual, fov, vy, vx, ay, ax = self._load(self.image_ids[idx % len(self.image_ids)])
        half = self.crop_size // 2
        use_vessel = self.rng.random() < self.vessel_prob and len(vy) > 0
        ys, xs = (vy, vx) if use_vessel else (ay, ax)
        k = int(self.rng.integers(0, len(ys)))
        y, x = ys[k] - half, xs[k] - half
        cs = self.crop_size
        out = self.transform(
            image=rgb[y:y+cs, x:x+cs],
            mask=manual[y:y+cs, x:x+cs],
            fov=fov[y:y+cs, x:x+cs],
        )
        return out['image'], out['mask'].float().unsqueeze(0), out['fov'].float().unsqueeze(0)


# ---- Loss ------------------------------------------------------------------
_smp_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)


def combined_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """BCE + Dice (z smp), oba maskowane przez FOV."""
    # weight=fov daje per-pixel maskowanie BCE bez ręcznej redukcji.
    bce = F.binary_cross_entropy_with_logits(
        logits, target, weight=fov, reduction="sum"
    ) / fov.sum().clamp(min=1.0)
    dice = _smp_dice(logits * fov, (target * fov).long())
    return bce + dice


@torch.no_grad()
def gmean_torch(probs: torch.Tensor, target: torch.Tensor, fov: torch.Tensor, thr: float = 0.5) -> float:
    """G-mean = sqrt(sens·spec) w FOV - z torchmetrics."""
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


# ---- Trening (PyTorch Lightning) -------------------------------------------
@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_gmean: list[float]


class _LitUNet(pl.LightningModule):
    """Cienka obwoluta Lightning: cała logika treningu/walidacji w step-ach."""
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y, fov = batch
        loss = combined_loss(self(x), y, fov)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, fov = batch
        logits = self(x)
        loss = combined_loss(logits, y, fov)
        gmean = gmean_torch(torch.sigmoid(logits), y, fov)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_gmean", gmean, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


class _HistoryAndBestCheckpoint(Callback):
    """Zbiera per-epoch historię + zapisuje state_dict modelu (nie Lightning) po best G-mean."""
    def __init__(self, save_best_to: Path | str | None = None):
        super().__init__()
        self.save_best_to = save_best_to
        self.history = TrainHistory(train_loss=[], val_loss=[], val_gmean=[])
        self.best_gmean = -1.0

    def on_validation_epoch_end(self, trainer, lit: _LitUNet):
        m = trainer.callback_metrics
        if "train_loss" not in m:  # sanity-check val przed pierwszą epoką treningu
            return
        tl = float(m["train_loss"])
        vl = float(m["val_loss"])
        vg = float(m["val_gmean"])
        self.history.train_loss.append(tl)
        self.history.val_loss.append(vl)
        self.history.val_gmean.append(vg)
        print(f"epoch {trainer.current_epoch + 1:3d}: train_loss={tl:.4f}  val_loss={vl:.4f}  val_gmean={vg:.4f}")
        if vg > self.best_gmean and self.save_best_to is not None:
            self.best_gmean = vg
            Path(self.save_best_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save(lit.model.state_dict(), self.save_best_to)


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
    """Adam + ReduceLROnPlateau. Zapisuje best-by-G-mean checkpoint. Backed by Lightning."""
    device = device or get_device()
    accelerator = {"mps": "mps", "cuda": "gpu", "cpu": "cpu"}[device.type]
    cb = _HistoryAndBestCheckpoint(save_best_to=save_best_to)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=progress,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(_LitUNet(model, lr=lr), train_loader, val_loader)
    return cb.history


# ---- Predykcja na pełnym obrazie -------------------------------------------
_predict_normalize = A.Normalize(
    mean=RetinaPatchDataset.IMAGENET_MEAN.tolist(),
    std=RetinaPatchDataset.IMAGENET_STD.tolist(),
    max_pixel_value=255.0,
)


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
    """Tiling z overlapem przez MONAI sliding_window_inference."""
    device = device or get_device()
    model.to(device)
    model.eval()
    x = _predict_normalize(image=rgb)["image"]  # HWC float32, znormalizowany
    x_t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(device)
    logits = sliding_window_inference(
        inputs=x_t,
        roi_size=(tile_size, tile_size),
        sw_batch_size=batch_size,
        predictor=model,
        overlap=overlap / tile_size,
        mode="constant",
        padding_mode="reflect",
    )
    pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > threshold).astype(np.uint8)
    return pred & (fov > 0).astype(np.uint8)


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
    device = device or get_device()
    model = UNet(in_channels=in_channels, out_channels=out_channels,
                 encoder=encoder, encoder_weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model
