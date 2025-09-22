import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.hybrid_hrnet import HybridHRNet
from dataset.hand_dataset import HandPoseDataset
from utils import mpjpe

# Test
from torch.utils.data import Subset


def setup_logging(log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_{ts}.log")

    # Configurar logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Limpiar handlers previos (por si se llama train() más de una vez)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("%(message)s")
    ch.setFormatter(ch_fmt)

    # Archivo
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fh_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # Encabezado del log
    logging.info("epoch, train_loss, train_mpjpe, val_loss, val_mpjpe")
    return log_path


def save_checkpoint(model, optimizer, epoch, ckpt_dir="checkpoints", is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    # Checkpoint por época
    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    # Mejor modelo (sobrescribe)
    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        logging.info(f"Saved new best model -> {best_path}")
    return ckpt_path


def train():
    # ===== Config =====
    num_epochs = 50
    batch_size = 8
    lr = 1e-4
    num_joints = 21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    log_file = setup_logging("logs")
    logging.info(f"Logging to: {log_file}")

    # ===== Transforms =====
    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )

    # ===== Datasets / Loaders =====
    train_dataset = HandPoseDataset(
        "data/splits/train.json", use_images=True, transform=image_transform
    )
    val_dataset = HandPoseDataset(
        "data/splits/val.json", use_images=True, transform=image_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # ===== Model / Loss / Optim =====
    model = HybridHRNet(num_joints=num_joints).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # -------- Train --------
        model.train()
        train_loss_sum, train_mpjpe_sum = 0.0, 0.0

        for batch in train_loader:
            kp2d = batch["kp2d"].clone().detach().float().to(device)
            kp3d = batch["kp3d"].clone().detach().float().to(device)

            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            optimizer.zero_grad()

            if images is not None:
                preds = model(images, kp2d)
            else:
                preds = model(kp2d)

            loss = criterion(preds, kp3d)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_mpjpe_sum += mpjpe(preds, kp3d).item()

        # -------- Val --------
        model.eval()
        val_loss_sum, val_mpjpe_sum = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                kp2d = batch["kp2d"].clone().detach().float().to(device)
                kp3d = batch["kp3d"].clone().detach().float().to(device)

                images = batch.get("image", None)
                if images is not None:
                    images = images.to(device)

                if images is not None:
                    preds = model(images, kp2d)
                else:
                    preds = model(kp2d)

                loss = criterion(preds, kp3d)
                val_loss_sum += loss.item()
                val_mpjpe_sum += mpjpe(preds, kp3d).item()

        # -------- Metrics --------
        train_loss = train_loss_sum / max(1, len(train_loader))
        train_mpjpe = train_mpjpe_sum / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))
        val_mpjpe = val_mpjpe_sum / max(1, len(val_loader))

        # Consola + .log (CSV-like)
        logging.info(
            f"{epoch},{train_loss:.6f},{train_mpjpe:.6f},{val_loss:.6f},{val_mpjpe:.6f}"
        )
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train MPJPE: {train_mpjpe:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MPJPE: {val_mpjpe:.4f}"
        )

        # -------- Checkpoints --------
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        ckpt_path = save_checkpoint(model, optimizer, epoch, "checkpoints", is_best=is_best)
        # También puedes imprimir la ruta del checkpoint por época
        logging.info(f"Saved epoch checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    train()

# python -m scripts.train2
