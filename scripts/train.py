import os
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


def train():
    # Config
    num_epochs = 1
    batch_size = 1
    lr = 1e-4
    num_joints = 21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image transforms (convert numpy image to torch.Tensor and normalize if needed)
    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )

    # Dataset with images loaded
    train_dataset = HandPoseDataset(
        "data/splits/train.json", use_images=True, transform=image_transform
    )
    val_dataset = HandPoseDataset(
        "data/splits/val.json", use_images=True, transform=image_transform
    )

    train_subset = Subset(train_dataset, list(range(20)))
    val_subset = Subset(val_dataset, list(range(20)))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Model initialization
    model = HybridHRNet(num_joints=num_joints).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")  # Initialize best validation loss

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mpjpe = 0, 0

        for batch in train_loader:
            kp2d = batch["kp2d"].clone().detach().float().to(device)
            kp3d = batch["kp3d"].clone().detach().float().to(device)

            images = batch.get("image", None)
            if images is not None:
                images = images.to(device)

            # Forward pass (adjust call depending on model signature)
            if images is not None:
                preds = model(images, kp2d)
            else:
                preds = model(kp2d)

            loss = criterion(preds, kp3d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mpjpe += mpjpe(preds, kp3d).item()

        # Validation
        model.eval()
        val_loss, val_mpjpe = 0, 0
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
                val_loss += loss.item()
                val_mpjpe += mpjpe(preds, kp3d).item()

        # Compute averages
        train_loss /= len(train_loader)
        train_mpjpe /= len(train_loader)
        val_loss /= len(val_loader)
        val_mpjpe /= len(val_loader)

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train MPJPE: {train_mpjpe:.4f} "
            f"Val Loss: {val_loss:.4f}, Val MPJPE: {val_mpjpe:.4f}"
        )

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "C:/Users/Chari/OneDrive/Escritorio/DATA/Monocular3d/3d-pose-lifting/checkpoints/best_model.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("Saved new best model")


if __name__ == "__main__":
    train()
