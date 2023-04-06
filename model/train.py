import os
from typing import Callable, Sized

import torch
import torch.nn as nn
import torch.optim as toptim
import torch.utils.data as tdata
from torch import Tensor
from torchvision import datasets as vdata
from torchvision import transforms as vxform

from model.pixelcnn import PixelCNN


# Define your loss function
def criterion(x: Tensor, y: Tensor) -> Tensor:
    return nn.functional.cross_entropy(x, y)


# Define your training loop
def train(
    model: PixelCNN,
    train_loader: tdata.DataLoader,
    optimizer: toptim.Optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device | str,
) -> float:
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss


# Define your validation loop
def validate(
    model: PixelCNN,
    val_loader: tdata.DataLoader[Tensor],
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device | str,
):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    assert isinstance(val_loader.dataset, Sized)
    val_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / total
    return val_loss, accuracy


def discretize(x: Tensor, discretes: int = 255) -> Tensor:
    return (x * discretes).long()


def get_loaders() -> tuple[tdata.DataLoader, tdata.DataLoader]:
    transform = vxform.Compose(
        [
            vxform.Resize(28),
            vxform.CenterCrop(28),
            vxform.ToTensor(),
            vxform.Normalize(0.5, 0.5),
            discretize,
        ]
    )
    train_ds = vdata.MNIST("../data", train=True, transform=transform, download=True)
    val_dl = vdata.MNIST("../data", train=False, transform=transform, download=True)

    num_workers = (os.cpu_count() or 0) // 2
    persistent_workers = num_workers > 0
    train_dl = tdata.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    val_dl = tdata.DataLoader(
        train_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    return train_dl, val_dl


# Define your main function
def main(
    start_epoch=0, num_epochs=100, best_val_loss=float("inf"), resume_checkpoint=None
):
    # Define your data loaders
    train_loader, val_loader = get_loaders()
    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate your model
    model = PixelCNN(1, 64, 4)

    # Define your optimizer
    optimizer = toptim.Adam(model.parameters(), lr=1e-3)

    # Define your lr-scheduler
    scheduler = toptim.lr_scheduler.StepLR(optimizer, 1, 0.99)

    # Define your checkpoint paths
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")

    # Load from checkpoint if given
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    # Train and validate your model
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, accuracy = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        # Checkpoint the latest weights
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, latest_checkpoint_path)

        # Check if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_checkpoint_path)
    print("Training completed!")
