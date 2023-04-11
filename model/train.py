import os
from typing import Callable, Optional, Sized

import torch
import torch.amp as amp 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as toptim
import torch.utils.data as tdata
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets as vdata
from torchvision import transforms as vxform
from torchvision import utils as vutils
from tqdm import tqdm

from model.pixelcnn import PixelCNN

DATADIR = "../data"
LOGDIR = "runs"

BATCH_SIZE = 64
LOG_EVERY = 50
NUM_EPOCHS = 200


# Define your training loop
def train(
    epoch: int,
    model: PixelCNN,
    train_loader: tdata.DataLoader[Tensor],
    optimizer: toptim.Optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    device: torch.device | str,
    writer: SummaryWriter,
) -> float:
    model.train()
    log_loss = 0
    log_step = 0
    total_loss = 0
    total_step = 0
    data: Tensor
    target: Tensor

    prog = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
    for batch_idx, (data, target) in enumerate(prog):
        with torch.autocast(str(device)): # type: ignore
            data, target = data.to(device), target.to(device)
            target = model.get_class_aux(data.shape[-2:], target)
            optimizer.zero_grad()
            loss = model.calc_likelihood(data, target)
            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, Tensor)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

        # logging
        loss = loss.detach().item()
        log_loss += loss
        log_step += 1
        total_loss += loss
        total_step += 1
        if batch_idx % LOG_EVERY == LOG_EVERY - 1 or batch_idx == len(train_loader) - 1:
            writer.add_scalar(
                "loss/train",
                log_loss / log_step,
                batch_idx + epoch * (len(train_loader) - 1),
            )
            log_loss = 0
            log_step = 0
    return total_loss / total_step


# Define your validation loop
@torch.no_grad()
def validate(
    epoch: int,
    model: PixelCNN,
    val_loader: tdata.DataLoader[Tensor],
    device: torch.device | str,
    writer: SummaryWriter,
) -> float:
    model.eval()
    data: Tensor
    target: Tensor

    total_loss = 0
    total_step = 0

    prog = tqdm(val_loader, desc=f"Validate {epoch}", leave=False)
    for batch_idx, (data, target) in enumerate(prog):
        with torch.autocast(str(device)): # type: ignore
            data, target = data.to(device), target.to(device)
            target = model.get_class_aux(data.shape[-2:], target)
            total_loss += model.calc_likelihood(data, target).item()
            total_step += 1

    assert isinstance(val_loader.dataset, Sized)
    avg_loss = total_loss / total_step
    writer.add_scalar("loss/val", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def sample(
    epoch: int,
    model: PixelCNN,
    val_loader: tdata.DataLoader[Tensor],
    device: torch.device | str,
    writer: SummaryWriter,
) -> None:
    with torch.autocast(str(device)): # type: ignore
        aux = model.get_class_aux((28, 28), torch.arange(model._aux_channels))
        imgd = (
            model.sample(shape=(aux.shape[0], model.in_channels, 28, 28), aux=aux, depth_first=True)
            / model.num_classes
        )
        gridd = vutils.make_grid(imgd, aux.shape[0] // 2, 0)
        writer.add_image("sample/d", gridd, epoch)
        imgb = (
            model.sample(shape=(aux.shape[0], model.in_channels, 28, 28), aux=aux, depth_first=False)
            / model.num_classes
        )
        gridb = vutils.make_grid(imgb, aux.shape[0] // 2, 0)
        writer.add_image("sample/b", gridb, epoch)


def discretize(x: Tensor, discretes: int = 256) -> Tensor:
    return (x * (discretes - 1)).long()


def get_loaders() -> tuple[tdata.DataLoader, tdata.DataLoader]:
    transform = vxform.Compose(
        [
            vxform.Resize(28),
            vxform.CenterCrop(28),
            vxform.ToTensor(),
            discretize,
        ]
    )
    train_ds = vdata.MNIST("../data", train=True, transform=transform, download=True)
    val_dl = vdata.MNIST("../data", train=False, transform=transform, download=True)

    num_workers = (os.cpu_count() or 0) // 2
    persistent_workers = num_workers > 0
    train_dl = tdata.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    val_dl = tdata.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    return train_dl, val_dl


# Define your main function
def main(
    name: str,
    start_epoch: int = 0,
    num_epochs: int = NUM_EPOCHS,
    best_val_loss: float = float("inf"),
    resume_checkpoint: Optional[str] = None,
):
    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your data loaders
    train_loader, val_loader = get_loaders()

    # Instantiate your model
    model = PixelCNN(in_channels=1, hidden_channels=64, aux_channels=10).to(device)

    # Define your optimizer
    optimizer = toptim.Adam(model.parameters(), lr=1e-3)

    # Define your grad scaler
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    # Define your lr-scheduler
    scheduler = toptim.lr_scheduler.StepLR(optimizer, 1, 0.99)

    out_path = os.path.join(LOGDIR, name)

    # Define your summary writer
    writer = SummaryWriter(out_path)

    # Define your checkpoint paths
    ckptdir = os.path.join(out_path, "ckpts")
    os.makedirs(ckptdir, exist_ok=True)
    latest_checkpoint_path = os.path.join(ckptdir, "latest.pt")
    best_checkpoint_path = os.path.join(ckptdir, "best.pt")

    # Load from checkpoint if given
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

    # Train and validate your model
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(epoch, model, train_loader, optimizer, scaler, device, writer)
        val_loss = validate(epoch, model, val_loader, device, writer)
        sample(epoch, model, val_loader, device, writer)

        scheduler.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
        )

        # Checkpoint the latest weights
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, latest_checkpoint_path)

        # Check if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_checkpoint_path)
    writer.close()
    print("Training completed!")
