import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt

from config import CONFIG, DEVICE
from model import VariationalAutoencoder, vae_loss
from utils import (
    set_seed,
    ensure_dirs,
    prepare_dataset,
    create_dataloaders,
    save_reconstructions,
    plot_loss_curves,
    save_checkpoint as utils_save_checkpoint
)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs, mu, log_var, _ = model(batch)
        loss, recon, kl = vae_loss(outputs, batch, mu, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

    num_batches = len(loader)
    return total_loss / num_batches, recon_loss / num_batches, kl_loss / num_batches

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)
        for batch in progress:
            batch = batch.to(device, non_blocking=True)
            outputs, mu, log_var, _ = model(batch)
            loss, recon, kl = vae_loss(outputs, batch, mu, log_var)
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

    num_batches = len(loader)
    return total_loss / num_batches, recon_loss / num_batches, kl_loss / num_batches

def save_checkpoint(model, optimizer, epoch, loss_value, cfg):
    filepath = os.path.join(cfg["checkpoint_dir"], f"vae_epoch_{epoch:03d}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_value,
        },
        filepath,
    )
    print(f"Checkpoint saved: {filepath}")

def main():
    ensure_dirs(CONFIG)
    set_seed(CONFIG["seed"])
    prepare_dataset(CONFIG)

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = create_dataloaders(CONFIG)

    model = VariationalAutoencoder(
        input_dim=3,
        hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"],
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print("Model Summary:")
    print(summary(model, (1, 3, CONFIG["image_size"], CONFIG["image_size"]), device=DEVICE))

    history = {
        "train_total": [],
        "val_total": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
    }

    training_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        epoch_start = time.time()
        train_tot, train_recon, train_kl = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_tot, val_recon, val_kl = evaluate(model, val_loader, DEVICE)
        epoch_duration = time.time() - epoch_start

        history["train_total"].append(train_tot)
        history["val_total"].append(val_tot)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)

        print(
            f"Train -> total: {train_tot:.4f}, recon: {train_recon:.4f}, kl: {train_kl:.4f} | "
            f"Val -> total: {val_tot:.4f}, recon: {val_recon:.4f}, kl: {val_kl:.4f}"
        )

        save_checkpoint(model, optimizer, epoch, train_tot, CONFIG)
        save_reconstructions(model, val_loader, DEVICE, epoch, CONFIG, max_items=8)
        print(f"Epoch {epoch} duration: {epoch_duration / 60:.2f} min ({epoch_duration:.1f} sec)")

    plot_loss_curves(history, CONFIG)
    total_duration = time.time() - training_start
    print(f"\nTraining complete in {total_duration / 3600:.2f} hours ({total_duration / 60:.1f} minutes)")

if __name__ == "__main__":
    main()