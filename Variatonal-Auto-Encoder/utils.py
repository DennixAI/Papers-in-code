import glob
import os
import shutil
import tarfile
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from datasets import load_dataset
from config import CONFIG

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs(cfg: dict) -> None:
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)
    os.makedirs(cfg["archives_dir"], exist_ok=True)
    os.makedirs(cfg["image_root"], exist_ok=True)
    os.makedirs(cfg["hf_local_dir"], exist_ok=True)

def dataset_available(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    for _, _, files in os.walk(folder):
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            return True
    return False

def extract_archive(archive_path: str, destination: str) -> None:
    print(f"Extracting {archive_path} -> {destination}")
    os.makedirs(destination, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def extract_all_archives(root_dir: str) -> None:
    patterns = ["*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2"]
    archives = []
    for pattern in patterns:
        archives.extend(glob.glob(os.path.join(root_dir, pattern)))
    if not archives:
        return
    for archive_path in archives:
        extract_archive(archive_path, root_dir)

def relocate_celeba_images(source_root: str, target_root: str) -> None:
    if dataset_available(target_root):
        return

    os.makedirs(target_root, exist_ok=True)
    moved = 0
    valid_ext = {".jpg", ".jpeg", ".png"}
    for dirpath, _, filenames in os.walk(source_root):
        try:
            if os.path.samefile(dirpath, target_root):
                continue
        except FileNotFoundError:
            pass

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_ext:
                continue

            src = os.path.join(dirpath, filename)
            dst = os.path.join(target_root, filename)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
            moved += 1

    print(f"Moved {moved} image files into {target_root}")

def download_via_huggingface(cfg: dict) -> None:
    image_root = cfg["image_root"]
    os.makedirs(image_root, exist_ok=True)

    print("Downloading CelebA from Hugging Face: nielsr/CelebA-faces")
    ds = load_dataset("nielsr/CelebA-faces", split="train")

    from tqdm import tqdm
    for i, example in enumerate(tqdm(ds, desc="Saving CelebA images")):
        img = example["image"]
        out_path = os.path.join(image_root, f"{i:06d}.jpg")
        img.save(out_path)

    print(f"Saved {len(ds)} images to {image_root}")

def prepare_dataset(cfg: dict) -> None:
    if dataset_available(cfg["image_root"]):
        print(f"Using existing dataset at {cfg['image_root']}")
        return

    hf_dir = cfg["hf_local_dir"]
    if dataset_available(hf_dir):
        print(f"Found existing Hugging Face snapshot at {hf_dir}")
        relocate_celeba_images(hf_dir, cfg["image_root"])
        if dataset_available(cfg["image_root"]):
            print(f"Dataset prepared at {cfg['image_root']}")
            return

    archives_dir = cfg["archives_dir"]
    extract_all_archives(archives_dir)
    relocate_celeba_images(archives_dir, cfg["image_root"])
    if dataset_available(cfg["image_root"]):
        print(f"Dataset prepared at {cfg['image_root']}")
        return

    download_via_huggingface(cfg)
    if not dataset_available(cfg["image_root"]):
        raise RuntimeError(
            "Dataset download finished, but no images were found. "
            "Place archives in the archives directory or verify the Hugging Face download."
        )
    print(f"Dataset prepared at {cfg['image_root']}")

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        patterns = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(root_dir, pattern)))
        self.image_paths = image_paths or glob.glob(os.path.join(root_dir, "*"))

        print(f"Found {len(self.image_paths)} images in {root_dir}")
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
            fallback = Image.new("RGB", (CONFIG["image_size"], CONFIG["image_size"]), color="black")
            return self.transform(fallback) if self.transform else fallback

def build_transforms(cfg: dict):
    return transforms.Compose(
        [
            transforms.CenterCrop((cfg["image_size"], cfg["image_size"])),
            transforms.ToTensor(),
        ]
    )

def create_dataloaders(cfg: dict):
    transform = build_transforms(cfg)
    dataset = ImageFolderDataset(cfg["image_root"], transform=transform)

    train_len = int(cfg["train_split"] * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    loader_args = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    print(f"DataLoaders -> Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader

def save_reconstructions(model, loader, device, epoch, cfg, max_items: int = 8):
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_batch = sample_batch.to(device, non_blocking=True)
        recon_batch, _, _, _ = model(sample_batch)

    max_items = min(max_items, sample_batch.size(0))
    stacked = torch.cat([sample_batch[:max_items], recon_batch[:max_items]], dim=0)
    grid = make_grid(stacked, nrow=max_items, normalize=True)
    npimg = grid.detach().cpu().permute(1, 2, 0).numpy()
    output_path = os.path.join(cfg["plot_dir"], f"epoch_{epoch:03d}_recon.png")
    plt.imsave(output_path, np.clip(npimg, 0, 1))
    print(f"Saved reconstructions to {output_path}")

def plot_loss_curves(history, cfg):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_total"], label="Train Total")
    plt.plot(history["val_total"], label="Val Total")
    plt.plot(history["train_recon"], label="Train Recon", linestyle="--")
    plt.plot(history["val_recon"], label="Val Recon", linestyle="--")
    plt.plot(history["train_kl"], label="Train KL", linestyle=":")
    plt.plot(history["val_kl"], label="Val KL", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(cfg["plot_dir"], "loss_curves.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curves saved to {loss_plot_path}")