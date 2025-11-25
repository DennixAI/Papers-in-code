import os
import torch
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

CONFIG = {
    "image_root": os.path.join("data", "celeba", "images"),
    "archives_dir": os.path.join("data", "celeba", "archives"),
    "hf_local_dir": os.path.join("data", "celeba", "hf_snapshot"),
    "hf_repo_ids": [
        os.getenv("HF_REPO_ID") or "ashraq/celeba",
        "celeba",
    ],
    "hf_repo_type": "dataset",
    "hf_token_env": "HG_TOKEN",
    "image_size": 128,
    "batch_size": 8,
    "num_workers": 2,
    "latent_dim": 64,
    "hidden_dim": 128,
    "epochs": 240,
    "learning_rate": 0.0005,
    "train_split": 0.8,
    "checkpoint_dir": "checkpoints",
    "plot_dir": os.path.join("data", "plots"),
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
