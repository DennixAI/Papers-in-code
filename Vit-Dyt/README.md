# DyT CIFAR-10 Notebook

This repository now focuses on a single Jupyter notebook, `dyt_cifar10.ipynb`, that trains the Dynamic Tanh Transformer (DyT) on CIFAR-10 with RandAugment, mixup, mixed precision, and local logging/plotting.

![DyT CIFAR-10 Training Curve](plot.png)

## Quick Start
1. Create an environment and install the minimal dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter (Notebook or Lab) inside this folder and open `dyt_cifar10.ipynb`.
3. Run the cells sequentially:
   - The setup cell seeds everything and prints your device info.
   - The config cell defines fixed hyperparameters tuned for an 8–10 GB GPU budget (batch size 256, DyT dim 384, depth 6, heads 6, mixup α=0.2, etc.).
   - Data cells download CIFAR-10 automatically, apply RandAugment + standard normalization, and build train/test dataloaders.
   - Training cells instantiate DyT (`models/dyt.py`), AdamW, cosine LR scheduling, and AMP; they also expose helper functions for mixup, metrics logging, checkpointing, and plotting.
   - The main training loop cell resumes from `checkpoint/dyt_cifar10_latest.pth` if present, otherwise starts from scratch and periodically saves both the latest and best checkpoints.
   - The final cell saves/opens `plots/dyt_cifar10_training.png` so you can monitor learning curves without external services.

## Logging, Checkpoints, and Plots
- Latest/best checkpoints live under `checkpoint/` (created automatically).
- A CSV plus human-readable log is written in `log/` for every epoch.
- Plots are stored under `plots/`; the repo ships with an empty `.gitkeep` so the directory persists even before you run training.
- Because everything happens inside the notebook, you can safely rerun it inside a managed notebook/Jupyter environment without CLI arguments.

## Utility Files
- `randomaug.py` provides the RandAugment implementation used in the data pipeline.
- `utils.py` supplies the CLI-style `progress_bar` helper for nicer iteration output.
- `models/dyt.py` contains the DyT definition (Transformer blocks with Dynamic Tanh normalization) imported directly by the notebook.

## References
- Vision Transformer (ViT): Alexey Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021. https://arxiv.org/abs/2010.11929
- Dynamic Tanh Transformer (DyT): Jiachen Zhu et al., *"Transformers without Normalization"*, 2025. https://arxiv.org/abs/2503.10622

Please cite those works (along with this notebook) if you build on the provided training recipe.
