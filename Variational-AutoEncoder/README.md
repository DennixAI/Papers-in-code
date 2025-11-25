# Convolutional Variational Autoencoder (VAE) for CelebA

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) trained on the CelebA dataset. The project includes dataset acquisition via Hugging Face, custom preprocessing, and a full training pipeline with visualization.

---

## Architecture

The model is a **Convolutional VAE** designed for **128×128 RGB images**.

### Encoder

- **Structure:** A deep stack of `Conv2d` layers.  
- **Downsampling:** Achieved via a stride of 2 in convolution layers (no pooling), allowing the network to learn spatial downsampling.  
- **Compression:** Input `(3, 128, 128)` is flattened into a vector of size **16,384** after the final convolution block.  
- **Probabilistic Projection:** The vector is split into two dense layers representing **μ (mean)** and **log(σ²) (log variance)** of the latent distribution.

### Decoder

- **Upsampling:** Uses `ConvTranspose2d` (deconvolution) layers with specific `output_padding` to match the input size `(128×128)`.  
- **Expansion:** A linear layer projects the latent vector back to 16,384 dimensions, reshaped to `(256, 8, 8)` to start convolutional reconstruction.  
- **Output Activation:** `Sigmoid` is applied to bound pixel values between `[0,1]`.

---

## Key Techniques & Tricks

1. **Reparameterization Trick**  
   Allows backpropagation through a random node:  z = μ + σ * ε, ε ~ N(0,1)

2. **Loss Function**  
Sum of two components:  
- **Reconstruction Loss (MSE):** Penalizes pixel-wise error over the entire image.  
- **KL Divergence:** Regularizes the latent space to approximate a standard normal distribution N(0,1).

3. **Activation & Normalization**  
- `LeakyReLU` to prevent dying neurons.  
- **Batch Normalization** stabilizes layer inputs for faster convergence.  
- **Dropout** prevents overfitting in encoder and decoder.

4. **Robust Dataset Handling**  
- **Hugging Face Streaming:** Downloads CelebA from `nielsr/CelebA-faces`.  
- **Fallbacks:** Corrupt images are replaced with black images to avoid training crashes.

---

## Usage

### Install Dependencies
pip install torch torchvision datasets tqdm matplotlib python-dotenv torchinfo

### Config
Modify config.py to adjust hyperparameters (e.g., batch_size, latent_dim, epochs).

### Train
python3 train.py

Visualizations: Reconstructed images and loss curves saved in data/plots/ after each epoch.