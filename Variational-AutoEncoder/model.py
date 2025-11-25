import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, negative_slope: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, negative_slope: float = 0.1):
        super().__init__()
        self.expand = nn.Linear(output_dim, 16384)
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.2),
        )

    def forward(self, z):
        z = self.expand(z)
        z = z.view(-1, 256, 8, 8)
        reconstructed = self.reconstruct(z)
        return torch.sigmoid(reconstructed)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim * 2, hidden_dim // 4, latent_dim)
        self.mu_head = nn.Linear(16384, latent_dim, bias=False)
        self.logvar_head = nn.Linear(16384, latent_dim, bias=False)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2.0)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_head(encoded)
        log_var = self.logvar_head(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var, z


def vae_loss(reconstructed, target, mu, log_var):
    recon = F.mse_loss(reconstructed, target, reduction="none")
    recon = recon.view(reconstructed.size(0), -1).sum(dim=1)
    recon = recon.mean()

    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl = kl.mean()

    total = kl + recon
    return total, recon, kl