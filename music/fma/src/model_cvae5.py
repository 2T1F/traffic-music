# model_cvae5.py

import torch
import torch.nn as nn

class ConditionalVAE5(nn.Module):
    """
    A small Conditional VAE that takes:
      - x:    (B, 1, n_mels=40, T=~312)   – normalized mel in [0,1]
      - cond: (B, 5)                      – five continuous features
    Encodes to z ∈ R^latent_dim, then decodes back to a mel.
    """

    def __init__(self, latent_dim: int, n_mels: int, n_feats: int, T: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mels     = n_mels
        self.n_feats    = n_feats
        self.T          = T

        # 1) Conv‐stack to downsample the mel
        self.encoder_conv = nn.Sequential(
            # (B,1,40,312) → (B, 8, 20, 156)
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # → (B,16,10,78)
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # → (B,32,5,39)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 2) Infer conv output size on a dummy, then make fc layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, T)
            h     = self.encoder_conv(dummy)
            conv_output_dim = h.numel()  # = 32 * 5 * 39 = 6240

        # 3) FC → produce μ, logvar each (latent_dim)
        self.fc_mu     = nn.Linear(conv_output_dim + n_feats, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_dim + n_feats, latent_dim)

        # 4) Decoder‐FC: (z+cond) → (n_mels * T) → reshape→(1,40,312)
        self.decoder_fc = nn.Linear(latent_dim + n_feats, n_mels * T)

    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x:    (B, 1, 40, 312)
        cond: (B, 5)
        """
        B = x.size(0)
        h      = self.encoder_conv(x)            # → (B,32,5,39)
        h_flat = h.view(B, -1)                   # → (B, 32 * 5 * 39)
        h_cat  = torch.cat([h_flat, cond], dim=1)  # → (B, conv_output_dim + 5)
        mu     = self.fc_mu(h_cat)               # → (B, latent_dim)
        logvar = self.fc_logvar(h_cat)           # → (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        z = μ + ϵ * σ, ϵ∼N(0,I).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor):
        """
        z:    (B, latent_dim)
        cond: (B,5)
        """
        B = z.size(0)
        h_cat   = torch.cat([z, cond], dim=1)   # → (B, latent_dim + 5)
        mel_flat = self.decoder_fc(h_cat)       # → (B, 40 * 312)
        mel      = mel_flat.view(B, 1, self.n_mels, self.T)  # → (B,1,40,312)
        mel_hat  = torch.sigmoid(mel)           # squash to [0,1]
        return mel_hat

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(x, cond)
        z          = self.reparameterize(mu, logvar)
        mel_hat    = self.decode(z, cond)
        return mel_hat, mu, logvar
