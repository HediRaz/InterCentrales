"""Discriminator utility."""

from torch import nn


class LatentCodesDiscriminator(nn.Module):
    """Latent codes discriminator."""

    def __init__(self, style_dim, n_mlp):
        """Initialize the module."""
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for _ in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent):
        """Forward pass."""
        return self.mlp(latent)
