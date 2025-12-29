import torch.nn as nn
from .blocks import ResidualBlock3D

class EDSR3D(nn.Module):
    def __init__(self, scale=4, nf=64, nb=16):
        super().__init__()
        self.head = nn.Conv3d(1, nf, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock3D(nf) for _ in range(nb)],
            nn.Conv3d(nf, nf, 3, padding=1)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False),
            nn.Conv3d(nf, nf, 3, padding=1)
        )
        self.tail = nn.Conv3d(nf, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        r = self.body(x)
        x = x + r
        x = self.up(x)
        return self.tail(x)
