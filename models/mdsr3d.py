import torch.nn as nn
from .blocks import ResidualBlock3D
from .deblur3d import DeblurBlock3D

class MDSR3D(nn.Module):
    """
    Multi-scale + Deblur-aware 3D Super-Resolution
    """
    def __init__(self, scale=4, nf=64, nb=12, use_deblur=True):
        super().__init__()
        self.use_deblur = use_deblur

        self.head = nn.Conv3d(1, nf, 3, padding=1)

        self.shared = nn.Sequential(
            *[ResidualBlock3D(nf) for _ in range(nb)]
        )

        if use_deblur:
            self.deblur = DeblurBlock3D(nf)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False),
            nn.Conv3d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv3d(nf, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.shared(x)
        if self.use_deblur:
            x = self.deblur(x)
        x = self.up(x)
        return self.tail(x)
