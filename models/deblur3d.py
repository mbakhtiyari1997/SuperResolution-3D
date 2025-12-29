import torch.nn as nn

class DeblurBlock3D(nn.Module):
    """
    Shallow deblurring branch for micro-CT degradation compensation
    """
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)
