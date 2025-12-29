import torch

def psnr3d(sr, hr, max_val=1.0):
    """
    Computes PSNR for 3D volumes
    sr, hr: [B, 1, D, H, W]
    """
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))
