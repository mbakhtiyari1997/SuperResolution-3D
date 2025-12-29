import torch
from metrics.psnr import psnr3d
from metrics.ssim import ssim3d_metric

def validate(model, loader, loss_fn, device):
    model.eval()
    loss_total, psnr_total, ssim_total = 0, 0, 0

    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            loss_total += loss_fn(sr, hr).item()
            psnr_total += psnr3d(sr, hr).item()
            ssim_total += ssim3d_metric(sr, hr).item()

    n = len(loader)
    return {
        "loss": loss_total / n,
        "psnr": psnr_total / n,
        "ssim": ssim_total / n
    }
