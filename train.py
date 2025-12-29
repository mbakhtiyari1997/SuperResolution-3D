import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# data
from data.tiff_loader import TiffSRDataset
from data.patch_sampler import PatchSampler3D

# models
from models.edsr3d import EDSR3D
from models.mdsr3d import MDSR3D

# loss & metrics
from losses.total_loss import SRLoss
from validate import validate


# --------------------------------------------------
# build model
# --------------------------------------------------
def build_model(cfg):
    if cfg["model"] == "edsr3d":
        return EDSR3D(scale=cfg["scale"])
    elif cfg["model"] == "mdsr3d":
        return MDSR3D(
            scale=cfg["scale"],
            use_deblur=cfg.get("use_deblur", False)
        )
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")


# --------------------------------------------------
# one epoch training
# --------------------------------------------------
def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    running_loss = 0.0

    for step, (lr, hr) in enumerate(loader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            sr = model(lr)
            loss = loss_fn(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


# --------------------------------------------------
# main training entry
# --------------------------------------------------
def main():

    # ---------- load config ----------
    with open("configs/mdsr3d_deblur_x4.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- dataset ----------
    sampler = PatchSampler3D(
        lr_size=cfg["patch_size"],
        scale=cfg["scale"]
    )

    train_set = TiffSRDataset(
        lr_list=cfg["train_lr"],
        hr_list=cfg["train_hr"],
        sampler=sampler
    )

    val_set = TiffSRDataset(
        lr_list=cfg["val_lr"],
        hr_list=cfg["val_hr"],
        sampler=sampler
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # ---------- model ----------
    model = build_model(cfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"]
    )

    loss_fn = SRLoss()
    scaler = GradScaler()

    # ---------- training loop ----------
    best_psnr = 0.0

    for epoch in range(cfg["epochs"]):

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} | "
            f"SSIM: {val_metrics['ssim']:.4f}"
        )

        # ---------- save last ----------
        torch.save(
            model.state_dict(),
            os.path.join(cfg["ckpt_dir"], "last.pth")
        )

        # ---------- save best ----------
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg["ckpt_dir"], "best.pth")
            )

    print("[INFO] Training finished.")


# --------------------------------------------------
# script entry
# --------------------------------------------------
if __name__ == "__main__":
    main()
