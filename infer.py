import torch, tifffile as tiff
from models.mdsr3d import MDSR3D

def infer(lr_path, ckpt, out_path):
    lr = tiff.imread(lr_path).astype("float32")
    lr = torch.from_numpy(lr)[None,None].cuda()

    model = MDSR3D(scale=4, use_deblur=True).cuda()
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    with torch.no_grad():
        sr = model(lr)[0,0].cpu().numpy()

    tiff.imwrite(out_path, sr)
