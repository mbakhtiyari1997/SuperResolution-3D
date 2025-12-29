import tifffile as tiff
import torch
import numpy as np

class TiffSRDataset(torch.utils.data.Dataset):
    def __init__(self, lr_list, hr_list, sampler=None):
        self.lr_list = lr_list
        self.hr_list = hr_list
        self.sampler = sampler

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, idx):
        lr = tiff.imread(self.lr_list[idx]).astype(np.float32)
        hr = tiff.imread(self.hr_list[idx]).astype(np.float32)

        lr = (lr - lr.min()) / (lr.max() - lr.min() + 1e-8)
        hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-8)

        if self.sampler:
            lr, hr = self.sampler(lr, hr)

        return (
            torch.from_numpy(lr)[None],
            torch.from_numpy(hr)[None]
        )
