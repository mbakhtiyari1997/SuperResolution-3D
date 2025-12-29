import numpy as np

class PatchSampler3D:
    def __init__(self, lr_size=64, scale=4):
        self.lr = lr_size
        self.hr = lr_size * scale
        self.scale = scale

    def __call__(self, lr, hr):
        z, y, x = lr.shape
        z0 = np.random.randint(0, z - self.lr)
        y0 = np.random.randint(0, y - self.lr)
        x0 = np.random.randint(0, x - self.lr)

        lr_p = lr[z0:z0+self.lr, y0:y0+self.lr, x0:x0+self.lr]
        hr_p = hr[
            z0*self.scale:(z0+self.lr)*self.scale,
            y0*self.scale:(y0+self.lr)*self.scale,
            x0*self.scale:(x0+self.lr)*self.scale
        ]
        return lr_p, hr_p
