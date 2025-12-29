from .charbonnier import CharbonnierLoss
from .ssim3d import ssim3d
from .gradient_loss import gradient_loss

class SRLoss:
    def __init__(self):
        self.l1 = CharbonnierLoss()

    def __call__(self, sr, hr):
        return (
            self.l1(sr, hr)
            + 0.1 * (1 - ssim3d(sr, hr))
            + 0.01 * gradient_loss(sr, hr)
        )
