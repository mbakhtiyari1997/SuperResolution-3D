import torch

def gradient_loss(sr, hr):
    def grad(x):
        dz = x[:,:,1:,:,:] - x[:,:,:-1,:,:]
        dy = x[:,:,:,1:,:] - x[:,:,:,:-1,:]
        dx = x[:,:,:,:,1:] - x[:,:,:,:,:-1]
        return dz.abs().mean() + dy.abs().mean() + dx.abs().mean()
    return grad(sr) - grad(hr)
