import torch.nn.functional as F

def ssim3d(x, y):
    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)
    sigma_x = F.avg_pool3d(x*x, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool3d(y*y, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool3d(x*y, 3, 1, 1) - mu_x*mu_y
    C1, C2 = 0.01**2, 0.03**2
    return (((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) /
            ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))).mean()
