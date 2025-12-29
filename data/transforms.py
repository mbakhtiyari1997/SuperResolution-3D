import numpy as np

def random_flip(x):
    if np.random.rand() > 0.5:
        x = np.flip(x, axis=0)
    if np.random.rand() > 0.5:
        x = np.flip(x, axis=1)
    if np.random.rand() > 0.5:
        x = np.flip(x, axis=2)
    return x.copy()
