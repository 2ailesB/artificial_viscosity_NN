import numpy.fft as fft
import numpy as np
import torch

def compute_fourier(x):
    return fft.fft(x)

def compute_torch_fourier(x):
    x_np = x.numpy()
    return torch.Tensor(fft.fft(x_np)) #(:, 7)

def select_stencil(x, d, n):
    "x = points, d = domain, n = number to sample (7)"
    ps = np.random.randint(n // 2, x.shape[1] - n // 2, size=x.shape[0]) 
    xs = np.zeros((x.shape[0], n))
    ds = np.zeros((x.shape[0], n))
    for i, p in enumerate(ps):
        idx = np.arange(p - n //2, p + n //2 + 1)
        xs[i, :] = x[i, idx]
        ds[i, :] = d[idx]
    return xs, ds

def select_stencils(x, d, n, m):
    "x = points, d = domain, n = number to sample (7), m = dataset size per function"
    ps = np.random.randint(n // 2, x.shape[1] - n // 2, size=x.shape[0]*m) 
    xs = np.zeros((x.shape[0]*m, n))
    ds = np.zeros((x.shape[0]*m, n))
    for i, p in enumerate(ps):
        idx = np.arange(p - n //2, p + n //2 + 1)
        xs[i, :] = x[i%x.shape[0], idx]
        ds[i, :] = d[idx]
    return xs, ds