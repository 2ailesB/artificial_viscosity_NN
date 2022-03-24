import numpy.fft as fft
import numpy as np

def compute_fourier(x):
    return fft.fft(x)

def select_stencil(x, n):
    print(n // 2, x.shape[1] - n // 2)
    ps = np.random.randint(n // 2, x.shape[1] - n // 2, size=x.shape[0])
    print(ps.shape)
    xs = np.zeros((x.shape[0], n))
    for i, p in enumerate(ps):
        idx = np.arange(p - n //2, p + n //2 + 1)
        xs[i, :] = x[i, idx]
    return xs