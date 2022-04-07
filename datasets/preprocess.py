import torch


def preprocessing(X, u):
    l = u[:, 0] + (X - X[:, 0]) * (u[:, 6] - u[:, 0]) / (X[:, 0] - X[:, 6])
    usharp = u - l
    M = torch.max(usharp, dim=1)[0]
    m = torch.min(usharp, dim=1)[0]
    ustar = (2 * usharp - M - m) / (M - m)
    return ustar 
