import torch

from datasets.fourriercollocation import compute_fourier, compute_torch_fourier

def preprocessing(X):
    u = compute_torch_fourier(X) # X and u are (:, 7)
    a = torch.sub(X.T, X[:, 0]).T
    b = (u[:, 6] - u[:, 0]) / (X[:, 6] - X[:, 0])
    l = torch.add(torch.mul(a.T, b), u[:, 0]).T
    usharp = u - l # (:, 7), :0 and :6 ~ 0
    M = torch.max(usharp, dim=1)[0]
    m = torch.min(usharp, dim=1)[0]
    num = torch.sub(2 * usharp.T, M + m).T
    denom = (M - m)
    ustar = torch.div(num.T, denom).T #(:, 7)
    return ustar 

if __name__ == "__main__":
    X = torch.randn((128, 7))
    print((X[:, 2] - X[:, 0]) == torch.sub(X.T, X[:, 0])[2, :])
    print(torch.sub(X.T, X[:, 0]).T.shape)
    print((X - X[:, 0]).shape)