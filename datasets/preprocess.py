from tarfile import USTAR_FORMAT
import torch
import matplotlib.pyplot as plt

def preprocessing(u, x, y=None, verbose=False):
    'u are the dataset X and x are the absisses x'
    # TODO : debug here ; M=m because spmetimes u is constant.
    a = torch.sub(x.T, x[:, 0]).T
    b = (u[:, 6] - u[:, 0]) / (x[:, 6] - x[:, 0])
    l = torch.add(torch.mul(a.T, b), u[:, 0]).T
    usharp = u - l # (:, 7), :0 and :6 ~ 0
    M = torch.max(usharp, dim=1)[0]
    m = torch.min(usharp, dim=1)[0]
    num = torch.sub(2 * usharp.T, M + m).T
    denom = (M - m) 
    ustar = torch.div(num.T, denom).T #(:, 7)
    # print(denom)
    # print(u.isnan().sum(), usharp.isnan().sum(), ustar.isnan().sum())
    # print(ustar.shape, y.shape)
    nan_idx = torch.any(ustar.isnan(), dim=1)
    ustar=ustar[~nan_idx, :]
    if y is not None:
        y = y[~nan_idx]
    # print(u.isnan().sum(), usharp.isnan().sum(), ustar.isnan().sum())
    # print(ustar.shape, y.shape)
    if verbose:
        plt.figure()
        plt.plot(x[1, :], u[1, :], '-x')
        plt.plot(x[1, :], l[1, :], '--')
        plt.plot(x[1, :], usharp[1, :], '-x')
        plt.plot(x[1, :], ustar[1, :], '-x')
        plt.legend(['u', 'l', 'usharp', 'ustar'])
        plt.savefig('results/preprocessing_example.png')
        plt.show()
        plt.close()
    return ustar, y

if __name__ == "__main__":
    X = torch.randn((128, 7))
    print((X[:, 2] - X[:, 0]) == torch.sub(X.T, X[:, 0])[2, :])
    print(torch.sub(X.T, X[:, 0]).T.shape)
    # print((X - X[:, 0]).shape)
    y = torch.arange(0, 7, 1).repeat((128, 1))
    print(y.shape, X.shape)
    preprocessing(X, y, True)