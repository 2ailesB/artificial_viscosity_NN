import torch
import matplotlib.pyplot as plt

def preprocessing(u, x):
    'u are the dataset X and x are the absisses x'
    a = torch.sub(x.T, x[:, 0]).T
    b = (u[:, 6] - u[:, 0]) / (x[:, 6] - x[:, 0])
    l = torch.add(torch.mul(a.T, b), u[:, 0]).T
    usharp = u - l # (:, 7), :0 and :6 ~ 0
    plt.figure()
    plt.plot(x[1, :], u[1, :], 'x')
    plt.plot(x[1, :], usharp[1, :], 'x')
    M = torch.max(usharp, dim=1)[0]
    m = torch.min(usharp, dim=1)[0]
    num = torch.sub(2 * usharp.T, M + m).T
    denom = (M - m)
    ustar = torch.div(num.T, denom).T #(:, 7)
    plt.plot(x[1, :], ustar[1, :], 'x')
    plt.legend(['u', 'usharp', 'ustar'])
    plt.show()
    return ustar 

if __name__ == "__main__":
    X = torch.randn((128, 7))
    print((X[:, 2] - X[:, 0]) == torch.sub(X.T, X[:, 0])[2, :])
    print(torch.sub(X.T, X[:, 0]).T.shape)
    print((X - X[:, 0]).shape)