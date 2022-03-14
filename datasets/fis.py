from cmath import pi
import torch
import matplotlib.pyplot as plt


def f1(x, a):
    return torch.sin(2 * x * a)


def f2(x, a):
    return a * torch.abs(x - pi)


def f3(x, a1, a2, a3):
    return a1 * (torch.abs(x - pi) <= a3) + a2 * (torch.abs(x - pi) > a3)


def f4(x, a1, a2, a3):
    xmpi = torch.abs(x - pi)
    return (a1 * xmpi - a1 * a3) * (xmpi <= a3) + (a2 * xmpi - a2 * a3) * (xmpi > a3)


def f5(x, a1, a2, a3):
    xmpi = torch.abs(x - pi)
    return (0.5 * a1 * xmpi ** 2 - a1 * a3) * (xmpi <= a3) + (a2 * xmpi ** 2 - a2 * a3 - 0.5 * a3 ** 2 * (a1 - a2)) * (xmpi > a3)

if __name__ == "__main__":
    x=torch.arange(0, 2*pi, 0.1)
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(511)
    plt.plot(x, f1(x, 1))
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.title('f1(x)')
    plt.subplot(512)
    plt.plot(x, f2(x, 1))
    plt.xlabel('x')
    plt.ylabel('f2(x)')
    plt.title('f2(x)')
    plt.subplot(513)
    plt.plot(x, f3(x, 1, 2, 0.5))
    plt.xlabel('x')
    plt.ylabel('f3(x)')
    plt.title('f3(x)')
    plt.subplot(514)
    plt.plot(x, f4(x, 1, 2, 0.5))
    plt.xlabel('x')
    plt.ylabel('f4(x)')
    plt.title('f4(x)')
    plt.subplot(515)
    plt.plot(x, f5(x, 1, 2, 0.5))
    plt.xlabel('x')
    plt.ylabel('f5(x)')
    plt.title('f5(x)')
    plt.show()
    plt.savefig('results/functionsf1-5.png')
    plt.close()