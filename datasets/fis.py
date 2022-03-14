from cmath import pi
import torch


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
