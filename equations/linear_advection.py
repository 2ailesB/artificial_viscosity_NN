from cmath import pi
import torch
import torch.nn as nn

def smooth_init(x):
    return torch.exp(torch.sin(2 * pi * (x - 0.25)))

def mix_init(x):
    pass