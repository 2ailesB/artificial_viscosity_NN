from turtle import forward
from matplotlib.pyplot import axis
import torch
import torch.functional as F
import torch.nn as nn

class smoothness_classifier(nn.Module):
    def __init__(self, indim, outdim, h, alpha, lbd):
        self.indim = indim
        self.outdim = outdim
        self.h = h
        self.alpha = alpha
        self.lbd = lbd
        super().__init__() # TODO : clean network construction
        self.net = nn.Sequential(nn.Linear(indim, h[0]), nn.ELU(alpha), 
                                nn.Dropout(lbd),
                                nn.Linear(h[0], h[1]), nn.ELU(alpha), 
                                nn.Linear(h[1], h[2]), nn.ELU(alpha), 
                                nn.Linear(h[2], outdim) , nn.Softmax(dim=1))

    def forward(self, x):
        print(x, self.net(x))
        return self.net(x)

    def predict(self, x):
        return torch.argmax(self.net(x), dim=1) # x is (128, 7) et self.net(x) is (128, 4)
