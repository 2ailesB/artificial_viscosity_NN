import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# data
train_dataloader = None
test_dataloader = None

# model
K = 4
h1, h2, h3 = 16, 16, 16
indim = 1
outdim = 4
alpha = 1
lbd = 0.1
model = nn.Sequential(nn.Linear(indim, h1), nn.ELU(alpha), 
        nn.Dropout(lbd),
        nn.Linear(h1, h2), nn.ELU(alpha), 
        nn.Linear(h2, h3), nn.ELU(alpha), 
        nn.Linear(h3, outdim), nn.Softmax())

# training procedure
lr = 0.001
epochs = 1000
path = 'results/'
verbose = 10
optim = torch.optim.Adam(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(path)

#learing loop
for epoch in range(epochs):
    epoch_l = []
    epoch_ltest = []
    for X, y in train_dataloaeder:
        yhat = model(X)
        optim.zero_grad()
        l = loss(y, yhat)
        l.backward()
        optim.step()
        epoch_l.append(l)

    writer.add_scalars('loss', epoch_l.mean(), epoch)

    if epoch % verbose:
        print(f'training loss at epoch {epoch} is {epoch_l.mean()}')
        print(f'training accuracy at epoch {epoch} is {epoch_l.mean()}') # TODO:compute accuracy : see utils

    for X, y in test_dataloaeder:
        yhat = model(X)
        ltest = loss(y, yhat)
        epoch_ltest.append(ltest)

    if epoch % verbose:
        print(f'testing loss at epoch {epoch} is {epoch_l.mean()}')
        print(f'testing accuracy at epoch {epoch} is {epoch_l.mean()}')
