from pyexpat import features
from sklearn.model_selection import train_test_split
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from datasets.inout import csv2np
from utils.metrics import accuracy

dataset = csv2np('data/custom_data.csv')

dataset_train, dataset_test= train_test_split(dataset, test_size=0.2, random_state=0)
features_train, features_test = torch.Tensor(dataset_train[:, 0:7]), torch.Tensor(dataset_test[:, 0:7])
labels_train, labels_test = torch.Tensor(dataset_train[:, 7]), torch.Tensor(dataset_test[:, 7])

# data
train_dataloader = DataLoader(TensorDataset(features_train, labels_train), batch_size=128)
test_dataloader = DataLoader(TensorDataset(features_test, labels_test), batch_size=128)

# model
K = 4
h1, h2, h3 = 16, 16, 16
indim = 7
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
path = 'results/runs/'
verbose = 100
optim = torch.optim.Adam(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(path)

#learing loop
for epoch in range(epochs):
    epoch_l = []
    epoch_acc = []
    epoch_ltest = []
    epoch_accTest = []
    for X, y in train_dataloader: # X is (128, 7), y is 128
        y = y.to(torch.long)
        yhat = model(X) # yhat is (128, 4)
        optim.zero_grad()
        l = loss(yhat, y)
        l.backward()
        optim.step()
        epoch_l.append(l)
        epoch_acc.append(accuracy(yhat, y.reshape(y.shape[0], 1)))

    writer.add_scalar('training loss', torch.Tensor(epoch_l).mean(), epoch)
    writer.add_scalar('training accuracy', torch.Tensor(epoch_acc).mean(), epoch)

    if epoch % verbose == 0:
        print(f'training loss at epoch {epoch} is {torch.Tensor(epoch_l).mean()}')
        print(f'training accuracy at epoch {epoch} is {torch.Tensor(epoch_acc).mean()}') # TODO:compute accuracy : see utils

    for X, y in test_dataloader:
        y = y.to(torch.long)
        yhat = model(X)
        ltest = loss(yhat, y)
        epoch_ltest.append(ltest)
        epoch_accTest.append(accuracy(yhat, y.reshape(y.shape[0], 1)))

    writer.add_scalar('testing loss', torch.Tensor(epoch_l).mean(), epoch)
    writer.add_scalar('testing accuracy', torch.Tensor(epoch_acc).mean(), epoch)

    if epoch % verbose == 0:
        print(f'testing loss at epoch {epoch} is {torch.Tensor(epoch_ltest).mean()}')
        print(f'testing accuracy at epoch {epoch} is {torch.Tensor(epoch_accTest).mean()}')
