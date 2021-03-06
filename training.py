from sklearn.model_selection import train_test_split
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import datetime
import numpy as np

from datasets.inout import csv2np
from datasets.preprocess import preprocessing
from utils.metrics import accuracy
from network.net import smoothness_classifier

dataset = csv2np('data/custom_data.csv')
abscisses = csv2np('data/custom_xs.csv')

dataset_train, dataset_test = train_test_split(np.concatenate(
    (dataset, abscisses), axis=1), test_size=0.2, random_state=0)
features_train, features_test = torch.Tensor(
    dataset_train[:, 0:7]), torch.Tensor(dataset_test[:, 0:7])
labels_train, labels_test = torch.Tensor(
    dataset_train[:, 7]), torch.Tensor(dataset_test[:, 7])
abscisses_train, abscisses_test = torch.Tensor(
    dataset_train[:, 8:]), torch.Tensor(dataset_test[:, 8:])

print(f'Training on {features_train.shape[0]} samples') # 338400
print(f'Testing on {features_test.shape[0]} samples') # 84600

# data
train_dataloader = DataLoader(TensorDataset(torch.cat(
    (features_train, abscisses_train), dim=1), labels_train), batch_size=128)
test_dataloader = DataLoader(TensorDataset(torch.cat(
    (features_test, abscisses_test), dim=1), labels_test), batch_size=128)

# model
K = 4
h1, h2, h3 = 16, 16, 16
indim = 7
outdim = 4
alpha = 1
lbd = 0.1
# model = nn.Sequential(nn.Linear(indim, h1), nn.ELU(alpha),
#         nn.Dropout(lbd),
#         nn.Linear(h1, h2), nn.ELU(alpha),
#         nn.Linear(h2, h3), nn.ELU(alpha),
#         nn.Linear(h3, outdim), nn.Softmax())

model = smoothness_classifier(indim, outdim, (h1, h2, h3), alpha, lbd)

# training procedure
lr = 0.001
epochs = 1000
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path = 'results/runs/run_' + start_time
verbose = 100
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(path)

#learing loop
for epoch in range(epochs):
    epoch_l = []
    epoch_acc = []
    epoch_ltest = []
    epoch_accTest = []
    for X, y in train_dataloader:  # X is (128, 7), y is 128
        X, abscisses = X[:, :7], X[:, 7:]
        y = y.to(torch.long)
        X, y = preprocessing(X, abscisses, y)
        # print(X.isnan().sum(), abscisses.isnan().sum(), y.isnan().sum())
        # print(X[X.isnan().sum(dim=1), :], y[X.isnan().sum(dim=1)])
        yhat = model(X)  # yhat is (128, 4)
        optim.zero_grad()
        l = loss(yhat, y)
        l.backward()
        optim.step()
        epoch_l.append(l)
        y_pred = model.predict(X)
        epoch_acc.append(accuracy(y_pred.reshape(
            y.shape[0], 1), y.reshape(y.shape[0], 1)))

    writer.add_scalar('training loss', torch.Tensor(epoch_l).mean(), epoch)
    writer.add_scalar('training accuracy',
                      torch.Tensor(epoch_acc).mean(), epoch)

    if epoch % verbose == 0:
        print(
            f'training loss at epoch {epoch} is {torch.Tensor(epoch_l).mean()}')
        # TODO:compute accuracy : see utils
        print(
            f'training accuracy at epoch {epoch} is {torch.Tensor(epoch_acc).mean()}')

    for X, y in test_dataloader:
        X, abscisses = X[:, :7], X[:, 7:]
        y = y.to(torch.long)
        X, y = preprocessing(X, abscisses, y)
        yhat = model(X)
        ltest = loss(yhat, y)
        epoch_ltest.append(ltest)
        y_pred = model.predict(X)
        epoch_accTest.append(accuracy(y_pred.reshape(
            y.shape[0], 1), y.reshape(y.shape[0], 1)))

    writer.add_scalar('testing loss', torch.Tensor(epoch_l).mean(), epoch)
    writer.add_scalar('testing accuracy',
                      torch.Tensor(epoch_acc).mean(), epoch)

    if epoch % verbose == 0:
        print(
            f'testing loss at epoch {epoch} is {torch.Tensor(epoch_ltest).mean()}')
        print(
            f'testing accuracy at epoch {epoch} is {torch.Tensor(epoch_accTest).mean()}')
