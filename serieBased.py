import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from graphGen.graphGen import graphFromSeries
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.view(-1, len(x))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x.squeeze())  # Add .squeeze() to remove extra dimensions

        return x


def train(model, train_loader, optimizer):
    model.train()

    for data in train_loader:
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, loader):
    model.eval()

    predict_list = []

    for data in loader:
        out = model(data.x.float(), data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        predict_list.append(pred)

    return predict_list


def fun_learn(y_t, lis):
    f = f1_score(y_t, lis)
    a = accuracy_score(y_t, lis)
    p = precision_score(y_t, lis)
    r = recall_score(y_t, lis)
    return f, a, p, r


def igraph_to_data(x_train, y_train):
    x_temp = []
    for j, i in enumerate(x_train):
        x_ = torch.Tensor(i.vs['mag']).unsqueeze(1)
        edge_index = torch.tensor(i.get_edgelist()).t().contiguous()

        y_ = torch.LongTensor(y_train[j])

        x_temp.append(Data(x=x_, edge_index=edge_index, y=y_))

    return x_temp


def main():
    num_features = None
    hidden_dim = 1867
    num_classes = 2

    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    if len(sys.argv) > 1 and sys.argv[1] == '-o':
        num_features = 900
        X = np.array(df[list(range(900, 1800))])
        for i, _ in enumerate(X):
            for j, _ in enumerate(X[i]):
                dv = X[i][1630 - 900:1660 - 900].max()
                X[i, j] = X[i, j] / dv
    else:
        num_features = 1867
        X = np.array(df[list(range(1867))])
    y = y.reshape([53, 3, 1])

    model = GCN(num_features, hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_g = []

    for j, i in enumerate(X):
        print("gerando grafo:", j)
        X_g.append(graphFromSeries(i, 5))

    X_g = np.array(X_g)
    X_g = X_g.reshape([53, 3])

    f1 = 0.
    acc = 0.
    prec = 0.
    rcall = 0.

    for i in range(10):
        print(i)
        x_train, x_test, y_train, y_test = train_test_split(X_g, y)
        x_train = x_train.reshape([3 * 39])
        x_test = x_test.reshape([3 * 14])
        y_train = y_train.reshape([3 * 39, 1])
        y_test = y_test.reshape([3 * 14, 1])

        x_train = igraph_to_data(x_train, y_train)
        x_test = igraph_to_data(x_test, y_test)

        for epoch in range(1, 171):
            print('epoch', epoch)
            train(model, x_train, optimizer)

        l = test(model, x_test)

        f, a, p, r = fun_learn(y_test, l)
        f1 += f
        acc += a
        prec += p
        rcall += r

    f1 /= 10
    acc /= 10
    prec /= 10
    rcall /= 10

    print(f1)
    print(acc)
    print(prec)
    print(rcall)


if __name__ == '__main__':
    main()
