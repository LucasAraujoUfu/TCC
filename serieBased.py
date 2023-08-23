import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from graphGen.tensorGraph import tGraphFromSeries
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=.5, training=self.training)
        x = self.lin(x)

        return x


def train(model, x):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()
    for data in x:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.Y).sum())
    return correct/len(loader.dataset)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    X_g = []

    for j, i in enumerate(X):
        print("gerando grafo:", j)
        X_g.append(tGraphFromSeries(i, 5, [df[1867][j]]))

    model = GCN(1, 72813)

    f1 = 0.
    acc = 0.
    prec = 0.
    rcall = 0.

    for i in range(10):
        # TODO: Depois que funcionar arrumar a separação do datasets (triplicatas)
        shuffle(X_g)
        train_dataset = DataLoader(X_g[:117], batch_size=39, shuffle=True)
        test_dataset = DataLoader(X_g[117:], batch_size=39, shuffle=False)

        for epoch in range(1, 100):
            train(model, train_dataset)
            train_acc = test(model, train_dataset)
            test_acc = test(model, test_dataset)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()
