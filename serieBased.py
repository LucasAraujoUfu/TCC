import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from graphGen.graphGen import graphFromSeries
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class GNN(nn.Module):
    def __init__(self, input_size, hidden_channels, conv, conv_params={}):
        super(GNN, self).__init__()
        # torch.manual_seed(12345)

        self.conv1 = conv(
            input_size, hidden_channels, **conv_params
        )

        self.conv2 = conv(
            hidden_channels, hidden_channels, **conv_params
        )

        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch=None, edge_col=None):
        x = self.conv1(x, edge_index, edge_col)
        x = x.relu()

        x = self.conv2(x, edge_index, edge_col)

        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=.5, training=self.training)
        x = self.lin(x)

        return x


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

        x = []
        edge_list = []
        # edge_col = []
        batch = []
        for ct, j in enumerate(x_train):
            x.append(j.vs['mag'])
            edge_list += list(j.get_edgelist())
            #edge_col += list(j.es['weight'])
            batch += [ct]*len(j.get_edgelist())

        x = torch.tensor(x, dtype=torch.double).to(device)
        print(x)
        edge_list = torch.tensor(edge_list).to(device)
        edge_list = torch.transpose(edge_list,0,1)
        # edge_col = torch.tensor(edge_col, dtype=torch.double).to(device)
        # print(edge_list.size())
        batch = torch.tensor(batch).to(device)

        val_x = []
        val_edge_list = []
        val_batch = []
        for ct, j in enumerate(x_test):
            val_x += j.vs['mag']
            val_batch += [ct] * num_features
            val_edge_list.append(list(j.get_adjacency()))

        val_x = torch.tensor(x).to(device)
        val_x = val_x.double()
        val_edge_list = torch.tensor(edge_list).to(device)
        val_batch = torch.tensor(batch).to(device)

        model = GNN(num_features, 500, GCNConv).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(x, edge_list, batch)
            loss = torch.nn.CrossEntropyLoss()(output, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 3:
                model.eval()
                with torch.no_grad():
                    val_output = model(val_x, val_edge_list, val_batch)
                    val_loss = torch.nn.CrossEntropyLoss()(val_output, y_test)
                    print(val_output)
                    print(val_loss)
                model.train()


        # TODO: fun_learn

        f, a, p, r = [0, 0, 0, 0]  # fun_learn(y_test, l)
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
