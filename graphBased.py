import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from graphGen.graphGen import *
from math import dist

class PGR:

    def __init__(self):
        self.fitted = False
        self.Ea = {}
        self.Y = 2.5
        self.fitted = None
        self.I = None
        self.X = None
        self.y = None

    def fit(self, g: ig.Graph, X, y):
        E = np.zeros(len(X))
        self.I = g.pagerank()
        q_y = {}
        self.X = X
        self.y = y
        for i, neighbors in enumerate(g.get_adjlist()):
            for j in neighbors:
                E[i] += g[i, j]
            E[i] /= len(neighbors)
            if y[i] in self.Ea:
                self.Ea[y[i]] += E[i]
            else:
                self.Ea[y[i]] = E[i]
            if y[i] in q_y:
                q_y[y[i]] += 1
            else:
                q_y[y[i]] = 1
        for i in self.Ea:
            self.Ea[i] /= q_y[i]
        self.fitted = True

    def predict(self, y):
        if not self.fitted:
            raise Exception("Para classificar uma nova amostra é necessario treinar o modelo")
        I_y = {}
        A = {}
        target = None
        target_value = 0

        for i, x_i in enumerate(self.X):
            f = self.Ea[self.y[i]] * self.Y - dist(x_i, y)
            if f >= 0:
                if self.y[i] in A:
                    A[self.y[i]].append(i)
                else:
                    A[self.y[i]] = [i]

        for i in A:
            for j in A[i]:
                if self.y[j] in I_y:
                    I_y[self.y[j]] += self.I[j]
                else:
                    I_y[self.y[j]] = self.I[j]

        for i, j in zip(I_y.keys(), I_y.values()):
            if j > target_value:
                target_value = j
                target = i

        return target


def main():
    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    X = np.array(df[list(range(1867))])
    X = X.reshape([53, 3, 1867])
    y = y.reshape([53, 3, 1])
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    x_train = x_train.reshape([3 * 39, 1867])
    x_test = x_test.reshape([3 * 14, 1867])
    y_train = y_train.reshape([3 * 39])
    y_test = y_test.reshape([3 * 14])

    g = sKnnGraph(x_train, 5, pon=1, target=1, y=y_train)
    pgr = PGR()
    pgr.fit(g, x_train, y_train)

    for i in x_test:
        print(pgr.predict(i))


if __name__ == '__main__':
    main()
