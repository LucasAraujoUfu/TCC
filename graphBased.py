import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from graphGen.graphGen import *
from math import dist

class PGR:

    def __init__(self):
        self.fitted = False
        self.Ea = {}
        self.Y = 2.39
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


def fun_learn(n: int, X, y, X_t, y_t):
    f1 = list()
    acc = list()
    f1_max = 0
    acc_max = 0
    for i in range(1, n + 1):
        g = sKnnGraph(X, n, pon=1, target=True, y=y)
        classifier = PGR()
        classifier.fit(g, X, y)
        l = list()
        for j in X_t:
            l.append(classifier.predict(j))
        l = np.array(l)
        f1_current = f1_score(y_t, l)
        acc_current = accuracy_score(y_t, l)
        f1_max = max(f1_max, f1_current)
        acc_max = max(acc_max, acc_current)
        f1.append(f1_current)
        acc.append(acc_current)
    df = {'acuracia': acc, 'f1': f1}
    df = pd.DataFrame.from_dict(df)
    df.to_csv('resultado/graphPGR.csv')
    return f1_max, acc_max


def main():
    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    X = np.array(df[list(range(1867))])
    X = X.reshape([53, 3, 1867])
    y = y.reshape([53, 3, 1])

    f1 = 0.
    acc = 0.
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        x_train = x_train.reshape([3 * 39, 1867])
        x_test = x_test.reshape([3 * 14, 1867])
        y_train = y_train.reshape([3 * 39])
        y_test = y_test.reshape([3 * 14])

        f, a = fun_learn(30, x_train, y_train, x_test, y_test)
        f1 += f
        acc += a

    f1 /= 10.
    acc /= 10.
    print(f1, acc)


if __name__ == '__main__':
    main()
