import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from graphGen.graphGen import *
from math import dist


class PGR:

    def __init__(self, Y=2.39, dst=dist):
        self.fitted = False
        self.Ea = {}
        self.Y = Y
        self.fitted = None
        self.I = None
        self.X = None
        self.y = None
        self.dist = dst

    def fit(self, g: ig.Graph, X, y):
        E = np.zeros(len(X))
        self.I = g.pagerank()
        q_y = {}
        self.X = X
        self.y = y
        for i, neighbors in enumerate(g.get_adjlist()):
            for j in neighbors:
                E[i] += g[i, j]
            if len(neighbors):
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
        target = 0
        target_value = 0

        for i, x_i in enumerate(self.X):
            f = self.Ea[self.y[i]] * self.Y - self.dist(x_i, y)
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


def distC(a: np.ndarray, b: np.ndarray):
    return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def fun_learn(n: int, X, y, X_t, y_t, graph=sKnnGraph):
    f1 = list()
    acc = list()
    prec = list()
    rcall = list()
    f1_max = 0
    acc_max = 0
    prec_max = 0
    rcall_max = 0

    it = []

    if graph is eNGraph:
        dst = 0.
        for i in X:
            for j in X:
                if '-c' in sys.argv:
                    dst += distC(i, j)
                else:
                    dst += dist(i, j)
        dst /= len(X)**2
        for i in range(10, 60, 10):
            it.append(dst*i/100)

    else:
        it = range(1, n + 1)

    for i in it:
        if '-c' in sys.argv:
            g = graph(X, i, pon=1, target=True, y=y, dif=distC)
            classifier = PGR(dst=distC)
        else:
            g = graph(X, i, pon=1, target=True, y=y)
            classifier = PGR()
        classifier.fit(g, X, y)
        lis = []
        for j in X_t:
            lis.append(classifier.predict(j))
        lis = np.array(lis)
        f1_current = f1_score(y_t, lis)
        acc_current = accuracy_score(y_t, lis)
        prec_current = precision_score(y_t, lis)
        rcall_current = recall_score(y_t, lis)
        if f1_current > f1_max:
            f1_max = f1_current
            acc_max = acc_current
            prec_max = prec_current
            rcall_max = rcall_current
        f1.append(f1_current)
        acc.append(acc_current)
        prec.append(prec_current)
        rcall.append(rcall_current)
    df = {'acuracia': acc, 'f1': f1, 'precision': prec, 'recall': rcall}
    df = pd.DataFrame.from_dict(df)
    df.to_csv('resultado/' + graph.__name__ + 'graphPGR.csv')
    return f1_max, acc_max, prec_max, rcall_max


def main():
    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    if '-o' in sys.argv:
        X = np.array(df[list(range(900, 1800))])
        for i, _ in enumerate(X):
            for j, _ in enumerate(X[i]):
                dv = X[i][1630 - 900:1660 - 900].max()
                X[i, j] = X[i, j] / dv
        X = X.reshape([53, 3, 900])
    else:
        X = np.array(df[list(range(1867))])
        X = X.reshape([53, 3, 1867])
    y = y.reshape([53, 3, 1])
    mp = {}

    stratify = []

    for i in y:
        stratify.append(i[0])

    for gr in [sKnnGraph, mKnnGraph, eNGraph]:
        f1 = 0.
        acc = 0.
        prec = 0.
        rcall = 0.
        for i in range(10):
            print(i)
            x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=stratify)
            if '-o' in sys.argv:
                x_train = x_train.reshape([3 * 39, 900])
                x_test = x_test.reshape([3 * 14, 900])
                y_train = y_train.reshape([3 * 39])
                y_test = y_test.reshape([3 * 14, 1])
            else:
                x_train = x_train.reshape([3 * 39, 1867])
                x_test = x_test.reshape([3 * 14, 1867])
                y_train = y_train.reshape([3 * 39])
                y_test = y_test.reshape([3 * 14, 1])

            f, a, p, r = fun_learn(50, x_train, y_train, x_test, y_test, gr)
            f1 += f
            acc += a
            prec += p
            rcall += r

        f1 /= 10.
        acc /= 10.
        prec /= 10.
        rcall /= 10.
        mp[gr] = (acc, prec, rcall, f1)
    print(mp)


if __name__ == '__main__':
    main()
