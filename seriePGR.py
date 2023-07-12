import sys
import pandas as pd
import numpy as np
import igraph as ig
from graphGen.graphGen import *
from graphGen.classifier import PGR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def diff(a: ig.Graph, b: ig.Graph):
    dif = 0
    for i, _ in enumerate(a.vs):
        dif += abs(a.vs[i].neighbors() - b.vs[i].neighbors())
    return dif


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
                dst += diff(i, j)

        dst /= len(X) ** 2
        for i in range(10, 60, 10):
            it.append(dst * i / 100)

    else:
        it = range(1, n + 1)

    for i in it:

        g = graph(X, i, pon=1, target=True, y=y, dif=diff)
        classifier = PGR(dst=diff)

        classifier.fit(g, X, y)
        lis = []
        for j in X_t:
            lis.append(classifier.predict(j))
        print(lis)
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
    else:
        X = np.array(df[list(range(1867))])
    y = y.reshape([53, 3, 1])

    stratify = []

    for i in y:
        stratify.append(i[0])

    mp = {}
    X_g = []

    for j, i in enumerate(X):
        print("gerando grafo:", j)
        X_g.append(graphFromSeries(i, 5))

    X_g = np.array(X_g)
    X_g = X_g.reshape([53, 3])

    for gr in [sKnnGraph, mKnnGraph, eNGraph]:
        f1 = 0.
        acc = 0.
        prec = 0.
        rcall = 0.

        for i in range(10):
            print(i)
            x_train, x_test, y_train, y_test = train_test_split(X_g, y, stratify=stratify)
            x_train = x_train.reshape([3 * 39])
            x_test = x_test.reshape([3 * 14])
            y_train = y_train.reshape([3 * 39])
            y_test = y_test.reshape([3 * 14, 1])

            f, a, p, r = fun_learn(30, x_train, y_train, x_test, y_test,)
            f1 += f
            acc += a
            prec += p
            rcall += r

        f1 /= 10
        acc /= 10
        prec /= 10
        rcall /= 10
        mp[gr] = (acc, prec, rcall, f1)
    print(mp)


if __name__ == '__main__':
    main()


