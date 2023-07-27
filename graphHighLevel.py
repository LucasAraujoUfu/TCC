import pandas as pd
import numpy as np
import sys
import scipy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from graphClass.highlevel import HighLevel
from math import dist


def fun_learn(n: int, X, y, X_t, y_t, graph):
    classifier = HighLevel()
    classifier.graph_construction_method = graph
    f1 = list()
    acc = list()
    prec = list()
    rcall = list()
    f1_max = 0
    acc_max = 0
    prec_max = 0
    rcall_max = 0

    it = []

    if graph == 3:
        dst = 0.
        for i in X:
            for j in X:
                dst += dist(i, j)
        dst /= len(X) ** 2
        for i in range(10, 60, 10):
            it.append(dst * i / 100)

    else:
        it = range(1, n + 1)

    for i in it:
        if graph == 3:
            classifier.e = i
        else:
            classifier.k = i
        classifier.fit(X, y)
        lis = classifier.predict(X_t)
        print(lis)

        # TODO: means
        return 0, 0, 0, 0





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

    for gr in range(1, 4):
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
                y_train = y_train.reshape([3 * 39, 1])
                y_test = y_test.reshape([3 * 14, 1])
            else:
                x_train = x_train.reshape([3 * 39, 1867])
                x_test = x_test.reshape([3 * 14, 1867])
                y_train = y_train.reshape([3 * 39, 1])
                y_test = y_test.reshape([3 * 14, 1])

            f, a, p, r = fun_learn(50, x_train, y_train,
                                   x_test, y_test, gr)
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
