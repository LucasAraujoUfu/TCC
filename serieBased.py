import pandas as pd
import numpy as np
from graphGen.graphGen import graphFromSeries
from sklearn.model_selection import train_test_split


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

    print(graphFromSeries(x_train[0]))


if __name__ == '__main__':
    main()
