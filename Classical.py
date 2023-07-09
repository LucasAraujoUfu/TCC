import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def func_learn(n: int, X, y, X_t, y_t, classifier=KNeighborsClassifier):
    classifier = classifier()
    f1 = list()
    acc = list()
    prec = list()
    rcall = list()
    f1_max = 0
    acc_max = 0
    prec_max = 0
    rcall_max = 0
    for i in range(1, n + 1):
        if type(classifier) is KNeighborsClassifier:
            classifier.n_neighbors = i
        elif type(classifier) is RandomForestClassifier:
            classifier.n_estimators = i
        else:
            raise Exception("classificador não suportado")

        if '-c' in sys.argv:
            classifier.metric = 'cosine'

        classifier.fit(X, y)
        l = list()
        for j in X_t:
            l.append(classifier.predict([j]))
        l = np.array(l)
        f1_current = f1_score(y_t, l)
        acc_current = accuracy_score(y_t, l)
        prec_current = precision_score(y_t, l)
        rcall_current = recall_score(y_t, l)
        if f1_current > f1_max:
            f1_max = f1_current
            acc_max = acc_current
            prec_max = prec_current
            rcall_max = rcall_current
        f1.append(f1_current)
        acc.append(acc_current)
        prec.append(prec_current)
        rcall.append(rcall_current)
    df = {'acuracia': acc, 'f1': f1, 'recall': rcall, 'precision': prec}
    df = pd.DataFrame.from_dict(df)
    df.to_csv('resultado/' + str(classifier) + '.csv')
    return f1_max, acc_max, prec_max, rcall_max


def neural_network(X, y, X_t, y_t):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((X.shape[1], 1), input_shape=(X.shape[1],)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    model.fit(X, y, epochs=20,)
    pre = model.predict(X_t)
    l = []
    for i in pre:
        l.append(np.argmax(i))

    f1 = f1_score(y_t, l)
    acc = accuracy_score(y_t, l)
    prec = precision_score(y_t, l)
    rcall = recall_score(y_t, l)
    return f1, acc, prec, rcall


def main():
    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    if '-o' in sys.argv:
        X = np.array(df[list(range(900, 1800))])
        for i, _ in enumerate(X):
            for j, _ in enumerate(X[i]):
                dv = X[i][1630-900:1660-900].max()
                X[i, j] = X[i, j]/dv
        X = X.reshape([53, 3, 900])
    else:
        X = np.array(df[list(range(1867))])
        X = X.reshape([53, 3, 1867])
    y = y.reshape([53, 3, 1])

    stratify = []

    for i in y:
        stratify.append(i[0])

    f1 = [0., 0., 0.]
    acc = [0., 0., 0.]
    prec = [0., 0., 0.]
    rcall = [0., 0., 0.]

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

        # f, a, p, r = func_learn(50, x_train, y_train, x_test, y_test)
        # f1[0] += f
        # acc[0] += a
        # prec[0] += p
        # rcall[0] += r
        # f, a, p, r = func_learn(50, x_train, y_train, x_test, y_test, classifier=RandomForestClassifier)
        # f1[1] += f
        # acc[1] += a
        # prec[1] += p
        # rcall[1] += r
        f, a, p, r = neural_network(x_train, y_train, x_test, y_test)
        f1[2] += f
        acc[2] += a
        prec[2] += p
        rcall[2] += r

    for i in range(3):
        f1[i] /= 10
        acc[i] /= 10
        prec[i] /= 10
        rcall[i] /= 10
    print(f1)
    print(acc)
    print(prec)
    print(rcall)


if __name__ == '__main__':
    main()
