import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


def func_learn(n: int, X, y, X_t, y_t, classifier=KNeighborsClassifier):
    classifier = classifier()
    f1 = list()
    acc = list()
    f1_max = 0
    acc_max = 0
    for i in range(1, n + 1):
        if type(classifier) is KNeighborsClassifier:
            classifier.n_neighbors = i
        elif type(classifier) is RandomForestClassifier:
            classifier.n_estimators = i
        else:
            raise Exception("classificador não suportado")
        classifier.fit(X, y)
        l = list()
        for j in X_t:
            l.append(classifier.predict([j]))
        l = np.array(l)
        f1_current = f1_score(y_t, l)
        acc_current = accuracy_score(y_t, l)
        f1_max = max(f1_max, f1_current)
        acc_max = max(acc_max, acc_current)
        f1.append(f1_current)
        acc.append(acc_current)
    df = {'acuracia': acc, 'f1': f1}
    df = pd.DataFrame.from_dict(df)
    df.to_csv('resultado/' + str(classifier) + '.csv')
    return f1_max, acc_max


def neural_network(X, y, X_t, y_t):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((1867, 1), input_shape=(1867,)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    model.fit(X, y, epochs=15,)
    pre = model.predict(X_t)
    l = []
    for i in pre:
        l.append(np.argmax(i))

    f1 = f1_score(y_t, l)
    acc = accuracy_score(y_t, l)
    return f1, acc


def main():
    df = pd.read_csv('data/controle_tea.dat', header=None, sep=' ')
    y = np.array(df[1867])
    X = np.array(df[list(range(1867))])
    X = X.reshape([53, 3, 1867])
    y = y.reshape([53, 3, 1])

    f1 = [0., 0., 0.]
    acc = [0., 0., 0.]

    for i in range(10):
        print(i)
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        x_train = x_train.reshape([3 * 39, 1867])
        x_test = x_test.reshape([3 * 14, 1867])
        y_train = y_train.reshape([3 * 39])
        y_test = y_test.reshape([3 * 14, 1])

        f, a = func_learn(50, x_train, y_train, x_test, y_test)
        f1[0] += f
        acc[0] += a
        f, a = func_learn(50, x_train, y_train, x_test, y_test, classifier=RandomForestClassifier)
        f1[1] += f
        acc[1] += a
        f, a = neural_network(x_train, y_train, x_test, y_test)
        f1[2] += f
        acc[2] += a

    for i in range(3):
        f1[i] /= 10
        acc[i] /= 10
    print(f1)
    print(acc)


if __name__ == '__main__':
    main()
