import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def func_learn(n:int,X,y,X_t,y_t,classifier = KNeighborsClassifier):
    classifier = classifier()
    f1 = list()
    f1_max = 0
    for i in range(1,n+1):
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
        f1_max = max(f1_max, f1_current)
        f1.append(f1_current)
    return (f1_max)


def neural_network(X,y,X_t,y_t):
    ipt = X.shape[1]
    l1 = (ipt+1)//2
    l2 = (l1+1)//2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1867,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X,y)
    test_loss, test_acc = model.evaluate(X_t,y_t)
    print(test_acc)




def main():
    df = pd.read_csv('data/controle_tea.dat',header=None,sep=' ')
    y = np.array(df[1867])
    X = np.array(df[list(range(1867))])
    X = X.reshape([53,3,1867])
    y = y.reshape([53,3,1])
    x_train,x_test,y_train,y_test = train_test_split(X,y)
    x_train = x_train.reshape([3*39, 1867])
    x_test = x_test.reshape([3*14, 1867])
    y_train = y_train.reshape([3*39])
    y_test = y_test.reshape([3*14,1])

    # print(func_learn(30, x_train, y_train, x_test, y_test))
    # print(func_learn(30, x_train, y_train, x_test, y_test,classifier=RandomForestClassifier))
    neural_network(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()