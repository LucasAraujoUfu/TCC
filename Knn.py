import pandas as pd
import numpy as np
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

    print(func_learn(30, x_train, y_train, x_test, y_test))
    print(func_learn(30, x_train, y_train, x_test, y_test,classifier=RandomForestClassifier))


if __name__ == '__main__':
    main()