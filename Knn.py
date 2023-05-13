import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def knn(n:int,X,y,X_t,y_t,classifier = KNeighborsClassifier):
    classifier = classifier()
    f1 = list()
    f1_max = 0
    for i in range(1,n):
        if type(classifier) is KNeighborsClassifier:
            classifier.n_neighbors = i
        elif type(classifier) is RandomForestClassifier:
            classifier.n_estimators = i
        else:
            raise Exception("classificador não suportado")
        classifier.fit(X, y)
        l = list()
        for j in X_t:
            l.append(classifier.predict(j))
        l = np.array(l)
        # talvez seja necessario dar m reshape aqui
        f1_current = f1_score(y_t, l)
        f1_max = max(f1_max, f1_current)
        f1.append(f1_current)
    return (f1_max)


def main():
    pass


if __name__ == '__main__':
    main()