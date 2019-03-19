from sklearn.feature_selection import VarianceThreshold
import numpy as np

from configs import *

FORWARD_ACCURACY = 0.1


def framework(X,selector):
    acc = 0
    while acc < REDUCED_ACCURACY:
        selector = VarianceThreshold(threshold=1)
        selector.fit_transform(X)

def reduce(X):
    framework(X)

if __name__ == "__main__":
    X, y = load_data()

    X = reduce(X)

    X_train, X_test, y_train, y_test = pre_process(X,y)


