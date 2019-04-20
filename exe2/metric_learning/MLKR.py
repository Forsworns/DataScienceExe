import sys
sys.path.append('..')
import os
from pre_process import pre_process
from load_data import load_data
from baseline import KNN_recommend_run
from configs import *
from metric_learn import MLKR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    X_train, X_test, y_train, y_test = pre_process(X, y)

    lda = LinearDiscriminantAnalysis(n_components=50)
    lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)

    mlkr = MLKR(verbose=True)
    mlkr.fit(X_train, y_train)
    X_train = mlkr.transform(X_train)
    X_test = mlkr.transform(X_test)
    KNN_recommend_run("LDAMLKR", X_train, X_test, y_train, y_test, paras={})
