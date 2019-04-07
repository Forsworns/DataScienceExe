import sys
sys.path.append('..')
from metric_learn import LMNN
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    lmnn =  LMNN(k=5,learn_rate=1e-6)
    lmnn.fit(X,y)
    X = lmnn.transform(X)
    X_train, X_test, y_train, y_test = pre_process(X,y)
    KNN_recommend_run("LMNN",X_train,X_test,y_train,y_test,paras={})
