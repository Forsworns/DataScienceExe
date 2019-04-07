import sys
sys.path.append('..')
from metric_learn import ITML_Supervised
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    itml =  ITML_Supervised(num_constraints=200)
    itml.fit(X)
    X = itml.transform(X)
    X_train, X_test, y_train, y_test = pre_process(X,y)
    KNN_recommend_run("ITML",X_train,X_test,y_train,y_test,paras={})
