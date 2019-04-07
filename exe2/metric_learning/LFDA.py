import sys
sys.path.append('..')
from metric_learn import LFDA
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    lfda =  LFDA(num_constraints=200)
    lfda.fit(X)
    X = lfda.transform(X)
    X_train, X_test, y_train, y_test = pre_process(X,y)
    KNN_recommend_run("LFDA",X_train,X_test,y_train,y_test,paras={})
