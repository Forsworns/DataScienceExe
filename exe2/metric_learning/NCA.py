import sys
sys.path.append('..')
from metric_learn import NCA
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os


if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    X_train, X_test, y_train, y_test = pre_process(X,y)
    nca =  NCA(max_iter=100,verbose=True)
    nca.fit(X_train,y_train)
    X_train = nca.transform(X_train)
    X_test = nca.transform(X_test)
    KNN_recommend_run("NCA",X_train,X_test,y_train,y_test,paras={})