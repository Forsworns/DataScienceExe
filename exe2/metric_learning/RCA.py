import sys
sys.path.append('..')
from metric_learn import RCA_Supervised
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    X_train, X_test, y_train, y_test = pre_process(X,y)
    rca =  RCA_Supervised(num_chunks=30, chunk_size=2)
    rca.fit(X_train,y_train)
    X_train = rca.transform(X_train)
    X_test = rca.transform(X_test)
    KNN_recommend_run("RCA",X_train,X_test,y_train,y_test,paras={})
