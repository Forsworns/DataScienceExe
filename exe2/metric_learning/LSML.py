import sys
sys.path.append('..')
from metric_learn import LSML_Supervised
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

if __name__ == '__main__':
    os.chdir('..')
    X, y = load_data()
    lsml =  LSML_Supervised(num_constraints=200)
    lsml.fit(X,y)
    X = lsml.transform(X)
    X_train, X_test, y_train, y_test = pre_process(X,y)
    KNN_recommend_run("LSML",X_train,X_test,y_train,y_test,paras={})
