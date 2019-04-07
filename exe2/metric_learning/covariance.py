import sys
sys.path.append('..')
from metric_learn import Covariance
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process

if __name__ == '__main__':
    X, y = load_data()
    cov =  Covariance().fit(X)
    X = cov.transform(X)
    X_train, X_test, y_train, y_test = pre_process(X,y)
    KNN_recommend_run("Covariance",X_train,X_test,y_train,y_test,paras={})
