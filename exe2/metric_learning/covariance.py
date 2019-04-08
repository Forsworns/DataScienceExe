import sys
sys.path.append('..')
from metric_learn import Covariance
from configs import *
from baseline import KNN_recommend_run
from load_data import load_data
from pre_process import pre_process
import os

# baseline, doesn't learn, just calculate the covariances
if __name__ == '__main__':
	os.chdir('..')
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	cov =  Covariance().fit(X_train)
	X_train = cov.transform(X_train)
	X_test = cov.transform(X_test)
	KNN_recommend_run("Covariance",X_train,X_test,y_train,y_test,paras={})

