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
	X_train, X_test, y_train, y_test = pre_process(X,y)
	itml =  ITML_Supervised(num_constraints=200)
	itml.fit(X_train,y_train)
	X_train = itml.transform(X_train)
	X_test = itml.transform(X_test)
	KNN_recommend_run("ITML",X_train,X_test,y_train,y_test,paras={})
