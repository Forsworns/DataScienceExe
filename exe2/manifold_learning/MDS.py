import sys
sys.path.append('..')
from sklearn.neighbors import KNeighborsClassifier
from baseline import KNN_recommend_run
import numpy as np
from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *
from sklearn.manifold import MDS
import os

if __name__ == "__main__":
	os.chdir("..")	
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	mds = MDS(n_components=2, max_iter=100, n_init=1)
	mds.fit(X_train,y_train)
	X_train = mds.transform(X_train)
	X_test = mds.transform(X_test)
	KNN_recommend_run("MDS",X_train, X_test, y_train, y_test)
