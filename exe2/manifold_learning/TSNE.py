import sys
sys.path.append('..')
from sklearn.neighbors import KNeighborsClassifier
from baseline import KNN_recommend_run
import numpy as np
from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *
from sklearn.manifold import TSNE
import os

if __name__ == "__main__":
	os.chdir("..")	
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	tsne = TSNE(n_neighbors=10,n_components=2)
	tsne.fit(X_train,y_train)
	X_train = tsne.transform(X_train)
	X_test = tsne.transform(X_test)
	KNN_recommend_run("TSNE",sX_train, X_test, y_train, y_test)
