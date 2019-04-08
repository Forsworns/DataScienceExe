import sys
sys.path.append('..')
from sklearn.neighbors import KNeighborsClassifier
from baseline import KNN_comapre_run
import numpy as np
from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

if __name__ == "__main__":
	os.chdir("..")	
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	lda = LinearDiscriminantAnalysis(n_components=50)
	lda.fit(X_train,y_train)
	X_train = lda.transform(X_train)
	X_test = lda.transform(X_test)
	KNN_comapre_run(X_train, X_test, y_train, y_test,model_name=LDA)
