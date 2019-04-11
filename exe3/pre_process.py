import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

from load_data import load_data
from configs import *

# 预处理分划数据

def pre_process(X=None,y=None,bReset=True):
	# 之前没有存下来，导致每次test set和train set会变化
	if os.path.exists(X_TRAIN) and os.path.exists(X_TEST) and os.path.exists(Y_TRAIN) and os.path.exists(Y_TEST) and not bReset:
		X_train = np.load(X_TRAIN)
		X_test = np.load(X_TEST)
		y_train = np.load(Y_TRAIN)
		y_test = np.load(Y_TEST)
		return [X_train, X_test, y_train, y_test]
	else:
		X = np.asarray(X)
		y = np.asarray(y)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1)
		np.save(X_TRAIN, X_train)
		np.save(X_TEST, X_test)
		np.save(Y_TRAIN,y_train)
		np.save(Y_TEST,y_test)
		return [X_train, X_test, y_train, y_test]


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = pre_process()
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)
