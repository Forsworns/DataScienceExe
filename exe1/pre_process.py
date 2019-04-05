import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

from load_data import load_data
from configs import *

# 预处理分划数据

def pre_process(X=None,y=None,bReset=False):
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
