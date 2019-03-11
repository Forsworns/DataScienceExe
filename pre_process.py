import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_data


def pre_process():
	X, y = load_data()
	X = np.asarray(X)
	y = np.asarray(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
	return [X_train, X_test, y_train, y_test]


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = pre_process()
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)
