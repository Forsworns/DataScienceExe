from sklearn import svm
from pre_process import pre_process

if __name__ == "__main__":
	X_train, X_test, y_train, y_test = pre_process()
	clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
	clf.fit(X_train, y_train) 
	clf.predict(X_test)
