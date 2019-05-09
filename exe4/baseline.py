from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

def KNN_recommend(**NN_paras):
	if NN_paras == {}:
		NN_paras = KNN_PARAS
	return KNeighborsClassifier(NN_paras)

def KNN_recommend_run(model_name, X_train, X_test, y_train, y_test, bStore=False, paras={}, **NN_paras):
	if NN_paras == {}:
		NN_paras = KNN_PARAS
	if paras == {}:
		paras.update(NN_paras)
	result = load_result(model_name, paras)
	if result is None:
		clf = load_model(model_name, paras)
		if clf is None:
			print("can't find clf",model_name)
			clf = KNeighborsClassifier(**NN_paras)
			print(clf)
			clf.fit(X_train, y_train)
			if bStore:
				save_model(clf, model_name, paras)
		sc = clf.score(X_test, y_test)
		# unweighted mean of metrics for labels
		result = {'score': sc}
		save_result(result, model_name, paras)
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf
	else:
		clf = load_model(model_name, paras)
		sc = result.values()
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf
	
def SVM_recommend(**SVM_paras):
	if SVM_paras == {}:
		SVM_paras = SVM_PARAS
	return LinearSVC(**SVM_paras)


def SVM_recommend_run(model_name, X_train, X_test, y_train, y_test, bStore=False, paras={}, **SVM_paras):
	if SVM_paras == {}:
		SVM_paras = SVM_PARAS
	if paras == {}:
		paras.update(SVM_paras)
	result = load_result(model_name, paras)
	if result is None:
		clf = load_model(model_name, paras)
		if clf is None:
			print("can't find clf", model_name)
			clf = LinearSVC(**SVM_paras)
			clf.fit(X_train, y_train)
			if bStore:
				save_model(clf, model_name, paras)
		sc = clf.score(X_test, y_test)
		# unweighted mean of metrics for labels
		result = {'score': sc}
		save_result(result, model_name, paras)
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf
	else:
		clf = load_model(model_name, paras)
		sc = result.values()
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf


if __name__ == "__main__":
	X_src, y_src, X_tgt, y_tgt = load_data()
	# baseline, train on the srouce domain, test on the target domain
	SVM_recommend_run(BASELINE_SVM, X_src, X_tgt, y_src, y_tgt)
	KNN_recommend_run(BASELINE_SVM, X_src, X_tgt, y_src, y_tgt)
