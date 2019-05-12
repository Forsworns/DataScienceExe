from sklearn.svm import SVC

from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

	
def SVM_recommend(**SVM_paras):
	if SVM_paras == {}:
		SVM_paras = SVM_PARAS
	return SVC(**SVM_paras)


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
			clf = SVC(**SVM_paras)
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
	for data in range(3):
		X_src, y_src, X_tgt, y_tgt = load_data(data)
		# baseline, train on the srouce domain, test on the target domain
		SVM_recommend_run(BASELINE_SVM, X_src, X_tgt, y_src, y_tgt,paras={"data":data})
