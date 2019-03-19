from sklearn.svm import SVC
from sklearn.metrics import f1_score

from pre_process import pre_process
from sl_rm import * 
from configs import *

# paras for baseline
deci_funcs = ['ovo', 'ovr']
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
Cs = [0.5, 1, 2]


if __name__ == "__main__":
	X_train, X_test, y_train, y_test = pre_process()
	for d in deci_funcs:
		for k in kernels:
			for C in Cs:
				paras = {'C': C, 'kernel': k, 'decision_function_shape': d}

				result = load_result(BASELINE, paras)
				if result is None:
					clf = load_model(BASELINE, paras)
					if clf is None:
						clf = SVC(**paras)
					clf.fit(X_train, y_train)
					y_pred = clf.predict(X_test)
					sc = clf.score()
					# unweighted mean of metrics for labels
					f1_sc = clf.f1_score(y_test, y_pred, average='macro')
					result = {'score': sc, 'f1_score': f1_sc}
					save_model(clf,BASELINE,paras)
					save_result(result,BASELINE, paras)
					print("{} with {}: score is {}, f1_score is {}".format(BASELINE, paras, sc, f1_sc))
				else:
					sc, f1_sc = result.values()
					print("{} with {}: score is {}, f1_score is {}".format(BASELINE, paras, sc, f1_sc))		
