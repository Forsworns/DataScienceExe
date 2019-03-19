from sklearn.svm import SVC
from sklearn.metrics import f1_score

from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *


def SVM_base(model_name, X, y):
    X_train, X_test, y_train, y_test = pre_process(X, y)
    for d in DECI_FUNCS:
            for k in KERNELS:
                for C in CS:
                    paras = {'C': C, 'kernel': k, 'decision_function_shape': d}

                    result = load_result(model_name, paras)
                    if result is None:
                        clf = load_model(model_name, paras)
                        if clf is None:
                            clf = SVC(**paras)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        sc = clf.score(X_test, y_test)
                        # unweighted mean of metrics for labels
                        f1_sc = f1_score(y_test, y_pred, average='macro')
                        result = {'score': sc, 'f1_score': f1_sc}
                        save_model(clf, model_name, paras)
                        save_result(result, model_name, paras)
                        print("{} with {}: score is {}, f1_score is {}".format(
                            model_name, paras, sc, f1_sc))
                    else:
                        sc, f1_sc = result.values()
                        print("{} with {}: score is {}, f1_score is {}".format(
                            model_name, paras, sc, f1_sc))


if __name__ == "__main__":
    model_name = BASELINE
    X, y = load_data()
    SVM_base(BASELINE, X, y)
