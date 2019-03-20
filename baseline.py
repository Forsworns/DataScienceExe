from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score

from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

def SVM_recommend(**SVM_paras):
    if SVM_paras == {}:
        SVM_paras = {'C': 0.01}
    return LinearSVC(**SVM_paras)

def SVM_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={}, **SVM_paras):
    if SVM_paras == {}:
        SVM_paras = {'C': 0.01}
    if paras == {}:
        paras.update(SVM_paras)
    result = load_result(model_name, paras)
    if result is None:
        clf = load_model(model_name, paras)
        if clf is None:
            print("can't find clf",model_name)
            clf = LinearSVC(**SVM_paras)
            clf.fit(X_train, y_train)
            save_model(clf, model_name, paras)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        # unweighted mean of metrics for labels
        f1_sc = f1_score(y_test, y_pred, average='macro')
        result = {'score': sc, 'f1_score': f1_sc}
        save_result(result, model_name, paras)
        print("{} with {}: score is {}, f1_score is {}".format(
            model_name, paras, sc, f1_sc))
        return clf
    else:
        clf = load_model(model_name, paras)
        sc, f1_sc = result.values()
        print("{} with {}: score is {}, f1_score is {}".format(
            model_name, paras, sc, f1_sc))
        return clf

def SVM_compare_run(model_name, X_train, X_test, y_train, y_test, paras={}, **SVM_paras):
    # 统一用于后续实验比较的SVM，提供前六个参数
    if SVM_paras == {}:
        SVM_paras = {'C': 2, 'kernel': 'rbf', 'decision_function_shape': 'ovo'}
    if paras == {}:
        paras.update(SVM_paras)
    result = load_result(model_name, paras)
    if result is None:
        clf = load_model(model_name, paras) # 有问题，训练集和测试集变了
        if clf is None:
            print("can't find clf",model_name)
            clf = SVC(**SVM_paras)
            clf.fit(X_train, y_train)
            save_model(clf, model_name, paras)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        # unweighted mean of metrics for labels
        f1_sc = f1_score(y_test, y_pred, average='macro')
        result = {'score': sc, 'f1_score': f1_sc}
        save_result(result, model_name, paras)
        print("{} with {}: score is {}, f1_score is {}".format(
            model_name, paras, sc, f1_sc))
        return clf
    else:
        clf = load_model(model_name, paras)
        sc, f1_sc = result.values()
        print("{} with {}: score is {}, f1_score is {}".format(
            model_name, paras, sc, f1_sc))
        return clf


def SVM_base(X, y):
    # model_name为采用的降维方法，X为降维后的feature数据
    X_train, X_test, y_train, y_test = pre_process(X, y)
    for d in DECI_FUNCS:
        for k in KERNELS:
            for C in CS:
                SVM_compare_run(COMPARE, X_train, X_test, y_train, y_test, paras={}, C=C, kernel=k, decision_function_shape=d)
    ''' for C in CS:
        SVM_recommend_run(BASELINE, X_train, X_test, y_train, y_test, paras={}, C=C, max_iter=2000) '''


if __name__ == "__main__":
    model_name = BASELINE
    X, y = load_data()
    SVM_base(X, y)
