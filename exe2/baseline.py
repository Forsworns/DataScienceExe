from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from operator import methodcaller

from distance import *
from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

def KNN_recommend(**NN_paras):
    if NN_paras == {}:
        NN_paras = {'n_neighbors':2,'algorithm':'kd_tree'}
    return KNeighborsClassifier(NN_paras)

def KNN_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={}, **SVM_paras):
    if NN_paras == {}:
        NN_paras = {'n_neighbors':2,'algorithm':'kd_tree'}
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
    # 确定KNN的metric和近邻数目
    if NN_paras == {}:
        NN_paras = {'n_neighbors':2,'algorithm':'kd_tree'}
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


def KNN_run(X, y):
    # model_name为采用的降维方法，X为降维后的feature数据
    X_train, X_test, y_train, y_test = pre_process(X, y)
    for d in DECI_FUNCS:
        for k in KERNELS:
            for C in CS:
                SVM_compare_run(COMPARE, X_train, X_test, y_train, y_test, paras={}, C=C, kernel=k, decision_function_shape=d)
    ''' for C in CS:
        SVM_recommend_run(BASELINE, X_train, X_test, y_train, y_test, paras={}, C=C, max_iter=2000) '''


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = pre_process(X,y)
    random_P = np.random.rand([1,X_train])
    random_M = random_P.T*random_P
    dist_obj = Distance(X_train,random_M)
    for dist_func in DIST_LIST:
        metric = methodcaller(dist_func)(dist_obj)
        clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto',metric=metric)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        sc = clf.score(X_test, y_test)
        # unweighted mean of metrics for labels
        f1_sc = f1_score(y_test, y_pred, average='macro')
        result = {'score': sc, 'f1_score': f1_sc}
