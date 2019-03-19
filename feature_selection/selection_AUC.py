from load_data import load_data
from baseline import SVM_recommend, SVM_recommend_run
from pre_process import pre_process
from baseline import SVM
from configs import *
from sklearn import metrics
import numpy as np
import sys
sys.path.append('..')


def evaluateScore(X, y):
    X_train, X_test, y_train, y_test = pre_process(X, y)
    clf = SVM_recommend()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = metrics.roc_auc_score(y_test, y_pred)
    return auc


def selectionLoop(X, y):
    score_history = []
    good_features = set([])
    num_features = X.shape[1]
    # 选择feature，直到AUC_ROC不再增加
    while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
        scores = []
        for feature in range(num_features):
            if feature not in good_features:
                selected_features = list(good_features) + [feature]
                Xts = np.column_stack(X[:, j] for j in selected_features)
                score = evaluateScore(Xts, y)
                scores.append((score, feature))
                print("Current AUC : ", np.mean(score))
        good_features.add(sorted(scores)[-1][1])
        score_history.append(sorted(scores)[-1])
        print("Current Features : ", sorted(list(good_features)))

    # Remove last added feature
    good_features.remove(score_history[-1][1])
    good_features = sorted(list(good_features))
    print("Selected Features : ", good_features)
    return good_features


def transform(X, y):
    good_features = selectionLoop(X, y)
    return X[:, good_features]


if __name__ == "__main__":
    X, y = load_data()
    X = transform(X, y)
    X_train, X_test, y_train, y_test = pre_process(X, y)
    SVM_recommend_run(AUC, X_train, X_test, y_train, y_test, {'feature_num':X.shape[1]})
