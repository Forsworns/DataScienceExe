import sys
sys.path.append('..')

import numpy as np
from sklearn import metrics

from configs import *
from baseline import SVM
from pre_process import pre_process
from baseline import SVM_recommend

def evaluateScore(X, y):
    X_train, X_test, y_train, y_test = pre_process(X, y)
    clf = SVM_recommend()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    auc = metrics.roc_auc_score(y_test, y_pred)
    return auc

def selectionLoop(self, X, y):
    score_history = []
    good_features = set([])
    num_features = X.shape[1]
    while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
        scores = []
        for feature in range(num_features):
            if feature not in good_features:
                selected_features = list(good_features) + [feature]
                Xts = np.column_stack(X[:, j] for j in selected_features)
                score = self.evaluateScore(Xts, y)
                scores.append((score, feature))
                if self._verbose:
                    print("Current AUC : ", np.mean(score))
        good_features.add(sorted(scores)[-1][1])
        score_history.append(sorted(scores)[-1])
        if self._verbose:
            print("Current Features : ", sorted(list(good_features)))

    # Remove last added feature
    good_features.remove(score_history[-1][1])
    good_features = sorted(list(good_features))
    if self._verbose:
        print("Selected Features : ", good_features)
    return good_features

def transform(self, X):
    X = self._data
    y = self._labels
    good_features = self.selectionLoop(X, y)
    return X[:, good_features]


if __name__ == "__main__":


