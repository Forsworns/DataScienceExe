import numpy as np

# type a art->reality
A_A = "data/a/A_A.csv"
A_R = "data/a/A_R.csv"
# type b clipart->reality
C_C = "data/b/C_C.csv"
C_R = "data/b/C_R.csv"
# type c product->reality
P_P = "data/c/P_P.csv"
P_R = "data/c/P_R.csv"

SRC_FILE = [A_A,C_C,P_P]
TGT_FILE = [A_R,C_R,P_R]

# file type
MODEL = "models"
RESULT = "results"

# methods name/model name
BASELINE_SVM = "baseline_svm"
BASELINE_KNN = "baseline_knn"
GFK = "GFK"

# test train split
TEST_SIZE = 0.4
# stored testing set and training set
X_TRAIN = "data/train_x.npy"
Y_TRAIN = "data/train_y.npy"
X_TEST = "data/test_x.npy"
Y_TEST = "data/test_y.npy"

# paras for baseline
SVM_PARAS = {'C': 0.01, 'kernel': 'linear', 'max_iter': 2000}

COLORS = np.array(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F',  # brown
                   '#000000'   # black
                   ])