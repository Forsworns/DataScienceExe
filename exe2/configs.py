import numpy as np

# file type
MODEL = "models"
RESULT = "results"

# distance
DIST_LIST = ['euclidean','manhattan','chebyshev','minkowski_3','minkowski_4','mahalanobis','canberra','braycurtis','cosine']

# test train split
TEST_SIZE = 0.4
# stored testing set and training set
X_TRAIN = "data/train_x.npy"
Y_TRAIN = "data/train_y.npy"
X_TEST = "data/test_x.npy"
Y_TEST = "data/test_y.npy"

# paras for SVM base
DECI_FUNCS = ['ovo', 'ovr']
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
KERNELS_MAP = {'linear':0, 'poly':1, 'rbf':2, 'sigmoid':3}
CS = [0.01, 0.04, 0.07, 0.1, 0.3, 0.5, 1, 2]

COLORS = np.array(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])