# file type
MODEL = "models"
RESULT = "results"

# models
BASELINE = "SVC_baseline"
COMPARE = "SVC_compare"
GA = "genetic_algorithm"
B_VT = "backward_variance_threshold"
F_UF = "forward_univariable_feature"
B_SFM = "backward_select_from_model"
AUC = "AUC_ROC"
BORUTA = "boruta"

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
CS = [0.01, 0.04, 0.07, 0.1, 0.3, 0.5, 1, 2]