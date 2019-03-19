# file type
MODEL = "models"
RESULT = "results"

# models
BASELINE = "SVC_baseline"
GA = "genetic_algorithm"
B_VT = "backward_variance_threshold"
F_UF = "forward_univariable_feature"
B_SFM = "backward_select_from_model"

# test train split
TEST_SIZE = 0.4

# paras for SVM base
DECI_FUNCS = ['ovo', 'ovr']
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
CS = [0.5, 1, 2]