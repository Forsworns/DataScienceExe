# file type
MODEL = "models"
RESULT = "results"

# models
BASELINE = "SVC_baseline"
GA = "genetic_algorithm"
B_VT = "backward_variance_threshold"
F_UF = "forward_univariable_feature"
B_SFM = "backward_select_from_model"
AUC = "AUC_ROC"

# test train split
TEST_SIZE = 0.4

# paras for SVM base
# DECI_FUNCS = ['ovo', 'ovr']
DECI_FUNCS = ['ovr']
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
CS = [0.5, 1, 2]