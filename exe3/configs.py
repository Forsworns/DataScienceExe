import numpy as np

# file type
MODEL = "models"
RESULT = "results"
COMPARE = "compare"
BASELINE = "baseline"

# test train split
TEST_SIZE = 0.4
# stored testing set and training set
X_TRAIN = "data/train_x.npy"
Y_TRAIN = "data/train_y.npy"
X_TEST = "data/test_x.npy"
Y_TEST = "data/test_y.npy"

COLORS = np.array(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])