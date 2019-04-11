# load data of features and labels
import numpy as np

def load_label():
    file_name = "data/AwA2-labels.txt"
    with open(file_name) as f:
        lines = f.readlines()
    y = [int(line.strip('\n')) for line in lines]
    return y


def load_feature():
    file_name = "data/AwA2-features.txt"
    with open(file_name) as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    X = [list(map(float, line.split(' '))) for line in lines]
    return X


def load_data():
    X = load_feature()
    y = load_label()
    return [X, y]

if __name__ == "__main__":
    X = load_feature()
    y = load_label()
    image_amount = len(X)
    feature_amount = len(X[0])
    label_amount = len(y)
    type_amount = len(set(y))
    print("images {}, features {}, labels {}, types {}".format(
          image_amount, feature_amount, label_amount, type_amount))
