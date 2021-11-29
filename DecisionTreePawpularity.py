from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV


class DecisionTreePawpularity(object):
    criterion = "friedman_mse"
    max_depth = 5
    min_samples_leaf = 11
    splitter = "best"
    min_weight_fraction_leaf = 0.1
    max_features = "log2"
    max_leaf_nodes = 50

    def __init__(self, criterion, max_depth, min_samples_leaf, splitter, min_weight_fraction_leaf, max_features, max_leaf_nodes):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def train(self, x_train, y_train):
        # drop the Id column
        x_train = x_train.drop(['Id'], axis=1)
        y_train = y_train.drop(['Id'], axis=1)
        # change df to numpy array
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # apply the decision tree regression
        clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=5, min_samples_leaf=11, splitter="best",
                            min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50)
        clf.fit(x_train, y_train.flatten())
        return clf

    def predict(self, clf, x_test, y_test):
        # drop the Id column
        x_test = x_test.drop(['Id'], axis=1)
        y_test = y_test.drop(['Id'], axis=1)
        # change df to numpy array
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        # predict the test set
        y_pred = clf.predict(x_test)
        # calculate the root mean squared error
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        return rmse
