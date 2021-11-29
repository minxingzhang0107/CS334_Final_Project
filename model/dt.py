from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


class DecisionTreePawpularity(object):
    def __init__(self, criterion="friedman_mse", max_depth=5, min_samples_leaf=11, splitter="best",
                 min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50):
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
        clf = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf, splitter=self.splitter,
                                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                    max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes)
        clf.fit(x_train, y_train.flatten())
        return clf

    def predict(self, x_test, y_test):
        # drop the Id column
        x_test = x_test.drop(['Id'], axis=1)
        y_test = y_test.drop(['Id'], axis=1)
        # change df to numpy array
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        y_pred = self.model.predict(x_test)
        # calculate the root mean squared error
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        return rmse


if __name__ == '__main__':
    x_train = pd.read_csv('../x_train.csv')
    y_train = pd.read_csv('../y_train.csv')
    x_test = pd.read_csv('../x_test.csv')
    y_test = pd.read_csv('../y_test.csv')

    # create the model
    model = DecisionTreePawpularity()
    # train the model
    model.train(x_train, y_train)
    # predict the test data
    rmse = model.predict(x_test, y_test)
    print(rmse)
