from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
import numpy as np


# load x_train.csv and y_train.csv
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
# drop the Id column
x_train = x_train.drop(['Id'], axis=1)
y_train = y_train.drop(['Id'], axis=1)
# convert the dataframe to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# apply the decision tree regression
clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, min_samples_leaf=11, splitter="best",
                             max_features="log2", max_leaf_nodes=50)
clf.fit(x_train, y_train.flatten())

# load x_train.csv and y_train.csv
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
# drop the Id column
x_train = x_train.drop(['Id'], axis=1)
y_train = y_train.drop(['Id'], axis=1)

fig = plt.figure(figsize=(16,14))
_ = tree.plot_tree(clf,
                   feature_names=x_train.columns,
                   class_names=y_train.columns,
                   filled=True)