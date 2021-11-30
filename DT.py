from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

# load x_train, y_train, x_test, y_test
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')
# drop the Id column
x_train = x_train.drop(['Id'], axis=1)
x_test = x_test.drop(['Id'], axis=1)
y_train = y_train.drop(['Id'], axis=1)
y_test = y_test.drop(['Id'], axis=1)
# change df to numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# apply the decision tree regression
clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=5, min_samples_leaf=11, splitter="best",
                            min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50)
clf.fit(x_train, y_train.flatten())
y_pred = clf.predict(x_test)
# calculate the root mean squared error
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print('RMSE:', rmse)
# use grid search to find the best parameters
parameters = {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
              "max_depth": range(1, 15),
              "min_samples_leaf": range(1, 100, 10),
              "splitter": ["best", "random"],
              "min_weight_fraction_leaf": [0.1, 0.3, 0.5],
              "max_features": ["auto", "log2", "sqrt", None],
              "max_leaf_nodes": [None, 10, 30, 50, 70, 90]
              }
model_tuning = GridSearchCV(DecisionTreeRegressor(), param_grid=parameters, scoring='neg_root_mean_squared_error'
                            , cv=5, verbose=1, n_jobs=14)
model_tuning.fit(x_train, y_train.flatten())
print(model_tuning.best_params_)

# load x_train, y_train, x_test, y_test
x_train_pca = pd.read_csv('x_train_pca.csv')
y_train = pd.read_csv('y_train.csv')
x_test_pca = pd.read_csv('x_test_pca.csv')
y_test = pd.read_csv('y_test.csv')
# drop the Id column
y_train = y_train.drop(['Id'], axis=1)
y_test = y_test.drop(['Id'], axis=1)
# change df to numpy array
x_train_pca = np.array(x_train_pca)
x_test_pca = np.array(x_test_pca)
y_train = np.array(y_train)
y_test = np.array(y_test)
# apply the decision tree regression
clf = DecisionTreeRegressor(max_depth=255)
clf.fit(x_train_pca, y_train.flatten())
y_pred = clf.predict(x_test_pca)
# calculate the root mean squared error
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print('RMSE (pca):', rmse)
