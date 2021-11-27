from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
# apply the random forest classifier
clf = RandomForestClassifier(n_estimators=255, max_depth=255, min_samples_leaf=3, criterion='gini', random_state=0)
clf.fit(x_train, y_train.flatten())
# predict the test set
y_pred = clf.predict(x_test)
# calculate the root mean squared error
rms = np.sqrt(np.mean((y_pred - y_test.flatten()) ** 2))
print('The RMSE is: ', rms)

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
# apply the random forest classifier
clf = RandomForestClassifier(n_estimators=255, max_depth=255, min_samples_leaf=3, criterion='gini', random_state=0)
clf.fit(x_train_pca, y_train.flatten())
# predict the test set
y_pred = clf.predict(x_test_pca)
# calculate the root mean squared error
rms = np.sqrt(np.mean((y_pred - y_test.flatten()) ** 2))
print('The RMSE(PCA) is: ', rms)



