from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
# apply SVR
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(x_train, y_train.flatten())
y_pred = svr.predict(x_test)
# calculate the rmse
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print('rmse: ', rmse)


