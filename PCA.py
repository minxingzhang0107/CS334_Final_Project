import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# load x_train, y_train, x_test, y_test
x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')
# drop the Id column
x_train = x_train.drop(['Id'], axis=1)
x_test = x_test.drop(['Id'], axis=1)
# change df to numpy array
x_train = np.array(x_train)
x_test = np.array(x_test)
# apply the PCA
pca = PCA(n_components=7)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
# change x_train_pca, x_test_pca to df
x_train_pca = pd.DataFrame(x_train_pca)
x_test_pca = pd.DataFrame(x_test_pca)
# save to csv
x_train_pca.to_csv('x_train_pca.csv', index=False)
x_test_pca.to_csv('x_test_pca.csv', index=False)
