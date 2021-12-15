from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


def rmse(xFeat, y):
    clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=5, min_samples_leaf=11, splitter="best",
                                min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50)
    # use the kfold here to split into k folds
    k_fold_CV = KFold(n_splits=10, shuffle=True)
    # create a list to store the rmse
    rmse = []
    # a for loop is used to calculate the training accuracy and testing accuracy for each fold
    for train_index, test_index in k_fold_CV.split(xFeat):
        # split the data into training and testing
        x_train = xFeat[train_index, :]
        x_test = xFeat[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        # fit the model using x_train and y_train
        clf.fit(x_train, y_train)
        # predict the y_test
        y_pred = clf.predict(x_test)
        # calculate the rmse
        rmse.append(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    # return the average rmse
    return rmse


def main():
    # load x_train.csv and y_train.csv
    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    # drop the Id column
    x_train = x_train.drop(['Id'], axis=1)
    y_train = y_train.drop(['Id'], axis=1)
    # convert the dataframe to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # apply the rmse function
    rmse_list = rmse(x_train, y_train)
    # print the average rmse
    print("The average rmse is: ", np.mean(rmse_list))
    # print the standard deviation of rmse
    print("The standard deviation of rmse is: ", np.std(rmse_list))
    # plot the rmse
    plt.plot(rmse_list)
    plt.show()


if __name__ == "__main__":
    main()





