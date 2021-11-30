from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def rmse_DT(xFeat, y):
    clf = DecisionTreeRegressor(criterion="friedman_mse", max_depth=5, min_samples_leaf=11, splitter="best",
                                min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50)
    # use the kfold here to split into k folds
    k_fold_CV = KFold(n_splits=5, shuffle=True)
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

def rmse_lr(xFeat, y):
    # build the linear regression model
    reg = LinearRegression()
    # use the kfold here to split into k folds
    k_fold_CV = KFold(n_splits=5, shuffle=True)
    # create a list to store the rmse
    rmse = []
    # a for loop is used to calculate the training accuracy and testing accuracy for each fold
    for train_index, test_index in k_fold_CV.split(xFeat):
        # split the data into training and testing
        x_train = xFeat[train_index, :]
        x_test = xFeat[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        # fit the model
        reg.fit(x_train, y_train)
        # predict the y_test
        y_pred = reg.predict(x_test)
        # calculate the rmse
        rmse.append(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    # return the average rmse
    return rmse


def rmse_rf(xFeat, y):
    # build the random forest regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=11,
                               min_weight_fraction_leaf=0.1, max_features="log2", max_leaf_nodes=50)
    # use the kfold here to split into k folds
    k_fold_CV = KFold(n_splits=5, shuffle=True)
    # create a list to store the rmse
    rmse = []
    # a for loop is used to calculate the training accuracy and testing accuracy for each fold
    for train_index, test_index in k_fold_CV.split(xFeat):
        # split the data into training and testing
        x_train = xFeat[train_index, :]
        x_test = xFeat[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        # fit the model
        rf.fit(x_train, y_train.flatten())
        # predict the y_test
        y_pred = rf.predict(x_test)
        # calculate the rmse
        rmse.append(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    # return the average rmse
    return rmse


def rmse_svm(xFeat, y):
    # build the svm regressor
    svm = SVR(kernel="rbf")
    # use the kfold here to split into k folds
    k_fold_CV = KFold(n_splits=5, shuffle=True)
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
        svm.fit(x_train, y_train.flatten())
        # predict the y_test
        y_pred = svm.predict(x_test)
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
    rmse_lr_list = rmse_lr(x_train, y_train)
    # print the average rmse
    print("The average rmse of linear regression is: ", np.mean(rmse_lr_list))
    # print the std of rmse
    print("The std of rmse of linear regression is: ", np.std(rmse_lr_list))


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
    rmse_dt_list = rmse_DT(x_train, y_train)
    # print the average rmse
    print("The average rmse of decision tree is: ", np.mean(rmse_dt_list))
    # print the sd of rmse
    print("The sd of rmse of decision tree is: ", np.std(rmse_dt_list))


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
    rmse_rf_list = rmse_rf(x_train, y_train)
    # print the average rmse
    print("The average rmse of random forest is: ", np.mean(rmse_rf_list))
    # print the sd of rmse
    print("The sd of rmse of random forest is: ", np.std(rmse_rf_list))

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
    rmse_svr_list = rmse_svm(x_train, y_train)
    # the average rmse
    print("The average rmse of svr is: ", np.mean(rmse_svr_list))
    # standard deviation of rmse
    print("The standard deviation of svr is: ", np.std(rmse_svr_list))

    # plot the four rmse list
    plt.plot(rmse_lr_list, label="linear regression")
    plt.plot(rmse_dt_list, label="decision tree")
    plt.plot(rmse_rf_list, label="random forest")
    plt.plot(rmse_svr_list, label="support vector regression")
    plt.legend()
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title("RMSE of different models")
    plt.show()


    # load x_train, y_train, x_test, y_test
    x_train_pca = pd.read_csv('x_train_pca.csv')
    y_train = pd.read_csv('y_train.csv')
    # drop the Id column
    y_train = y_train.drop(['Id'], axis=1)
    # change df to numpy array
    x_train_pca = np.array(x_train_pca)
    y_train = np.array(y_train)
    # apply the rmse function
    rmse_list_pca = rmse_DT(x_train_pca, y_train)
    # find the average rmse
    print("The average rmse with PCA is:", np.mean(rmse_list_pca))
    # find the standard deviation
    print("The standard deviation of rmse with PCA is:", np.std(rmse_list_pca))
    # plot the rmse_list_pca and rmse_dt_list
    plt.plot(rmse_list_pca, label='PCA')
    plt.plot(rmse_dt_list, label='Normal')
    plt.legend()
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE with PCA and Normal Data')
    plt.show()



if __name__ == "__main__":
    main()