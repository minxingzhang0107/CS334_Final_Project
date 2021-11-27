from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# load the training data
train_data = pd.read_csv('train.csv')
# split the train_data into features and labels
y_train = train_data[['Id','Pawpularity']]
x_train = train_data.drop('Pawpularity', axis=1)
# apply train_test_split to the data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# save to csv
x_train.to_csv('x_train.csv', index=False)
x_test.to_csv('x_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)





