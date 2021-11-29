from .conv_net import ConvNet
from sklearn.tree import DecisionTreeClassifier


class PawNet:
    def __init__(self, input_shape, num_classes, x_train_path='x_train.csv',
                 y_train_path='y_train.csv', x_test_path='x_test.csv', y_test='y_test.csv'):
        self.cnn = ConvNet(input_shape)
        self.shallow = DecisionTreeClassifier(max_depth=4)

    def train(self):
        pass

    def predict(self):
        pass
