from main_test import kfold_test
from model.resnet import ConvNet

if __name__ == '__main__':
    kfold_test(model=ConvNet)
