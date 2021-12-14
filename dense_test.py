from model.paw_net import PawNet
from model.DenseNet import DenseNet

if __name__ == '__main__':
    paw_net = PawNet(input_shape=56, epochs=12)
    paw_net.deep = DenseNet()
    paw_net.train(evaluate=True)
