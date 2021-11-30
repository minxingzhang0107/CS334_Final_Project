from model.paw_net import PawNet

if __name__ == '__main__':
    paw_net = PawNet(input_shape=256)
    paw_net.initialize_dataloader()
    paw_net.train(evaluate=True)
