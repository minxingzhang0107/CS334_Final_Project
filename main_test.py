from model.paw_net import PawNet

if __name__ == '__main__':
    paw_net = PawNet(input_shape=256, epochs=12)
    # paw_net.train(evaluate=True)
    paw_net.load_model('deep_model.pt')
    paw_net.deep.to(paw_net.device)
    # paw_net.explain()
    paw_net.predict(paw_net.initialize_dataloader()[1])
