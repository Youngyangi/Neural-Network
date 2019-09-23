import FullConnectedNet

nn = FullConnectedNet.FullConnectedNet(inputdim=2, outputdim=5)
nn.add_layer(3)
nn.add_layer(4)
nn.setup()
