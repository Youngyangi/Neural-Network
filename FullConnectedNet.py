import numpy as np


class FullConnectedNet:
    """
    The Full Connected Network is the basic type of a neural network, each node
    in each layer is fully connected with the one before and after itself.
    """
    def __init__(self, inputdim=0, outputdim=0, learningrate=0.05, niter=1e6):
        """
        :param inputdim: indicates the dim of the input.
        :param outputdim: indicates the dim of the output, usually depends on how many categories or
        how many features the network trains.
        :param learningrate: the scale of the gradient influences on each update, should be adjusted.
        :param niter: numbers of iterations when applying stochastic gradient descent.
        """
        self.outputdim = outputdim
        self.inputdim = inputdim
        self.learningrate = learningrate
        self.Niter = niter
        self.layers = []
        self.weights = []
        self.biases = []

    def add_input(self, ndim):
        # Unless given the input_dim, it can be implemented here.
        self.inputdim = ndim

    def add_layer(self, nodes_number):
        """ Add a specified full-connected layer with numbers of nodes:
        nodes_number: stands for numbers of the nodes in this layer."""
        self.layers.append(nodes_number)
        print(f"Added a {nodes_number} nodes layer at No.{len(self.layers)} place")

    def add_output(self, ndim):
        # Unless given the output_dim, it can be implemented here.
        self.outputdim = ndim

    def setup(self):
        # Initialize the network with random values
        if not self.layers:
            raise NotImplementedError("Not given layers yet.")
        weights = []
        biases = []
        w1 = np.random.random((self.layers[0], self.inputdim))
        w2 = np.random.random((self.layers[-1], self.outputdim))
        weights.append(w1)
        for i in range(len(self.layers)-1):
            matrix = np.random.random((self.layers[i+1], self.layers[i]))
            bias = np.random.random((self.layers[i], 1))
            weights.append(matrix)
            biases.append(bias)
        weights.append(w2)

        bias1 = np.random.random((self.layers[-1], 1))
        bias2 = np.random.random((self.outputdim, 1))
        biases.append(bias1)
        biases.append(bias2)
        self.weights = weights
        self.biases = biases
        print("Network Initialized.")

    def forward(self):
        pass


    # class Nodes:
    #     def __init__(self):




