import numpy as np
import Acitivation


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
        self.outputlayer = self.Layer(outputdim, 'sigmoid')
        self.inputdim = inputdim
        self.learningrate = learningrate
        self.Niter = niter
        self.layers = []
        self.weights = []
        self.biases = []

    def add_input(self, ndim):
        # Unless given the input_dim, it can be implemented here.
        self.inputdim = ndim

    def add_layer(self, nodes_number, activation='sigmoid'):
        """ Add a specified full-connected layer with numbers of nodes:
        nodes_number: stands for numbers of the nodes in this layer."""
        self.layers.append(self.Layer(nodes_number, activation))
        print(f"Added a {nodes_number} nodes layer at No.{len(self.layers)} place with '{activation}' method")

    def add_output(self, ndim, activation='sigmoid'):
        # Unless given the output_dim, it can be implemented here.
        self.outputlayer = self.Layer(ndim, activation)

    def setup(self):
        # Initialize the network with random values
        if not self.layers:
            raise NotImplementedError("Not given layers yet.")
        self.layers.append(self.outputlayer)
        weights = []
        biases = []
        w1 = np.random.random((self.layers[0].num_nodes, self.inputdim))
        weights.append(w1)
        for i in range(len(self.layers)-1):
            matrix = np.random.random((self.layers[i+1].num_nodes, self.layers[i].num_nodes))
            bias = np.random.random((self.layers[i].num_nodes, 1))
            weights.append(matrix)
            biases.append(bias)
        bias = np.random.random((self.layers[-1].num_nodes, 1))
        biases.append(bias)
        self.weights = weights
        self.biases = biases
        print("Network Initialized.")

    def _weights(self):
        print(self.weights)

    def _biases(self):
        print(self.biases)

    def forward(self, inputdata):
        if inputdata.shape[0] != self.inputdim:
            raise ValueError("The dimension of the input doesn't match.")
        weights = self.weights
        biases = self.biases
        activate = Acitivation.activate(self.layers[0].activation)
        weight, bias = weights[0], biases[0]
        output = weight @ inputdata + bias
        activated = activate(output)
        for i in range(1, len(weights)):
            weight, bias = weights[i], biases[i]
            output = weight @ activated + bias
            activate = Acitivation.activate(self.layers[i].activation)
            activated = activate(output)
        print(activated)

    def back_propagation(self):
        pass

    class Layer:
        def __init__(self, number, activation):
            self.num_nodes = number
            self.activation = activation

        def calculate(self):

            pass

        def activate(self):
            pass






