import numpy as np


activate_method = {
            'sigmoid': lambda x: 1/(1+np.exp(-x)),
            'relu': lambda x: 0 if x < 0 else x,
            # tanh = sinh/cosh
            'tanh': lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)),
            'leakyrelu': lambda x: 0.1*x if x < 0 else x,
            'elu': lambda x, a: a*(np.exp(x)-1) if x < 0 else x
}


def activate(method):
    return activate_method[method]
