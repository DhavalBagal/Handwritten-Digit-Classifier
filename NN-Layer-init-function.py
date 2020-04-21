def __init__(self, n_units, activation="relu", cuda=True):
        self.n_units = n_units
        
        assert(activation in ("relu","sigmoid", "tanh", "leaky-relu")), "InitError: Invalid Activation Function"
        
        self.activationFunction = activation
        self.weights = None
        self.biases = None
        self.activations = None
        self.Z = None
        self.cuda = cuda