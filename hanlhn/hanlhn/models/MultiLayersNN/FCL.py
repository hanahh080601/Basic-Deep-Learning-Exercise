from config import *

class FCL:
    """
    Fully connected layer
    ---------------------
    Arguments
        input_nodes (int):
            Number of input nodes = Number of features in dataset.
        hidden_nodes (int):
            Number of nodes in 1 hidden layer.
        activation_fn (str):
            Activation function names ('relu', 'tanh', 'sigmoid')
        init (str):
            Weight initialization techniques ('xavier', 'he', 'none - random')
        use_bias (bool):
            Add bias or not.
    """
    def __init__(self, input_nodes, hidden_nodes, activation_fn, init, use_bias):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.activation_function = activation_fn
        
        # Initialize weights
        if init == 'xavier':
            self.std_xavier = np.sqrt(2 / (self.input_nodes + self.hidden_nodes + 1))
            self.weights = np.random.random((self.input_nodes, self.hidden_nodes)) * self.std_xavier

        elif init == 'he':
            self.std_he = np.sqrt(1 / (self.hidden_nodes + 1))
            self.weights = np.random.normal(0.0, self.std_he,
                                    (self.input_nodes, self.hidden_nodes))
        else:
            self.weights = 0.01 * np.random.random((self.input_nodes, self.hidden_nodes))
        
        # Initialize biases
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = np.zeros((1, self.hidden_nodes))
            self.delta_bias = np.zeros(self.bias.shape)

        # Initialize delta weights                           
        self.delta_weights = np.zeros(self.weights.shape)
        
    def initialize_delta(self):
        self.delta_weights = np.zeros(self.weights.shape)
        if self.use_bias:
            self.delta_bias = np.zeros(self.bias.shape)
            
    def update_weights(self, lr, n_features):
        self.weights += lr * self.delta_weights / n_features
        if self.use_bias:
            self.bias += lr * self.delta_bias / n_features
                                        
    def forward(self, features):
        """
        Feed forward
        ------------------
        Parameters:
            features (array): 
                Array of features (X_train)
        """
        self.input = features
        self.output = np.dot(features, self.weights)
        if self.use_bias:
            self.output += self.bias
        if self.activation_function is not None:
            self.output = self.activation_function(self.output)
        return self.output
        
    def backprop(self, error, activation_devr, first_layer=False):
        """
        Backpropagation
        -------------------
        Parameters:
            error (float):
                Error of predicted value and ground truth.
            activation_devr (function):
                Deriviation of activation function.
            first_layer (bool):
                Is it the first layer or not.
        """
        # Calculate delta to update weights
        #print('Input shape:', self.input.T.shape, 'Error shape:', error.shape)
        self.delta_weights += np.dot(self.input.T, error)
        if self.use_bias:
            self.delta_bias += error
        # If it's not the first layer, calculate contribution to error
        # to pass to previous layer. Else save up computation time.
        if not first_layer:
            layer_error = np.dot(error, self.weights.T)
            # The derivative of the the activation function 
            layer_error *= activation_devr(self.input)
            return layer_error
        else:
            return None
