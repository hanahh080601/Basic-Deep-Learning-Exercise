from config import *
from FCL import FCL

# Multi-layers Neural Network
class MultiLayersNeuralNetwork:
    """
    Multi-layers Neural Network
    ---------------------
    Arguments
        input_nodes (int):
            Number of input nodes = Number of features in dataset.
        hidden_layers (int):
            Number of hidden layers.
        hidden_nodes (list):
            List of number of nodes in `hidden_layers` hidden layers.
        output_nodes (int):
            Number of output layers (number of classes).
        learning_rate (float):
            Learning rate.
        init (str):
            Weight initialization techniques ('xavier', 'he', 'none - random')
        use_bias (bool):
            Add bias or not.
        activation_mode (str):
            Activation function names ('relu', 'tanh', 'sigmoid')
    """
    def __init__(self, input_nodes, hidden_layers, hidden_nodes, output_nodes, learning_rate, init = 'xavier', use_bias=False, activate_mode='tanh'):
        
        # Set class variables
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.activation_mode = activate_mode
        
        # Define activation function
        # ReLU activation function
        if self.activation_mode == 'relu':
            self.activation_function = lambda x : x * (x > 0)
            self.activation_devr = lambda x : (x > 0)
        
        # Sigmoid activation function
        elif self.activation_mode == 'sigmoid':
            self.activation_function = lambda x:  1 / (1 + np.exp(-x))
            self.activation_devr = lambda x: x * (1 - x)
        
        # Tanh activation function
        else: 
            self.activation_function = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))    
            self.activation_devr = lambda x: 1 - x ** 2

        # Initialize network with 2 + `hidden layers` layers.
        if hidden_layers == 1:
            self.layers = [FCL(input_nodes, hidden_nodes[0], self.activation_function, init, use_bias)] + \
            [FCL(hidden_nodes[0], output_nodes, None, init, use_bias)]
        else:
            self.layers = [FCL(input_nodes, hidden_nodes[0], self.activation_function, init, use_bias)] + \
                [FCL(hidden_nodes[i], hidden_nodes[i+1], self.activation_function, init, use_bias) for i in range(hidden_layers-1)] + \
                [FCL(hidden_nodes[-1], output_nodes, None, init, use_bias)]
            
    def _forward(self, X):
        """
        Feed forward
        --------------
        Parameters:
            X (array):
                Array of input features.
        """
        input_x = X
        for layer in self.layers:
            output = layer.forward(input_x)
            input_x = output
        return output
            
    def _backprop(self, output, target):
        """
        Backpropagation
        ---------------
        Parameters:
            output (array):
                Predicted values.
            target (array):
                Ground truth (y_test)
        """
        error = target - output
        for i, layer in enumerate(reversed(self.layers)):
            if i < len(self.layers) - 1:
                layer_error = layer.backprop(error, self.activation_devr)
                error = layer_error
                assert error is not None
            else:
                # Skip error calculation for backprop on first layer
                # since it's connected to inputs
                assert layer.backprop(error, self.activation_devr, first_layer=True) is None
                
    def _initialize_deltas(self):
        for layer in self.layers:
            layer.initialize_delta()
                
    def _update_weights(self, n_samples):
        for layer in self.layers:
            layer.update_weights(self.lr, n_samples)

    def train(self, features, targets):
        """
        Train the network on batch of features and targets. 
        
        Parameters
        ---------
            features (array) 
                Numpy 2D array, each row is one data record, each column is a feature (X_train)
            targets (array) 
                Numpy 1D array of target values (y_train)
        """
        if len(features.shape) != 2:
            raise ValueError('Features dimensions do not match. Correct ' +\
                              'shape should be (n_samples, n_samples).')
        if len(targets.shape) != 1:
            raise ValueError('Targets dimensions do not match. Correct '  +\
                              'shape should be (n_samples,).')
        if features.shape[0] != targets.shape[0]:
            raise ValueError('Features and targets have different amount of samples.')
        
        features = features[:, None]
        n_samples = features.shape[0]
        # Reset delta for every layer before training with new batch
        self._initialize_deltas()
            
        # Start training
        for X, y in zip(features, targets):
            # Forward pass
            output = self._forward(X)          
            # Backpropagation
            self._backprop(output, y)      
        # Update weights after training with batch
        self._update_weights(n_samples)
            
    def predict(self, features):
        """
        Make predictions based on input features.
        Parameters
        ---------
            features (array) 
                Numpy 2D array, each row is one data record, each column is a feature (X_test)
        """
        if len(features.shape) != 2:
            raise ValueError('Features dimensions do not match. Correct ' +\
                              'shape should be (n_samples, n_samples).')
        if features.shape[1] != self.input_nodes:
            raise ValueError('Number of input features do not match training ' +\
                             'number of features (' + str(features.shape[1]) +\
                             ' != ' + str(self.input_nodes) + ')')
                            
        return self._forward(features)

    def get_batches(self, X, y, batch_size):
        """
        Split dataset into batches for training, validating.
        ---------------
        Parameters:
            X (array):
                Array of features (X_train / X_val).
            y (array):
                Array of targets (y_train / y_val).
            batch_size (int):
                Number of samples in 1 batch.
        """
        for batch in range(0, len(X), batch_size):
            yield X[batch:batch+batch_size], y[batch:batch+batch_size]

    def MSE(self, labels, predicted):
        """
        Mean square error
        """
        return np.mean((labels - predicted) ** 2)

    def accuracy(self, labels, predicted):
        return (labels == predicted.round()).sum() * 100 / labels.shape[0]

    