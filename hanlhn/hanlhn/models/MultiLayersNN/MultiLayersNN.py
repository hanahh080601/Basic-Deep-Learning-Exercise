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

    def execute(self, X_train_val, y_train_val, X_test, y_test, loss='hinge', epochs=1000, batch_size=16):
        """
        Execution training, validating and testing.
        --------------------
        Parameters:
            X_train_val (array):
                Combine of training and validating features.
            y_train_val (array):
                Combine of training and validating targets.
            X_test (array):
                Testing features.
            y_test (array):
                Testing targets.
            loss (str):
                Loss function names ('hinge', 'log', 'mse'). Default: 'hinge'.
            epochs (int):
                Number of epochs. Default: 1000
            batch_size (int):
                Number of samples in one batch. Default: 16
        """
        if loss == 'hinge':
            loss = hinge_loss
        elif loss == 'log':
            loss = log_loss
        else:
            loss = self.MSE

        test_accuracies = []
        test_losses = []
        best_acc_train = 0
        best_acc_test = 0
        last_loss = 1000
        patience = 10
        trigger_times = 0
        isStopped = False
        w_best = None

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        for train_index, val_index in kf.split(X_train_val, y_train_val):
            X_train = X_train_val[train_index]
            y_train = y_train_val[train_index]

            X_val = X_train_val[val_index]
            y_val = y_train_val[val_index]
                    
            for e in range(epochs):
                train_accuracies = []
                train_losses = []
                val_accuracies = []
                val_losses = []

                for batch_x, batch_y in self.get_batches(X_train, y_train, batch_size):
                    self.train(batch_x, batch_y)
                    train_acc = self.accuracy(y_train[:, None], self.predict(X_train))
                    train_loss = loss(y_train, self.predict(X_train))
                    train_accuracies.append(train_acc)
                    train_losses.append(train_loss)

                for batch_x, batch_y in self.get_batches(X_val, y_val, batch_size):
                    val_acc = self.accuracy(y_val[:, None], self.predict(X_val))
                    val_loss = loss(y_val, self.predict(X_val))
                    val_accuracies.append(val_acc)
                    val_losses.append(val_loss)

                test_acc = self.accuracy(y_test[:, None], self.predict(X_test))
                test_loss = loss(y_test, self.predict(X_test))
                test_accuracies.append(test_acc)
                test_losses.append(test_loss)
                    
                if best_acc_train < train_acc:
                    best_acc_train = train_acc
                if best_acc_test < test_acc:
                    best_acc_test = test_acc
                    
                if e % 100 == 0:
                    print('Epoch', e, ':')
                    print('Training loss:', train_loss, 'Training accuracy:', train_acc)
                    print('Validating loss:', val_loss, 'Validating accuracy:', val_acc)
                    print('Test loss:', test_loss, 'Test accuracy:', test_acc)
                # Early stopping
                current_loss = val_loss

                if current_loss >= last_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        '''
                        print('Early stopping! at epoch ', e)
                        isStopped = True
                        break  
                        '''
                        self.lr = float(self.lr * 0.2)

                else:
                    trigger_times = 0
                    last_loss = current_loss
                    if not isStopped:
                        w_best = self.layers
                        #print('Validation loss {:.6f}.  Saving weights ...'.format(current_loss))

        print("Best accuracy on training set: ", best_acc_train)
        print("Best accuracy on testing set: ", best_acc_test)
        return w_best
    