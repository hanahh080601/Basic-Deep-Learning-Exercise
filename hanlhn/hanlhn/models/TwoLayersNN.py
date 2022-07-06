from config import *

class TwoLayerNeuralNetwork(object):
    """
    A two-layer feedforward neural network.
    -----------------------
    Arguments: 
        data_matrix (np.array):
            Training features (X_train) 
        target_variable (np.array) 
            Training targets (y_train)
        hidden (int):
            Number of nodes in hidden layer.
        iteration_limit (int):
            Number of iterations. 
        learning_rate (float):
            Learning rate
        init (str):
            Weight nitialization ('xavier', 'he', 'none (random)')
    """
    def __init__(self, data_matrix = None, target_variable = None, hidden = None,
                 iteration_limit = None, learning_rate = None, init='xavier'):

        # Create design matrix
        self.N = data_matrix.shape[0]  
        design_matrix = pd.DataFrame(data_matrix)
        design_matrix.insert(0, 'bias', np.ones(self.N))  
        self.X = design_matrix

        num_inputs = data_matrix.shape[1]  
        self.T = pd.get_dummies(target_variable) 
        num_ouputs = len(self.T.columns)  # number of classes
        self.Y = target_variable  # target variable
        self.iteration = iteration_limit
        self.lr = learning_rate
        
        if init == 'xavier':
            std_W1 = np.sqrt(2/(num_inputs + hidden + 1))
            std_W2 = np.sqrt(2/(num_ouputs + hidden + 1))

            self.W_1 = pd.DataFrame(std_W1 * (np.random.rand((num_inputs + 1), hidden) - 0.5))
            self.W_2 = pd.DataFrame(std_W2 * (np.random.rand((hidden + 1), num_ouputs) - 0.5))

        elif init == 'he':
            std_W1 = np.sqrt(1/(num_inputs + 1))
            std_W2 = np.sqrt(1/(hidden + 1))

            self.W_1 = pd.DataFrame(np.random.normal(0, std_W1, size=(num_inputs + 1, hidden)))
            self.W_2 = pd.DataFrame(np.random.normal(0, std_W2, size=(hidden+1, num_ouputs)))

        else:
            self.W_1 = pd.DataFrame(0.01 * np.random.random(((num_inputs + 1), hidden)))
            self.W_2 = pd.DataFrame(0.01 * np.random.random(((hidden + 1), num_ouputs)))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def forward_propogation(self):
        self.A_1 = np.dot(self.X, self.W_1)
        self.H = self.tanh(self.A_1)
        self.Z_1 = np.insert(self.H, 0, 1, axis=1)  # add column of 1's for bias
        self.A_2 = np.dot(self.Z_1, self.W_2)
        self.Yhat = self.softmax(self.A_2)
        return self

    def cross_entropy_loss(self):
        '''Cross entropy cost function '''
        eps = 1e-10
        yhat = np.clip(self.Yhat, eps, 1-eps)
        return -np.nansum(self.T * np.log(self.Yhat))
    """
    def accuracy(self):
        # check the prediction accuracy for a current iteration
        return sum(pd.DataFrame(self.Yhat).idxmax(axis=1) == self.Y) / self.N
    """

    def accuracy(self, pred, truth):
        return sum(pd.DataFrame(pred).idxmax(axis=1) == truth) / len(truth)

    def backprop(self):
        # calculate gradient for W_2
        self.delta_k = self.Yhat - self.T  # a notation borrowed from Bishop's text (Yhat - T)
        self.W_2_gradient = np.dot(np.transpose(self.Z_1), self.delta_k)
       
        # calculate gradient for W_1
        self.H_prime = 1 - self.H**2  # derivative of tanh
        self.W_2_reduced = self.W_2.iloc[1:,]  # drop first row from W_2
        self.W_1_gradient = np.dot(np.transpose(self.X),
                                   (self.H_prime * np.dot(self.delta_k,
                                                          np.transpose(self.W_2_reduced))))

        # update weights with gradient descent
        self.W_1 = self.W_1 - self.lr * self.W_1_gradient 
        self.W_2 = self.W_2 - self.lr * self.W_2_gradient
        return self

    def train(self):
        current_loss = 1000

        for i in range(int(self.iteration)):
            self.forward_propogation()
            self.backprop()
            loss = self.cross_entropy_loss()

            if current_loss > loss:
                current_loss = loss
                self.best_W1 = self.W_1
                self.best_W2 = self.W_2

            if (i % 100) == 0:
                print('Iteration: ', i,
                      'Loss: ', round(self.cross_entropy_loss(), 4),
                      'Accuracy: ', round(self.accuracy(self.Yhat, self.Y), 4))
                     
    def predict(self, X_test):

        self.N_test = X_test.shape[0]  # assumes df is np array
        X_test = pd.DataFrame(X_test)
        X_test.insert(0, 'bias', np.ones(self.N_test))  # add column of 1's
        self.X_test = X_test

        self.A_1 = np.dot(self.X_test, self.best_W1)
        self.H = self.tanh(self.A_1)
        self.Z_1 = np.insert(self.H, 0, 1, axis=1)  # add column of 1's for bias
        self.A_2 = np.dot(self.Z_1, self.best_W2)
        self.Yhat = self.softmax(self.A_2)
        return self

    