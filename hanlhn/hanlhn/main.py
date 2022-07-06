from data.DataPipeline import DataPipeline
from models.TwoLayersNN import TwoLayerNeuralNetwork

if __name__ == "__main__":

    # Read, preprocess and split data.
    data = DataPipeline()
    X_train_val, y_train_val = data(mode='train', scale_mode='standard')
    X_test, y_test = data(mode='test', scale_mode='standard')

    # Run two-layers neural network
    ann = TwoLayerNeuralNetwork(data_matrix=X_train_val, target_variable=y_train_val,
                        hidden=8, iteration_limit=2e3, learning_rate=0.0001, init='none')  # create ANN
    ann.train()
    ann.execute(X_test, y_test)





