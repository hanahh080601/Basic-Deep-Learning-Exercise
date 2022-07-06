from data.DataPipeline import DataPipeline
from models.TwoLayersNN import TwoLayerNeuralNetwork
from models.MultiLayersNN.MultiLayersNN import MultiLayersNeuralNetwork

if __name__ == "__main__":

    # Read, preprocess and split data.
    data = DataPipeline()
    X_train_val, y_train_val = data(mode='train', scale_mode='standard')
    X_test, y_test = data(mode='test', scale_mode='standard')

    mode = int(input('Two layers neural network: 1, Multi layers neural network: 2 --> '))
    if mode == 1:
        # Run two-layers neural network
        ann = TwoLayerNeuralNetwork(data_matrix=X_train_val, target_variable=y_train_val,
                        hidden=8, iteration_limit=2e3, learning_rate=0.0001, init='none')  # create ANN
        ann.train()
        ann.execute(X_test, y_test)

    else:
        # Run multi-layers neural network
        ann = MultiLayersNeuralNetwork(X_train_val.shape[1], hidden_layers=1, hidden_nodes=[8], output_nodes=1,
                learning_rate=0.001, use_bias=True)
        best_weight = ann.execute(X_train_val, y_train_val, X_test, y_test, loss='hinge', epochs=1000, batch_size=16)
    





