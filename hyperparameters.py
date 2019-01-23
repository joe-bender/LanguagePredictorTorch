"""Hyperparameters to be used by the application"""

hps = {
    'learning_rate': .01, # for the Adam optimizer
    'dropout': .2,
    'lstm_layers': 3, # how many LSTM modules to stack in the network
    'batch_size': 64, # how many names to train on at once
    'epochs': 5000000000, # number of training epochs
    'print_every': 1, # print the loss every n epochs
    'softmax_tuning': 3, # how much to exaggerate softmax probabilities
    'filename': 'names2017.csv', # where to pull names from
    'onehot_length': 27, # number of categories (all lowercase letters and _)
    'save_every': 1, # how often to save the model
}
