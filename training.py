import torch
import torch.nn as nn
import torch.optim as optim
import data
from hyperparameters import hps
import helpers
import random

# inputs are letters, output is category (english, french, or spanish)
model = nn.LSTM(hps['onehot_length'], 3, hps['lstm_layers'], dropout=hps['dropout'])
dataset = data.get_pairs('datasets/training_set.csv')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hps['learning_rate'])

batch_size = hps['batch_size']

for t in range(hps['epochs']):
    epoch_losses = []
    # train on as many pairs are in the dataset
    for _ in range(len(dataset) // batch_size):
        batch = random.sample(dataset, batch_size)
        # keep losses from each batch to be averaged later
        batch_losses = []
        for pair in batch:
            xs, y = helpers.pair_to_xy(pair)
            # make a tensor for the whole sequence
            xs = torch.stack(xs)
            y_pred, _ = model(xs)
            # we only want the final prediction of the sequence
            y_pred = y_pred[-1]
            seq_loss = criterion(y_pred, y)
            batch_losses.append(seq_loss)
        # get the mean of all losses from the batch of names
        batch_loss = torch.mean(torch.stack(batch_losses))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(batch_loss)
    epoch_loss = torch.mean(torch.stack(epoch_losses)).item()

    # print loss at intervals
    if (t+1) % hps['print_every'] == 0 or t == 0:
        print(t+1, epoch_loss)

    # save model at intervals
    if (t+1) % hps['save_every'] == 0:
        torch.save(model.state_dict(), 'models/model.pt')
