# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:33:08 2025

@author: matej
"""

from LSTM_train import train
import numpy as np


#=======HYPERPARAMETERS=======

lags = {2, 4, 6}
batches = {16, 32, 64}
num_layers = {1,2,3}
learning_rates = { 0.1, 0.01, 0.001}
decays = {0, 0.001}

#=========PARAMETERS==========

d = 4
h = 10
bch = 16
hidden_size = 12
epochs = 20
lr = 0.001
gamma = 0.95
num_layer = 1

#==============================
if __name__ == "__main__":
    
    path = "./datasets_npz/lorenz_dataset.npz"
    losses = {}

    for lag in lags: 
        for batch_size in batches:
            
            key = 'lag'+str(lag)+'bch'+str(batch_size)
            
            avg_loss = train(
                            path = path,
                            d = lag,
                            t = h,
                            batch_size = batch_size,
                            hidden_size = hidden_size,
                            epochs = epochs,
                            lr = lr,
                            gamma = gamma,
                            nlayers = num_layer,
                            verbose = True,
                            key = key
                            )
            
            losses[key] = avg_loss
    
    print('======================RESULTS===================================================')
    for k, v in losses.items():
        print(f' {k}\t| {v}')