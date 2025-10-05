# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:33:08 2025

@author: matej
"""

from LSTM_train import train
import numpy as np


#=======HYPERPARAMETERS=======

lags = {4, 8, 14}
batches = {16, 48, 128}
num_layers = {1,2,4}
learning_rates = { 0.1, 0.01, 0.001}
decays = {0, 0.001}

#=========PARAMETERS==========

lag = 4
h = 100
batch_size = 16
hidden_size = 12
epochs = 20
lr = 0.001
gamma = 1
num_layer = 1
total_data = 10000

#==============================
if __name__ == "__main__":
    path = "./datasets_npz/lorenz_dataset.npz"
    losses = {}
    key=''
    for lag in lags: 
        for batch_size in batches:
            
            key = 'lag'+str(lag)+'bch'+str(batch_size)
            # key = 'lr'+str(lr)+'gam'+str(gamma)
            
            loss, mse, r2 = train(
                            path = path,
                            d = lag,
                            t = h,
                            batch_size = batch_size,
                            hidden_size = hidden_size,
                            epochs = epochs,
                            lr = lr,
                            gamma = gamma,
                            nlayers = num_layer,
                            td = total_data,
                            verbose = True,
                            key = key
                            )
            
            losses[key] = [loss, mse, r2]
    
    print('======================RESULTS===================================================')
    print('\n Key\t\t| Avg. Loss\t| MSE\t\t| R² score')
    print('=================================================')
    for k, v in losses.items():
        print(f' {k}\t| {v[0]:.4f}\t| {v[1]:.4f}\t| {v[2]:.4f}')