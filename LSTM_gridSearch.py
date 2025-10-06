# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:33:08 2025

@author: matej
"""

from LSTM_train import train
import numpy as np
import time

#=======HYPERPARAMETERS=======

lags = {4, 8}
batches = {16, 32, 64}
num_layers = {1,2,4}
hidden_sizes = {6, 12}
learning_rates = { 0.1, 0.01, 0.001}
decays = {0, 0.001}

#=========PARAMETERS==========

lag = 4
h = 100
batch_size = 16
hidden_size = 12
epochs = 150
lr = 0.001
gamma = 1
num_layer = 1
total_data = 30000

#==============================
if __name__ == "__main__":
    path = "./datasets_npz/lorenz_dataset.npz"
    losses = {}
    key=''
    time_start = time.time()

    for lag in lags: 
        for num_layer in num_layers:
            for hidden_size in hidden_sizes:
            
                key = 'lag'+str(lag)+'nl'+str(num_layer)+'hs'+str(hidden_size)
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
    
    time_end = time.time()
    elapsed = round(time_end - time_start)

    print('======================RESULTS===================================================')
    print(f"Total elapsed time: {elapsed//3600} : {elapsed%3600//60} : {elapsed%60} (h:m:s)")
    print('\n Key\t\t| Avg. Loss\t| MSE\t\t| R² score')
    print('=====================================================')
    for k, v in losses.items():
        print(f' {k}\t| {v[0]:.4f}\t| {v[1]:.4f}\t| {v[2]:.4f}')