# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:33:08 2025

@author: matej
"""

from LSTM_train import train
import numpy as np
import time
import sys

#=======HYPERPARAMETERS=======

lags = {4, 8}
batches = { 128}
num_layers = { 7}
hidden_sizes = { 24, 16}
learning_rates = { 0.01, 0.001}
decays = {0.01}

#=========PARAMETERS==========

lag = 4
h = 500
batch_size = 16
hidden_size = 28
epochs = 150
lr = 0.001
gamma = 0.98
num_layer = 5
total_data = 25000

#==============================
if __name__ == "__main__":
    path = "./datasets_npz/lorenz_dataset.npz"
    losses = {}
    key=''
    time_start = time.time()

    with open('./training_plots/output.txt', 'w') as f:
        sys.stdout = f

        # for lag in lags: 
        for decay in decays:
            gamma = 1 - decay
            for batch_size in batches:

                key = 'h600'+'y'+str(gamma)+'bch'+str(batch_size)
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
                                key = key,
                                file = f
                                )
                
                losses[key] = [loss, mse, r2]
                f.flush()

        time_end = time.time()
        elapsed = round(time_end - time_start)

        print('======================RESULTS===================================================')
        print(f"Total elapsed time: {elapsed//3600} : {elapsed%3600//60} : {elapsed%60} (h:m:s)")
        print('\n Key\t\t| Avg. Loss\t| MSE\t\t| R² score')
        print('=====================================================')
        for k, v in losses.items():
            print(f' {k}\t| {v[0]:.4f}\t| {v[1]:.4f}\t| {v[2]:.4f}')