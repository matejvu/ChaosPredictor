# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:23:24 2025

@author: matej
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import matplotlib.pyplot as plt

from LSTM_model import LSTMnoTF
import metrics as mtr

LLE = 0.9056
tL = 1/LLE
dt = 0.01

#=========================================
#   ARGS
#       path - path to npz file
#       d - number of lags
#       t - number of predicted steps
#   RETURN
#       loaders - tuple of DataLoaders
#       datasets - tuple of Datasets
#       D - dimension of data

#=========================================
def TensorTestDS(path, d, t, batch_size, particles = 1, data_len = 1000):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    if(D!=3):
        print('Only 3D flows supported currently!')
        return
		    
    X = data['X'][-1][0:data_len]
    Y = data['Y'][-1][0:data_len]
    Z = data['Z'][-1][0:data_len]

    timeseries = np.stack([X, Y, Z], axis=-1)

    def make_windows(series, d, t):
        data_in = []
        data_out = []

        for i in range(len(series) - d - t):
            window = series[i:i+d]
            output = series[i+d:i+d+t]
            
            data_in.append(window)
            data_out.append(output)

        return np.array(data_in), np.array(data_out)

    # build windows separately for train / val / test
    test_in, test_out = make_windows(timeseries, d, t)

    # Convert to PyTorch tensors
    test_in_t = torch.FloatTensor(test_in)
    test_out_t = torch.FloatTensor(test_out)
    
    

    #Datasets
    test_dataset = TensorDataset(test_in_t, test_out_t)
    
    #DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    return  test_loader, test_dataset, D

def one_test_run(model, loader):

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:

            predictions = model(data, future_steps=target.shape[1])
            loss = nn.functional.mse_loss(predictions, target)
            test_loss += loss.item()
            
            all_predictions.append(predictions.numpy())
            all_targets.append(target.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r2_score = mtr.R2(all_predictions, all_targets)
    mse = np.mean((all_predictions - all_targets) ** 2)

    return mse, r2_score

def load_model(model_path, isize,hsize, osize, nlayers, timesteps):
    model = LSTMnoTF(isize, hsize, osize, timesteps, nlayers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def test_prediction(model, data_path, bach_size = 16, timesteps = 100, verbose = False):

    model.eval()

    results = []

    if verbose:
        print(model)
        for i in range(timesteps // 50):

            loader, dataset, D = TensorTestDS(data_path, 3, 50*(i+1), bach_size, 1, 10000)
            mse, r2 = one_test_run(model, loader)
            print(f"{50*(i+1)}->")
            results.append([mse, r2])

        print('\n-------------------------------------------')
        for i, res in enumerate(results):
            print(f'timesteps: {(i+1)*50}\tMSE: {res[0]}\tR2:{res[1]}')

        # Plot
        time = np.linspace(50, timesteps, timesteps//50)
        time = time*dt/tL
        plt.figure()
        plt.plot(time, [r[1] for r in results], linestyle=':', marker='o', color='red')
        plt.ylim(0,1)
        plt.ylabel('$\mathbf{R}^2$')
        plt.xlabel('$t_{pred}$ [$t_L$]')
        plt.grid()
        plt.show()

    else:
        loader, dataset, D = TensorTestDS(data_path, 3, timesteps, bach_size, 1, 10000)
        mse, r2 = one_test_run(model, loader)
        print(f"Test R2 Score: {r2}")
        print(f"Test MSE: {mse}")
    

def test_structure(model):
    
    model.eval()

if __name__ == "__main__":
    model = load_model(
                        model_path = 'best_model.pth',
                        isize = 3,
                        hsize = 24,
                        osize = 3,
                        nlayers = 4, 
                        timesteps = 600)


    test_prediction(    
                        model = model,
                        data_path = 'datasets_npz/lorenz_dataset.npz',
                        isize = 3,
                        hsize = 24,
                        osize = 3,
                        nlayers = 4,
                        bach_size = 1024,
                        timesteps = 600,
                        verbose = True
                    )
    