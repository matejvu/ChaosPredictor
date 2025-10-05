# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:34:24 2025

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

def plot_training_curves(train_losses, val_losses, title = 'Training and Validation Loss'):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.yscale('log')  # Useful if loss values vary widely
    plt.tight_layout()
    plt.show()


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
def TDatasetFromSeries(path, d, t, batch_size, particles = 1, data_len = 1000):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    X = data['X'][0][0:data_len]
    Y = data['Y'][0][0:data_len]
    Z = data['Z'][0][0:data_len]
    
    timeseries = np.stack([X, Y, Z], axis=-1)
    
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    
    data_in = []
    data_out = []

    for i in range(len(X) - d - t):
        window = timeseries[i:i+d]
        output = timeseries[i+d:i+d+t]
        
        data_in.append(window)
        data_out.append(output)


    data_in = np.array(data_in)    
    data_out = np.array(data_out)    
    
    # train_d = np.expand_dims(train_d, axis=-1)
    # train_out = np.expand_dims(train_out, axis=-1)
    
    # Convert to PyTorch tensors 
    data_in = torch.FloatTensor(data_in)
    data_out = torch.FloatTensor(data_out)
    
    # Split data to Train Val Test - 70/15/15
    train_size = int(0.7 * len(data_in))
    val_size = int(0.85 * len(data_in))
    
    #Datasets
    train_dataset = TensorDataset(data_in[:train_size], data_out[:train_size])
    val_dataset = TensorDataset(data_in[train_size:val_size], data_out[train_size:val_size])
    test_dataset = TensorDataset(data_in[val_size:], data_out[val_size:])
    
    #DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset), D
    

#=========================================
#   ARGS
#       model - LSTM model
#       train_loader - 
#       val_loader - 
#       epochs - 
#       learning_rate - 
#   RETURN
#       train_losses
#       val_losses
#=========================================
def train_lstm_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, gamma = 0.95):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Train Batches
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
        
            # Forward pass
            predictions = model(data, future_steps=target.shape[1])
            
            # Calculate loss
            loss = criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        #current_lr = scheduler.get_last_lr()[0]
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                predictions = model(data, future_steps=target.shape[1])
                loss = criterion(predictions, target)
                val_loss += loss.item()
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses


def train(path, d, t, batch_size, hidden_size, epochs, lr, gamma, nlayers, key = '', verbose = False):
    
    #load data
    (TrainLD, ValLD, TestLD),(TrainDS, ValDS, TestDS), D = TDatasetFromSeries(path, d, t, batch_size, data_len=10000)
    
    # Initialize model
    input_size = output_size = D    
    model = LSTMnoTF(input_size, hidden_size, output_size, t)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    
    print('======================TRAINING==================================================')
    print(f"Model initialized:")
    print(f"Hidden size: {hidden_size}")
    print(f"Total parameters: {total_params}")
    print(f"Input DS shape: {TrainDS.tensors[0].shape}")
    print(f"Output DS shape: {TrainDS.tensors[1].shape}")
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses = train_lstm_model(
        model, TrainLD, ValLD, epochs=epochs, learning_rate=lr, gamma = gamma
    )
    
    
    #PLot
    if verbose:
        t = 'Traning Loss ['+key+']'
        plot_training_curves(train_losses, val_losses, t)
    
    # Test the trained model
    print("\nTesting trained model...")
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in TestLD:
            predictions = model(data, future_steps=target.shape[1])
            loss = nn.functional.mse_loss(predictions, target)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(TestLD)
    print(f'Avg. test loss: {avg_test_loss}')
    return avg_test_loss


