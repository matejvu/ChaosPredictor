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
import metrics as mtr

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
    #plt.show()
    plt.savefig("./training_plots/"+title+".png")


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

    Xv = data['X'][1][0:data_len]
    Yv = data['Y'][1][0:data_len]
    Zv = data['Z'][1][0:data_len]

    Xt = data['X'][2][0:data_len]
    Yt = data['Y'][2][0:data_len]
    Zt = data['Z'][2][0:data_len]

    timeseries = np.stack([X, Y, Z], axis=-1)
    timeseriesV = np.stack([Xv, Yv, Zv], axis=-1)
    timeseriesT = np.stack([Xt, Yt, Zt], axis=-1)

    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    


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
    train_in, train_out = make_windows(timeseries, d, t)
    val_in, val_out = make_windows(timeseriesV, d, t)
    test_in, test_out = make_windows(timeseriesT, d, t)

    # Convert to PyTorch tensors
    train_in_t = torch.FloatTensor(train_in)
    train_out_t = torch.FloatTensor(train_out)
    val_in_t = torch.FloatTensor(val_in)
    val_out_t = torch.FloatTensor(val_out)
    test_in_t = torch.FloatTensor(test_in)
    test_out_t = torch.FloatTensor(test_out)
    
    # Split data to Train Val Test - 70/15/15
    train_size = int(0.7 * len(train_in_t))
    val_size = test_size = int(0.15 * len(train_in_t))

    #Datasets
    train_dataset = TensorDataset(train_in_t[:train_size], train_out_t[:train_size])
    val_dataset = TensorDataset(val_in_t[:val_size], val_out_t[:val_size])
    test_dataset = TensorDataset(test_in_t[:test_size], test_out_t[:test_size])
    
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
#       file - file to write output
#       epochs - 
#       learning_rate - 
#       gamma - learning rate decay
#   RETURN
#       train_losses
#       val_losses
#=========================================
def train_lstm_model(model, train_loader, val_loader, file, epochs=100, learning_rate=0.001, gamma=0.95, early_stopping_patience=30):
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device('cuda')
    else:
        print("No GPU available. Training will run on CPU.")
        device = torch.device('cpu')
    model = model.to(device)    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    early_stopping = mtr.EarlyStopping(patience=early_stopping_patience, path='best_model.pth')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Train Batches
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
        
            # Forward pass
            predictions = model(data, future_steps=target.shape[1])
            
            # Calculate loss
            loss = criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        # python
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        scheduler.step()
        #current_lr = scheduler.get_last_lr()[0]
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
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
            file.flush()

        #Early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses


def train(path, d, t, batch_size, hidden_size, epochs, lr, gamma, nlayers, key = '', td = 1000, file = None, verbose = False):
    
    #load data
    (TrainLD, ValLD, TestLD),(TrainDS, ValDS, TestDS), D = TDatasetFromSeries(path, d, t, batch_size, data_len=td)
    
    # Initialize model
    input_size = output_size = D    
    model = LSTMnoTF(input_size, hidden_size, output_size, t, nlayers=nlayers)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    
    print('======================TRAINING==================================================')
    print(f"Total parameters: {total_params}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of stacked layers: {nlayers}")
    print(f"Learning rate: {lr}")
    print(f"Learning decay: {gamma}")
    print(f"Batch size: {batch_size}")
    print(f"Input train data shape: {TrainDS.tensors[0].shape}")
    print(f"Output train data shape: {TrainDS.tensors[1].shape}")

    if file:
        file.flush()

    # Train the model
    print("\nStarting training...")
    train_losses, val_losses = train_lstm_model(
        model, TrainLD, ValLD, file, epochs=epochs, learning_rate=lr, gamma = gamma
    )
    
    
    #Plot
    if verbose:
        plot_training_curves(train_losses, val_losses, f"Training Loss [{key}]")
    
    # Test the trained model
    print("\nTesting trained model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in TestLD:
            # --- Move batch to GPU ---
            data, target = data.to(device), target.to(device)

            predictions = model(data, future_steps=target.shape[1])
            loss = nn.functional.mse_loss(predictions, target)
            test_loss += loss.item()
            
            # Move back to CPU for metrics and concatenation
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics (on CPU / NumPy)
    r2_score = mtr.R2(all_predictions, all_targets)
    mse = np.mean((all_predictions - all_targets) ** 2)

    print(f"Test R2 Score: {r2_score}")
    print(f"Test MSE: {mse}")
    print(f"Shapes - Predictions: {all_predictions.shape}, Targets: {all_targets.shape}")

    avg_test_loss = test_loss / len(TestLD)
    print(f"Test avg. loss: {avg_test_loss}")

    return avg_test_loss, mse, r2_score

if __name__ ==  '__main__':
    path = "./datasets_npz/lorenz_dataset.npz"
    d = 4
    t = 100
    batch_size = 16
    td = 1000
    (TrainLD, ValLD, TestLD),(TrainDS, ValDS, TestDS), D = TDatasetFromSeries(path, d, t, batch_size, data_len=td)
    
