# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:34:24 2025

@author: matej
"""

#Hyperparameters
import numpy as np
import torch
from LSTM_model import LSTMnoTF

path = "./datasets_npz_awng/lorenz_dataset_50dB.npz"
data = np.load(path)
dt = data["dt"]
init = data["init"]
D = data["dimension"]


lags = {2, 4, 6}
batch = {16, 32, 64}
num_layers = {1,2,3}



#=========================================
#       path - path to npz file
#       d - number of lags
#       t - number of predicted steps
#=========================================
def TDatasetFromSeries(path, d, t,particles = 1, data_len = 1000):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    X = data['X'][0][0:data_len]
    Y = data['Y'][0][0:data_len]
    Z = data['Z'][0][0:data_len]
    timeseries = np.stack([X, Y, Z], axis=-1)
    print(X[3], Y[3], Z[3])
    print(timeseries[3])
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    
    train_d = []
    train_out = []

    for i in range(len(X) - d - t):
        window = X[i:i+d]
        output = X[i+d:i+d+t]
        
        train_d.append(window)
        train_out.append(output)


    train_d = np.array(train_d)    
    train_out = np.array(train_out)    
    
    train_d = np.expand_dims(train_d, axis=-1)
    train_out = np.expand_dims(train_out, axis=-1)
    # train_out = np.expand_dims(train_out, axis=-1)
    # print(train_d.shape)
    
    # Convert to PyTorch tensors FIRST
    train_d_tensor = torch.FloatTensor(train_d)
    train_out_tensor = torch.FloatTensor(train_out)
    
    # THEN split and create datasets
    train_size = int(0.8 * len(train_d_tensor))
    
    train_dataset = TensorDataset(train_d_tensor[:train_size], train_out_tensor[:train_size])
    val_dataset = TensorDataset(train_d_tensor[train_size:], train_out_tensor[train_size:])
    
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Training function
def train_lstm_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(data, future_steps=target.shape[1])
            
            # print(predictions.shape)
            # print(f'Target:{target.shape}')
            # Calculate loss
            loss = criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
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

# Main training execution
if __name__ == "__main__":
    
    input_size = 1
    output_size = 1
    max_future_steps = 10
    hidden_size = 12  # You can test different sizes
    batch_size = 4
    epochs = 10
    
    #make time frames
    
    
    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = LSTMnoTF(input_size, hidden_size, output_size, max_future_steps)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized:")
    print(f"Hidden size: {hidden_size}")
    print(f"Total parameters: {total_params}")
    print(f"Input shape: {train_d.shape}")
    print(f"Target shape: {train_out.shape}")
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses = train_lstm_model(
        model, train_loader, val_loader, epochs=epochs, learning_rate=0.001
    )
    
    # Test the trained model
    print("\nTesting trained model...")
    model.eval()
    
    print(train_dataset[:5][0].shape)
    
    with torch.no_grad():
        test_sample = train_dataset[:5][0]  # Test on first 5 samples
        predictions = model(test_sample, future_steps=max_future_steps)
        
        print(f"Sample input shape: {test_sample.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Target shape: {train_out[:5].shape}")
        
        # Calculate test loss
        target_tensor = torch.FloatTensor(train_out[:5])
        test_loss = nn.functional.mse_loss(predictions, target_tensor)
        print(f"Test loss on sample: {test_loss:.6f}")

