# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:37:49 2025

@author: matej
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

#model of LSTM with no Teacher Forcing

class LSTMnoTF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_future_steps, nlayers = 1):
        super().__init__()
        self.max_future_steps = max_future_steps
        
        #input_size - broj feature-ova
        #output_size - broj feature-ova
        #hidden_size - dubina memorije
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=nlayers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, future_steps=None, teacher_forcing_ratio=0.5):
        if future_steps is None:
            future_steps = self.max_future_steps
        
        #Process lag frame through lstm
        _, (hidden, cell) = self.lstm(x)
        
        #initial prediction(t+1)
        last_output = hidden[-1]  # shape: (batch, hidden_size)
        prediction_next = self.fc(last_output)
        prediction_next = prediction_next.unsqueeze(1)
        
        predictions = []
        
        for t in range(future_steps):
            predictions.append(prediction_next)
            lstm_out, (hidden, cell) = self.lstm(prediction_next, (hidden, cell))
            prediction_next = self.fc(lstm_out)
            
        
        return torch.cat(predictions, dim=1)
    
for i in range(20):
    model = LSTMnoTF(3, i+1, 3, 20)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{i}]Total number of parameters: {total_params}")