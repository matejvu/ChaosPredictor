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
    def __init__(self, input_size, hidden_size, output_size, max_future_steps, nlayers = 1, stand_scales=None):
        super().__init__()
        self.max_future_steps = max_future_steps
        self.stand_scales = stand_scales

        #input_size - broj feature-ova
        #output_size - broj feature-ova
        #hidden_size - dubina memorije
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=nlayers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, future_steps=None):
        if future_steps is None:
            future_steps = self.max_future_steps
        
        # print(x.shape)
        #Process lag frame through lstm
        _, (hidden, cell) = self.lstm(x)
        
        # print(hidden.shape)

        #initial prediction(t+1)
        #hidden vraca sve lejere u poslednjem vremenskom koraku a uzet je samo njavisi lejer
        last_output = hidden[-1]  # shape: (batch, hidden_size)
        # print(last_output.shape)

        #da uvek bude u istom formatu, kao output
        prediction_next = self.fc(last_output.unsqueeze(1))
        # print(f'prediction next {prediction_next.shape}') 
        prediction_next = prediction_next
        # print(f'prediction next unsq {prediction_next.shape}')
        predictions = []
        
        for t in range(future_steps):
            predictions.append(prediction_next)
            lstm_out, (hidden, cell) = self.lstm(prediction_next, (hidden, cell))
            # print(f'lstm out {lstm_out.shape}')
            # print(f'hidden {hidden.shape}')

            #detach zbog gpu memmory leak
            hidden = hidden.detach()
            cell = cell.detach()


            prediction_next = self.fc(lstm_out)
            # print(f'prediction next {prediction_next.shape}')
            
        
        return torch.cat(predictions, dim=1)
    
if __name__ == "__main__":
    #Test
    model = LSTMnoTF(input_size=3, hidden_size=6, output_size=3, max_future_steps=10, nlayers=2)
    x = torch.randn(16, 4, 3)  # batch_size=16, lag=4, features=3
    
    model.forward(x)

    # out = model(x)
    # print(out.shape)  # Expected shape: (16, 10, 3)
