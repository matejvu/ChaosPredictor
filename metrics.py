# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 23:16:29 2025

@author: matej
"""
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def R2(predictions, timeseries):
    mean = np.mean(timeseries)
    mean = np.full_like(timeseries, mean)
   
    meanMSE = np.sum((mean - timeseries)**2)
    modelMSE = np.sum((predictions - timeseries)**2)
    
    return 1 - modelMSE / meanMSE
    
    