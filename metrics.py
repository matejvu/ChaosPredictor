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
    
def Standardize(data, mean, std):
    return (data - mean) / std

def Destandardize(data, mean, std):
    return data * std + mean

def Normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def Denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def denormalize_3d_array(array, norm_scales):
    for i, el in enumerate(array):
        for j, (min_val, max_val) in enumerate(norm_scales):
            array[i, j] = Denormalize(array[i, j], min_val, max_val)

    return array


def load_norm_scales_from_string(s):
    """
    Parses a string representation of normalization scales into a list of tuples.

    Args:
        s (str): String representation of normalization scales, e.g.,
                 "((min_x, max_x), (min_y, max_y), (min_z, max_z))"

    Returns:
        list of tuples: Parsed normalization scales, e.g.,
                        [(-21.364112300269195, 22.20895754015633),
                         (-28.24657733906283, 29.750518195812635),
                         (3.376750699146575, 56.256212935739285)]
    """
    # Remove outer parentheses and split into individual tuples
    s = s.strip("()")
    scales = s.split("), (")
    
    norm_scales = []
    for scale in scales:
        # Remove any remaining parentheses and split into min/max
        min_max = scale.replace("(", "").replace(")", "").split(", ")
        norm_scales.append((float(min_max[0].strip('np.float64')), float(min_max[1].strip('np.float64'))))
    
    return norm_scales

