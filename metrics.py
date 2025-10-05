# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 23:16:29 2025

@author: matej
"""

import numpy as np

def R2(predictions, timeseries):
    mean = np.mean(timeseries)
    mean = np.full_like(timeseries, mean)
   
    meanMSE = np.sum((mean - timeseries)**2)
    modelMSE = np.sum((predictions - timeseries)**2)
    
    return 1 - modelMSE / meanMSE
    
    