# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 04:08:01 2025

@author: matej
"""

import rosenstein as lr
import numpy as np

limit = 10000

if __name__ == "__main__":
    traceJ = float(input('Enter valid analytical value of trace(J) for your system:'))
    
    data = np.load("./datasets_npz/lorenz_dataset.npz")
    D = data["dimension"]
    x = data["X"][0][0:limit]
    y = data["Y"][0][0:limit]
    z = data["Z"][0][0:limit]

    l1=[]
    
    
    ts = x  
    np.save("tmp.npy", ts)  
    lle = lr.main(
        ts_path="tmp.npy",
        lag=11,
        emb_dim=9,
        t_0=80,
        t_f=150,
        delta_t=0.01,
        method="welch",
        show=False,
    )
    l1.append(lle)
    
    ts = y 
    np.save("tmp.npy", ts)  
    lle = lr.main(
        ts_path="tmp.npy",
        lag=11,
        emb_dim=9,
        t_0=80,
        t_f=150,
        delta_t=0.01,
        method="welch",
        show=False,
    )
    l1.append(lle)
    
    if D==3:
        ts = z  
        np.save("tmp.npy", ts)  
        lle = lr.main(
            ts_path="tmp.npy",
            lag=11,
            emb_dim=9,
            t_0=80,
            t_f=150,
            delta_t=0.01,
            method="welch",
            show=False,
        )
        l1.append(lle)
        
    l1 = sum(l1)/len(l1)
    l2 = 0.0
    l3 = traceJ - l1 - l2
    
    print(f'Lyapunov Spectrum from data: \n\tlambda1 = {l1} \n\tlambda2 = {l2} \n\tlambda3 = {l3}')
    print(f'LLE is averaged from {D} dimension data')
    