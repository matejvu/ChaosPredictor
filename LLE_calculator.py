# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 16:02:20 2025

@author: matej
"""
import numpy as np
import random as r
import attractors_catalog as ac
import matplotlib.pyplot as plt
import math as m

dR0 = 10**(-10)

functions2D = {"clifford" : ac.clifford2D,
               "thinkerbell" : ac.thinkerbell2D,
               "quadratic" : ac.quadratic2D}
functions3D = {"lorenz" : ac.lorenz3D,
               "sprott" : ac.sprott3D}




def find_LLE(path, time=100000):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    attractor = path.split('/')[-1].split('_')[0]
    dB = path.split('/')[-1].split('_')[-1].split('.')[0]
    
    print(f"Loaded dataset from {path}")
    print(f"Dimension: {D}, dt={dt}, init_conditions={init.shape}")

    #choose rand direction
    random_dir = np.random.randn(D)
    unit_vector = random_dir / np.linalg.norm(random_dir)
    dR = unit_vector * dR0
    
    #inital conditions
    R0 = np.array(init[0])
    R1 = R0 + dR
    
    # Store trajectories
    traj0, traj1 = [], []
    
    #lambda
    lambda1 = 0
    
    l = []
    
    for t in range(time):
        #evolve
        if D == 2:
            new_R0 = np.array(functions2D[attractor](*R0, 1, 0))
            new_R1 = np.array(functions2D[attractor](*R1, 1, 0))
        elif D == 3:
            new_R0 = np.array(functions3D[attractor](*R0, 1, 0, dt))
            new_R1 = np.array(functions3D[attractor](*R1, 1, 0, dt))
        
        #find new dif
        R0, R1 = new_R0, new_R1
        dR = R1 - R0
        
        #normalize distance
        R1 = R0 + dR * dR0/np.linalg.norm(dR) 
        
        #calculate lambda1
        lambda1 += m.log(np.linalg.norm(dR) /dR0)
        
        l.append(lambda1/(dt*(t+1)))
        
    #results
    print(f'Largest Lyapunov exponent: {lambda1/time/dt}')
    fig = plt.figure()
    t = [i for i in range(time)]
    plt.plot(t, l)
    plt.xlabel('t')
    plt.ylabel('$\lambda_1$')
    plt.title('Lyapunov Exponent Convergence')
    plt.savefig('lyapunov_plot_'+attractor+'_'+dB+'.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    
if __name__ == "__main__":
    #Warning: 2D map might not be supported
    
    find_LLE("./datasets_npz_awng/lorenz_dataset_0dB.npz", time=10**6 )
    find_LLE("./datasets_npz_awng/lorenz_dataset_10dB.npz", time=10**6 )
    find_LLE("./datasets_npz_awng/lorenz_dataset_20dB.npz", time=10**6 )
    find_LLE("./datasets_npz_awng/lorenz_dataset_30dB.npz", time=10**6 )
    find_LLE("./datasets_npz_awng/lorenz_dataset_40dB.npz", time=10**6 )
    find_LLE("./datasets_npz_awng/lorenz_dataset_50dB.npz", time=10**6 )
