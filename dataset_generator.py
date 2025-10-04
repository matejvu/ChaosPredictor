# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 15:53:30 2025

@author: matej
"""

import h5py
import numpy as np
import attractors_catalog as ac
import random as r

functions2D = {"clifford" : ac.clifford2D,
               "thinkerbell" : ac.thinkerbell2D,
               "quadratic" : ac.quadratic2D}
functions3D = {"lorenz" : ac.lorenz3D,
               "sprott" : ac.sprott3D}

def generate_dataset(attractor, D = 2, particles = 10, steps = 100000, discard = 1000, radius = 1.0, dt = 0.01):
    
    xs_all = np.empty((particles, steps - discard))
    ys_all = np.empty((particles, steps - discard))
    zs_all = np.empty((particles, steps - discard))

    init_conditions = np.empty((particles, D))
    
    for i in range(particles):
        x = r.uniform(-radius, radius)
        y = r.uniform(-radius, radius)
        z = r.uniform(-radius, radius)
    
        if D==2:
            init_conditions[i] = [x, y]
            xs, ys = functions2D[attractor](x, y, steps, discard)
            xs_all[i] = xs
            ys_all[i] = ys
        elif D==3:
            init_conditions[i] = [x, y, z]
            xs, ys, zs = functions3D[attractor](x, y, z, steps, discard, dt)
            xs_all[i] = xs
            ys_all[i] = ys
            zs_all[i] = zs
        else: 
            print("Invalid dimension D = "+str(D))
            return
                    
    with h5py.File("./datasets_h5/"+attractor+"_dataset.h5", "w") as f:
        f.create_dataset("x", data=xs_all)
        f.create_dataset("y", data=ys_all)
        f.create_dataset("z", data=zs_all)
        f.create_dataset("init_conditions", data=init_conditions)
        f.attrs["system"] = attractor
        f.attrs["dimension"] = D
        f.attrs["dt"] = dt

    np.savez(
        "./datasets_npz/"+attractor+"_dataset.npz",
        dt=dt,
        dimension=D,            
        X=xs_all,
        Y=ys_all,
        Z=zs_all,
        init=init_conditions
    )
    print(f"Saved dataset at ./datasets_npz/"+attractor+"_dataset.npz")

    return 

if __name__ == "__main__":
    generate_dataset("sprott", 3, 1, radius = 2.0)

    
    


