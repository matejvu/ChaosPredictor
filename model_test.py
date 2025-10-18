# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:23:24 2025

@author: matej
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import math as m
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
palette = px.colors.qualitative.Dark24

from LSTM_model import LSTMnoTF
from LLE_from_data import calcL1 
import metrics as mtr

LLE = 0.9056
tL = 1/LLE
dt = 0.01

#=========================================
#   ARGS
#       path - path to npz file
#       d - number of lags
#       t - number of predicted steps
#   RETURN
#       loaders - tuple of DataLoaders
#       datasets - tuple of Datasets
#       D - dimension of data

#=====================================================
def TensorTestDS(path, d, t, batch_size, particles = 1, data_len = 1000, norm_scales = None):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    if(D!=3):
        print('Only 3D flows supported currently!')
        return
		    
    X = data['X'][-1][0:data_len]
    Y = data['Y'][-1][0:data_len]
    Z = data['Z'][-1][0:data_len]

    if(norm_scales is not None):
        #Normalization
        min_x, min_y, min_z = norm_scales[0][0], norm_scales[1][0], norm_scales[2][0]
        max_x, max_y, max_z = norm_scales[0][1], norm_scales[1][1], norm_scales[2][1]
        X = mtr.Normalize(X, min_x, max_x)
        Y = mtr.Normalize(Y, min_y, max_y)
        Z = mtr.Normalize(Z, min_z, max_z)

    timeseries = np.stack([X, Y, Z], axis=-1)

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
    test_in, test_out = make_windows(timeseries, d, t)

    # Convert to PyTorch tensors
    test_in_t = torch.FloatTensor(test_in)
    test_out_t = torch.FloatTensor(test_out)
    
    

    #Datasets
    test_dataset = TensorDataset(test_in_t, test_out_t)
    
    #DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    return  test_loader, test_dataset, D

def one_test_run(model, loader):

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:

            predictions = model(data, future_steps=target.shape[1])
            loss = nn.functional.mse_loss(predictions, target)
            test_loss += loss.item()
            
            all_predictions.append(predictions.numpy())
            all_targets.append(target.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r2_score = mtr.R2(all_predictions, all_targets)
    mse = np.mean((all_predictions - all_targets) ** 2)

    return mse, r2_score

def load_model(model_path, isize,hsize, osize, nlayers, timesteps):
    model = LSTMnoTF(isize, hsize, osize, timesteps, nlayers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    return model

#Testiranjepreciznosti predikcije modela za vremenske intervale od 0 do timestes, u koracima od 50 ako je verbose = True
def test_prediction(model, data_path, bach_size = 16, timesteps = 100, verbose = False, norm_scales = None):

    model.eval()

    results = []

    if verbose:
        print(model)
        for i in range(timesteps // 50):

            loader, dataset, D = TensorTestDS(data_path, 3, 50*(i+1), bach_size, 1, 10000, norm_scales=norm_scales)
            mse, r2 = one_test_run(model, loader)
            print(f"{50*(i+1)}->")
            results.append([mse, r2])

        print('\n-------------------------------------------')
        for i, res in enumerate(results):
            print(f'timesteps: {(i+1)*50}\tMSE: {res[0]}\tR2:{res[1]}')

        # Plot
        time = np.linspace(50, timesteps, timesteps//50)
        time = time*dt/tL
        plt.figure()
        plt.plot(time, [r[1] for r in results], linestyle=':', marker='o', color='red')
        plt.ylim(0,1)
        plt.ylabel('$\mathbf{R}^2$')
        plt.xlabel('$t_{pred}$ [$t_L$]')
        plt.grid()
        plt.show()

    else:
        loader, dataset, D = TensorTestDS(data_path, 3, timesteps, bach_size, 1, 10000, norm_scales=norm_scales)
        mse, r2 = one_test_run(model, loader)
        print(f"Test R2 Score: {r2}")
        print(f"Test MSE: {mse}")
    
#====================================================

def evolve_model(model, x_t):
    print('x_t',x_t)
    x_t = torch.FloatTensor(x_t)
    x_t = x_t.unsqueeze(0).unsqueeze(0)
    
    prediction= model(x_t, future_steps=2)
    return prediction[0][0]
#Model mora da bude podesen na timesteps jednak dataset size

def numericalLLE(D, init, time, dt):
    np.set_printoptions(precision=16)
    torch.set_printoptions(precision=16, sci_mode=False)
    dR0 =  10**(-5)
    #choose rand direction
    random_dir = np.random.randn(D)
    unit_vector = random_dir / np.linalg.norm(random_dir)
    dR = unit_vector * dR0
    print(dR)
    dR = np.array(dR)
    
    #inital conditions
    R0 = np.array(init)
    R1 = R0 + dR
    print(R0,R1)
    
    # Store trajectories
    traj0, traj1 = [], []
    
    #lambda
    lambda1 = 0
    
    l = []
    
    model.eval()
    with torch.no_grad():
    
        for t in range(time):
            #evolve
            new_R0 = np.array(evolve_model(model, R0))
            new_R1 = np.array(evolve_model(model, R1))
            print(new_R0)
            print(new_R1)
            #find new dif
            R0, R1 = new_R0, new_R1
            dR = R1 - R0
            print(dR)
            #normalize distance
            R1 = R0 + dR * dR0/np.linalg.norm(dR) 
            #calculate lambda1
            lambda1 += m.log(np.linalg.norm(dR) /dR0)
            
            l.append(lambda1/(dt*(t+1)))
    return lambda1/time/dt
    
def test_structure(model, data_path, literature_LLE, dataset_size=10000):
    
    all_predictions = []
    loader, dataset, D = TensorTestDS(data_path, d=4, t=dataset_size, batch_size=1, particles=1, data_len=1+4+dataset_size )
    model.eval()
    
    # with torch.no_grad():
    #     for data, target in loader:
            
    #         predictions = model(data, future_steps=target.shape[1])
    #         all_predictions.append(predictions.numpy())
    
    # pred = all_predictions[0]
    # pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred

    # Extract the x, y, z arrays
    pred_np = []
    data = np.load(data_path)
    x_t = [ data['X'][-1][0], data['Y'][-1][0],data['Z'][-1][0] ]
    
    l1 = numericalLLE(3, x_t, 10, 0.01)
    
    # model.eval()
    # with torch.no_grad():
    #     for _ in range(dataset_size):
    #         x_t = evolve_model(model, x_t)
    #         pred_np.append(x_t)
    
    # x = pred_np[0, :, 0]  # shape (10000,)
    # y = pred_np[0, :, 1]  # shape (10000,)
    # z = pred_np[0, :, 2]  # shape (10000,)
    
    # fig = go.Figure()
    # line_dict = dict(
    #     color='red',
    #     width=1.5
    # )
    # # axis_style = dict(showbackground=False, showgrid=False, zeroline=False)
    # axis_style = dict(showbackground=True, showgrid=True, zeroline=True)
    

    # fig.add_trace(go.Scatter3d(
    #     x=x, y=y, z=z,
    #     mode="lines",
    #     line=line_dict,
    #     name=f"Predicted trrajectory"
    # ))
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="X", **axis_style),
    #         yaxis=dict(title="Y", **axis_style),
    #         zaxis=dict(title="Z", **axis_style),
    #     ),
    #     width=1024, height=1024
    # )

    # fig.show()
    # l1 = calcL1(x, y, z, 2)
    print(f'Largest Lyapunov Exponent from model generated data:\t {l1}')
    print(f'Largest Lyapunov Exponent from literature:\t\t {literature_LLE}')
    #all_predictions = np.concatenate(all_predictions, axis=0)

#====================================================


#VIZUALIZATIONS
def visualize_trajectory(model, data_path, d, h, particles=1, norm_scales = None):
    data = np.load(data_path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    if(D!=3):
        print('Only 3D flows supported currently!')
        return
            
    X = data['X'][-1][0:d+h]
    Y = data['Y'][-1][0:d+h]
    Z = data['Z'][-1][0:d+h]

    if(norm_scales is not None):
        #Normalization
        min_x, min_y, min_z = norm_scales[0][0], norm_scales[1][0], norm_scales[2][0]
        max_x, max_y, max_z = norm_scales[0][1], norm_scales[1][1], norm_scales[2][1]
        X = mtr.Normalize(X, min_x, max_x)
        Y = mtr.Normalize(Y, min_y, max_y)
        Z = mtr.Normalize(Z, min_z, max_z)

    
    timeseries = np.stack([X, Y, Z], axis=-1)
    init = timeseries[0:d]
    target = timeseries[d:d+h]
    pred_np = np.array([])

    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(init).unsqueeze(0), future_steps=h)
        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred

    to_plot = [target, pred_np[0]]
    print(to_plot[0].shape, to_plot[1].shape)
    labels = ['Target', 'Predicted']
    color_scale = ['greens', 'oranges']

    fig = go.Figure()
    for i, data in enumerate(to_plot):
        time_index = np.linspace(0, 1, len(data))

        fig.add_trace(go.Scatter3d(
            x=data[:, 0], y=data[:, 1], z=data[:, 2],
            mode='lines',
            name=labels[i],
            line=dict(
                color=time_index,  # Map time index to color
                colorscale=color_scale[i],  # Choose a color scale (e.g., Viridis, Plasma, etc.)
                width=4
            )
        ))

    fig.show()

def visualize_axis(model, data_path, d, h, particles=1, norm_scales = None):
    data = np.load(data_path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]
    
    if(particles!=1):
        print('Multiple particles not yet supported!')
        return
    if(D!=3):
        print('Only 3D flows supported currently!')
        return
            
    X = data['X'][-1][0:d+h]
    Y = data['Y'][-1][0:d+h]
    Z = data['Z'][-1][0:d+h]
    
    if(norm_scales is not None):
        #Normalization
        min_x, min_y, min_z = norm_scales[0][0], norm_scales[1][0], norm_scales[2][0]
        max_x, max_y, max_z = norm_scales[0][1], norm_scales[1][1], norm_scales[2][1]
        X = mtr.Normalize(X, min_x, max_x)
        Y = mtr.Normalize(Y, min_y, max_y)
        Z = mtr.Normalize(Z, min_z, max_z)
    
    timeseries = np.stack([X, Y, Z], axis=-1)
    init = timeseries[0:d]
    target = timeseries[d:d+h]
    pred_np = np.array([])

    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(init).unsqueeze(0), future_steps=h)
        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    
    mtr.denormalize_3d_array(pred_np[0], norm_scales)
    mtr.denormalize_3d_array(target, norm_scales)

    to_plot = [target, pred_np[0]]
    print(to_plot[0].shape, to_plot[1].shape)
    labels = ['Target', 'Predicted']
    color_scale = ['greens', 'oranges']

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    time = np.linspace(0, h*dt/tL, h)

    for i, axis in enumerate(['X', 'Y', 'Z']):
        for j, data in enumerate(to_plot):
            axs[i].plot(time, data[:, i], label=labels[j])
        
        axs[i].set_title(f'{axis} over Time')
        axs[i].set_xlabel('Time [$t_L$]')
        axs[i].set_ylabel(f'{axis} Value')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    norm_scales = mtr.load_norm_scales_from_string("((np.float64(-21.364112300269195), np.float64(22.20895754015633)), (np.float64(-28.24657733906283), np.float64(29.750518195812635)), (np.float64(3.376750699146575), np.float64(56.256212935739285)))")
    
    model = load_model(
                        model_path = 'best_model_minmax.pth',
                        isize = 3,
                        hsize = 24,
                        osize = 3,
                        nlayers = 4, 
                        timesteps = 100)


    visualize_trajectory(model, data_path = 'datasets_npz/lorenz_dataset.npz',
                          d = 4, h = 100, norm_scales=norm_scales)
    visualize_axis(model, data_path = 'datasets_npz/lorenz_dataset.npz',
                          d = 4, h = 700, norm_scales=norm_scales)

    # test_prediction(    
    #                     model = model,
    #                     data_path = 'datasets_npz/lorenz_dataset.npz',
    #                     # isize = 3,
    #                     # hsize = 24,
    #                     # osize = 3,
    #                     # nlayers = 4,
    #                     bach_size = 1024,
    #                     timesteps = 300,
    #                     verbose = True,
    #                     norm_scales = norm_scales
    #                 )

    # test_structure(
    #     model = model,
    #     data_path = 'datasets_npz/lorenz_dataset.npz',
    #     literature_LLE = 0.9056,
    #     dataset_size=100
    # )
    