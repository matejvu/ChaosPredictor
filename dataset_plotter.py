# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 22:20:49 2025

@author: matej
"""

# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
palette = px.colors.qualitative.Dark24


def plot_dataset(path, particles=1, time=-1):
    data = np.load(path)
    dt = data["dt"]
    init = data["init"]
    D = data["dimension"]

    if particles>init.shape[0]:
        print(f"There is less than {particles} particles in dataset")
        return
    
    print(f"Loaded dataset from {path}")
    print(f"Dimension: {D}, dt={dt}, init_conditions={init.shape}")

    fig = go.Figure()
    for p in range(particles):
        
        line_dict = dict(
            color=palette[p % len(palette)],
            width=1.5
        )
        marker=dict(
            size=0.5,
            color=np.linspace(0, 1, len(data["X"][0][:time])),
            colorscale='Turbo'
        )
        
        if D == 3:
            #Background choice
            # axis_style = dict(showbackground=False, showgrid=False, zeroline=False)
            axis_style = dict(showbackground=True, showgrid=True, zeroline=True)
            
            xs = data["X"][p][:time]
            ys = data["Y"][p][:time]
            zs = data["Z"][p][:time]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=line_dict,
                name=f"Particle {p+1}"
            ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title="X", **axis_style),
                    yaxis=dict(title="Y", **axis_style),
                    zaxis=dict(title="Z", **axis_style),
                ),
                width=1024, height=1024
            )
        elif D == 2:
            xs = data["X"][p][:time]
            ys = data["Y"][p][:time]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=marker,
                name=f"Particle {p+1}"
            ))
            fig.update_layout(
                width=700, height=700,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )

    fig.show()

if __name__ == "__main__":
    # plot_dataset("./datasets_npz/clifford_dataset.npz", particles=1, time = 50000 )
    plot_dataset("./datasets_npz_awng/lorenz_dataset_50dB.npz", particles=3, time=3000)
    plot_dataset("./datasets_npz_awng/lorenz_dataset_40dB.npz", particles=3, time=3000)
    plot_dataset("./datasets_npz_awng/lorenz_dataset_30dB.npz", particles=3, time=3000)
    plot_dataset("./datasets_npz_awng/lorenz_dataset_20dB.npz", particles=3, time=3000)
    plot_dataset("./datasets_npz_awng/lorenz_dataset_10dB.npz", particles=3, time=3000)
    plot_dataset("./datasets_npz_awng/lorenz_dataset_0dB.npz", particles=3, time=3000)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_0dB.npz", particles=1)
    # plot_dataset("./datasets_npz_awng/sprott_dataset_40dB.npz", particles=1)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_20dB.npz", particles=3)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_10dB.npz", particles=2)
