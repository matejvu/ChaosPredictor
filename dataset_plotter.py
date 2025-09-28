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

    
    print(f"Loaded dataset from {path}")
    print(f"Dimension: {D}, dt={dt}, init_conditions={init.shape}")

    if D == 3:
        fig = go.Figure()
        for p in range(particles):
            xs = data["X"][p][:time]
            ys = data["Y"][p][:time]
            zs = data["Z"][p][:time]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(
                    color=palette[p % len(palette)],
                    width=1.5
                ),
                name=f"Particle {p+1}"
            ))
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            width=1024, height=1024
        )
    else:
        fig = go.Figure()
        for p in range(particles):
            xs = data["X"][p][:time]
            ys = data["Y"][p][:time]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(
                    color=palette[p % len(palette)],
                    width=1.5
                ),
                name=f"Particle {p+1}"
            ))
        fig.update_layout(
            width=700, height=700,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        )

    fig.show()

if __name__ == "__main__":
    plot_dataset("./datasets_npz_awng/lorenz_dataset_00dB.npz",time=500, particles=3)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_40dB.npz", particles=2)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_30dB.npz", particles=4)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_20dB.npz", particles=3)
    # plot_dataset("./datasets_npz_awng/lorenz_dataset_10dB.npz", particles=2)
