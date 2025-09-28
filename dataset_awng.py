# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 22:08:52 2025

@author: matej
"""

import numpy as np
import h5py
import os

# Global SNR dictionary (in dB)
SNR_LEVELS = {
    "0dB" : 0,
    "10dB": 10,
    "20dB": 20,
    "30dB": 30
}

def add_awgn(signal, snr_db):
    """
    Add AWGN to a signal for a given SNR (in dB).
    """
    # Signal power
    power_signal = np.mean(signal**2)
    
    # Target noise power from SNR
    snr_linear = 10**(snr_db / 10)
    power_noise = power_signal / snr_linear
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(power_noise), signal.shape)
    
    return signal + noise


def process_dataset(attractor, input_dir="./datasets_npz", output_dir="./datasets_npz_awng"):
    """
    Load dataset, add AWGN at multiple SNR levels, and save noisy versions.
    Works with .npz datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load original dataset (.npz format)
    file_path = os.path.join(input_dir, f"{attractor}_dataset.npz")
    data = np.load(file_path)

    xs_all, ys_all, zs_all = data["X"], data["Y"], data["Z"]
    dt, init_conditions, D = data["dt"], data["init"], data["dimension"]

    for label, snr_db in SNR_LEVELS.items():
        xs_noisy = add_awgn(xs_all, snr_db)
        ys_noisy = add_awgn(ys_all, snr_db)
        zs_noisy = add_awgn(zs_all, snr_db)

        out_path = os.path.join(output_dir, f"{attractor}_dataset_{label}.npz")
        np.savez(
            out_path,
            dt=dt,
            dimension=D,
            X=xs_noisy,
            Y=ys_noisy,
            Z=zs_noisy,
            init=init_conditions,
            snr=snr_db
        )
        print(f"Saved noisy dataset at {snr_db} dB SNR → {out_path}")


if __name__ == "__main__":
    
    process_dataset("lorenz")  
