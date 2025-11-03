# Predicting Chaotic Systems with Deep Learning  
**Investigating LSTM Robustness to Noise without Teacher Forcing**

**Author:** Matej Vučković  
**Mentor:** Prof. Goran Kvaščev  
**Faculty:** University of Belgrade – School of Electrical Engineering  
**Date:** September 2025  

---

## 🧠 Overview  

This repository contains the source code and experimental results for the bachelor’s thesis  
**“Predicting Chaotic Systems with Deep Learning: Investigating LSTM Robustness to Noise without Teacher Forcing.”**

The project studies how **Long Short-Term Memory (LSTM)** neural networks can learn and predict **chaotic dynamical systems** such as the **Lorenz**, **Clifford**, and **Thinkerbell** attractors.  
It focuses on **noise robustness** and training **without teacher forcing (no-TF)** — an approach that improves long-term sequence prediction stability in chaotic systems.

---

## ⚙️ Key Features  

- Synthetic dataset generation for multiple chaotic systems  
- Configurable white Gaussian noise (via SNR in dB)  
- LSTM training with and without teacher forcing  
- Automated hyperparameter tuning using grid search  
- Evaluation with both statistical and chaos-specific metrics:  
  - MSE, R²  
  - Lyapunov exponents and Lyapunov time  
- Visualization of trajectories, convergence curves, and prediction results  
- Full implementation in **PyTorch**

---

## 📁 Repository Structure  
├── attractors_catalog.ipynb # Interactive attractor generation notebook

├── attractors_catalog.py # Python module for generating chaotic systems

├── dataset_generator.py # Synthetic dataset creation

├── dataset_awgn.py # Adds Gaussian noise to datasets

├── metrics.py # Metrics: MSE, R², Lyapunov exponents

├── LSTM_model.py # LSTM network definition (no-TF)

├── LSTM_gridSearch.py # Hyperparameter search

├── LSTM_train.py # Training pipeline

├── LSTM_evaluate.py # Evaluation and visualization

├── LLE_calculator.py # Lyapunov exponent estimator

├── /models # Trained model checkpoints (.pth)

├── /data # Generated datasets (.npz)

├── /results # Output plots and metrics

└── README.md # Project documentation

---

## 🧩 Methodology  

1. **Data Generation**  
   - Dynamical systems (Lorenz, Clifford, Thinkerbell) are simulated using first-order Taylor integration.  
   - Generated trajectories represent chaotic time series.

2. **Noise Injection**  
   - White Gaussian noise is added to datasets with configurable **Signal-to-Noise Ratio (SNR)** in dB.

3. **Model Architecture**  
   - Single LSTM layer with stacked cells  
   - 24 hidden units per cell  
   - Window size (lags): 4  
   - Prediction horizon: 100 timesteps  
   - ≈17k trainable parameters

4. **Training Setup**  
   - Optimizer: Adam with weight decay  
   - Learning rate: 0.001 with exponential decay (γ = 0.98)  
   - Epochs: 180  
   - Batch size: 48  
   - Early stopping and checkpoint saving enabled

5. **Evaluation Metrics**  
   - **MSE** and **R²** for accuracy  
   - **Lyapunov exponents** and **Lyapunov time** for chaos preservation  
   - Comparison of prediction stability under different noise levels

---

## 📊 Results Summary  

| SNR (dB) | R² (approx.) | Chaotic Structure Preserved | Notes |
|-----------|---------------|-----------------------------|-------|
| ∞ (ideal) | 0.996 | ✅ Yes | Perfect reconstruction |
| 50 | 0.995 | ✅ Yes | Stable predictions |
| 40 | 0.992 | ✅ Yes | Slight deviation |
| 30 | 0.976 | ⚠️ Partial | Reduced divergence |
| 20 | 0.94  | ❌ No | Lost chaotic behavior |
| 10–0 | <0.9 | ❌ No | Model fails to generalize |

LSTM models successfully predict chaotic dynamics for **several Lyapunov times**, maintaining both **trajectory accuracy** and **global attractor geometry** under moderate noise levels.

---

## 🚀 Usage  

### 1. Generate Dataset  
```bash
python dataset_generator.py --attractor lorenz --steps 50000 --radius 1.0
python dataset_awgn.py --input data/lorenz.npz --snr 30
```
### 2. Train Model
```bash
python LSTM_train.py --data data/lorenz_30dB.npz --epochs 180
```
### 3. Evaluate and Visualize
```bash
python LSTM_evaluate.py --model models/lorenz_lstm.pth
```

## 🧮 Dependencies

Python ≥ 3.10

PyTorch ≥ 2.0

NumPy

Matplotlib

SciPy

Install all requirement
```bash
pip install -r requirements.txt
```
## 📈 Citation

If you use this work, please cite:

Matej Vučković (2025).
Predicting Chaotic Systems with Deep Learning: Investigating LSTM Robustness to Noise without Teacher Forcing.
University of Belgrade, School of Electrical Engineering.
