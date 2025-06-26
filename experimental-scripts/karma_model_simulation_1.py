# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:31:54 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION
N = 1000       # number of agents
T = 100.0      # total simulation time
dt = 0.01      # time step

# Model parameters
kappa       = 0.10
gamma_m     = 0.03
epsilon     = 0.01
eta_M       = 0.20
mu          = 0.01
lambda_w    = 0.01
alpha_A     = 0.20
beta_AP     = 0.10
gamma_A     = 0.05
sigma_A     = 0.05
alpha_V     = 0.15
beta_VE     = 0.10
gamma_V     = 0.03
theta       = 0.02
sigma_V     = 0.05
alpha_I     = 0.15
beta_IP     = 0.10
delta       = 0.00
gamma_I     = 0.05
sigma_I     = 0.05
alpha_P     = 0.10
gamma_P     = 0.02
alpha_E     = 0.10
beta_EV     = 0.05
gamma_E     = 0.02
lambda_psi  = 0.02

# Initial conditions
init_means = {
    'm':   0.20,
    'w':   0.30,
    'A':   0.10,
    'V':   0.05,
    'I':   0.10,
    'P':   0.03,
    'E':   0.1,
    'Psi': 0.00
}

steps = int(T / dt)
time = np.linspace(0, T, steps)

# Allocate arrays
m = np.zeros((steps, N))
w = np.zeros((steps, N))
A = np.zeros((steps, N))
V = np.zeros((steps, N))
I = np.zeros((steps, N))
P = np.zeros((steps, N))
E = np.zeros((steps, N))
Psi = np.zeros((steps, N))

# Initialize with small random perturbations
for var, mean in init_means.items():
    globals()[var][0] = mean + 0.01 * np.random.randn(N)

# Helper: logistic activation
def S(m_val):
    return 1 / (1 + np.exp(-0.5 * (m_val - 5)))

# Euler-Maruyama simulation
for t in range(1, steps):
    m_prev, w_prev = m[t-1], w[t-1]
    A_prev, V_prev = A[t-1], V[t-1]
    I_prev, P_prev = I[t-1], P[t-1]
    E_prev, Psi_prev = E[t-1], Psi[t-1]

    # Deterministic terms
    dm = kappa * A_prev - gamma_m * m_prev * (1 + np.tanh(w_prev)) - epsilon * V_prev * m_prev
    dw = eta_M - mu * w_prev - lambda_w * I_prev * w_prev
    drift_A = alpha_A * A_prev + beta_AP * A_prev * np.tanh(m_prev) - gamma_A * w_prev * A_prev
    drift_V = alpha_V * V_prev + beta_VE * V_prev * E_prev - gamma_V * w_prev * V_prev + theta * S(m_prev)
    drift_I = alpha_I * I_prev + beta_IP * np.tanh(I_prev * P_prev) - gamma_I * w_prev * I_prev
    drift_P = alpha_P * P_prev - gamma_P * w_prev * P_prev
    drift_E = alpha_E * E_prev + beta_EV * (Psi_prev * E_prev) / (1 + Psi_prev) - gamma_E * w_prev * E_prev
    dPsi = -lambda_psi * Psi_prev + V_prev

    # Euler-Maruyama update
    m[t] = m_prev + dm * dt
    w[t] = w_prev + dw * dt
    A[t] = A_prev + drift_A * dt + sigma_A * A_prev * np.sqrt(dt) * np.random.randn(N)
    V[t] = V_prev + drift_V * dt + sigma_V * V_prev * np.sqrt(dt) * np.random.randn(N)
    I[t] = I_prev + drift_I * dt + sigma_I * I_prev * np.sqrt(dt) * np.random.randn(N)
    P[t] = P_prev + drift_P * dt
    E[t] = E_prev + drift_E * dt
    Psi[t] = Psi_prev + dPsi * dt

# Plot for a single agent and average
agent = 0
variables = {'m': m, 'w': w, 'A': A, 'V': V, 'I': I, 'P': P, 'E': E, 'Psi': Psi}

# Plot individual vs average
fig, axs = plt.subplots(4, 2, figsize=(12, 10))
axs = axs.flatten()
for i, (label, data) in enumerate(variables.items()):
    axs[i].plot(time, data[:, agent], label=f'{label} (Agent 0)', alpha=0.7)
    axs[i].plot(time, data.mean(axis=1), label=f'{label} (Average)', linestyle='--')
    axs[i].set_title(label); axs[i].legend()

plt.tight_layout()
plt.show()

# Histogram at final time step
plt.figure(figsize=(10, 6))
for label, data in variables.items():
    plt.hist(data[-1], bins=30, alpha=0.5, label=label, density=True)
plt.title('Distribution at Final Time Step')
plt.legend()
plt.tight_layout()
plt.show()

# Scatterplot matrix between selected variables
plt.figure(figsize=(12, 10))
plt.scatter(m[-1], w[-1], alpha=0.3, label='m vs w')
plt.scatter(A[-1], V[-1], alpha=0.3, label='A vs V')
plt.scatter(I[-1], P[-1], alpha=0.3, label='I vs P')
plt.scatter(E[-1], Psi[-1], alpha=0.3, label='E vs Psi')
plt.title('Scatterplots at Final Time Step')
plt.legend()
plt.tight_layout()
plt.show()
