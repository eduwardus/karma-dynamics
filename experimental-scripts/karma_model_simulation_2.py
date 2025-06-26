# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:26:36 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== CONFIGURATION ====================
# Simulation settings
N = 1000       # number of agents
T = 100.0      # total simulation time
dt = 0.01      # time step

# Model parameters
kappa       = 0.50
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
lambda_psi  = 0.05

# Initial conditions (randomized for diversity among agents)
np.random.seed(42)
init_vals = {
    'm':   np.random.uniform(0.1, 0.3, N),
    'w':   np.random.uniform(0.2, 0.4, N),
    'A':   np.random.uniform(0.05, 0.15, N),
    'V':   np.random.uniform(0.02, 0.10, N),
    'I':   np.random.uniform(0.08, 0.12, N),
    'P':   np.random.uniform(0.03, 0.07, N),
    'E':   np.random.uniform(0.03, 0.07, N),
    'Psi': np.zeros(N)
}
# =======================================================

steps = int(T / dt)
time = np.linspace(0, T, steps)

# State arrays
m = np.zeros((steps, N))
w = np.zeros((steps, N))
A = np.zeros((steps, N))
V = np.zeros((steps, N))
I = np.zeros((steps, N))
P = np.zeros((steps, N))
E = np.zeros((steps, N))
Psi = np.zeros((steps, N))

# Initialize
for var, val in init_vals.items():
    globals()[var][0, :] = val

# Helper: logistic function
def S(m_val):
    return 1 / (1 + np.exp(-0.5 * (m_val - 5)))

# Euler-Maruyama simulation
for t in range(1, steps):
    m_prev, w_prev = m[t-1], w[t-1]
    A_prev, V_prev = A[t-1], V[t-1]
    I_prev, P_prev = I[t-1], P[t-1]
    E_prev, Psi_prev = E[t-1], Psi[t-1]

    # Deterministic components
    dm = kappa * A_prev - gamma_m * m_prev * (1 + np.tanh(w_prev)) - epsilon * V_prev * m_prev
    dw = eta_M - mu * w_prev - lambda_w * I_prev * w_prev
    drift_A = alpha_A * A_prev + beta_AP * A_prev * np.tanh(m_prev) - gamma_A * w_prev * A_prev
    drift_V = alpha_V * V_prev + beta_VE * V_prev * E_prev - gamma_V * w_prev * V_prev + theta * S(m_prev)
    drift_I = alpha_I * I_prev + beta_IP * np.tanh(I_prev * P_prev) - gamma_I * w_prev * I_prev
    drift_P = alpha_P * P_prev - gamma_P * w_prev * P_prev
    drift_E = alpha_E * E_prev + beta_EV * (Psi_prev * E_prev) / (1 + Psi_prev) - gamma_E * w_prev * E_prev
    dPsi = -lambda_psi * Psi_prev + V_prev

    # Stochastic updates (Euler-Maruyama)
    m[t] = m_prev + dm * dt
    w[t] = w_prev + dw * dt
    A[t] = A_prev + drift_A * dt + sigma_A * A_prev * np.sqrt(dt) * np.random.randn(N)
    V[t] = V_prev + drift_V * dt + sigma_V * V_prev * np.sqrt(dt) * np.random.randn(N)
    I[t] = I_prev + drift_I * dt + sigma_I * I_prev * np.sqrt(dt) * np.random.randn(N)
    P[t] = P_prev + drift_P * dt
    E[t] = E_prev + drift_E * dt
    Psi[t] = Psi_prev + dPsi * dt

# ==================== PLOTTING ====================
# Plot one agent
agent_idx = 0
plt.figure(figsize=(10,5))
plt.plot(time, m[:,agent_idx], label='Merit (m)')
plt.plot(time, w[:,agent_idx], label='Wisdom (w)')
plt.legend(); plt.title('Merit and Wisdom - Agent 0'); plt.xlabel('Time'); plt.ylabel('Value')
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
plt.plot(time, A[:,agent_idx], label='Altruism (A)')
plt.plot(time, V[:,agent_idx], label='Vulnerability (V)')
plt.plot(time, I[:,agent_idx], label='Influence (I)')
plt.legend(); plt.title('A, V, I - Agent 0'); plt.xlabel('Time'); plt.ylabel('Value')
plt.tight_layout(); plt.show()

# Plot population averages
plt.figure(figsize=(10,5))
plt.plot(time, m.mean(axis=1), label='Mean Merit')
plt.plot(time, w.mean(axis=1), label='Mean Wisdom')
plt.legend(); plt.title('Mean Merit and Wisdom (All Agents)'); plt.xlabel('Time'); plt.ylabel('Value')
plt.tight_layout(); plt.show()

# Final time histograms
plt.figure(figsize=(12,6))
for i, (arr, label) in enumerate(zip([m, w, A, V, I, P, E, Psi],
                                     ['m','w','A','V','I','P','E','Psi'])):
    plt.subplot(2, 4, i+1)
    sns.histplot(arr[-1], bins=30, kde=True, color='skyblue')
    plt.title(label)
plt.tight_layout(); plt.show()

# Correlation heatmap
import pandas as pd
final_df = pd.DataFrame({
    'm': m[-1], 'w': w[-1], 'A': A[-1], 'V': V[-1],
    'I': I[-1], 'P': P[-1], 'E': E[-1], 'Psi': Psi[-1]
})
plt.figure(figsize=(8,6))
sns.heatmap(final_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix at Final Time")
plt.tight_layout(); plt.show()
