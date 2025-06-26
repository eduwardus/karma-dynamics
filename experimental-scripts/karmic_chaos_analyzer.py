# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:29:40 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

"""
Programa: KarmicChaosAnalyzer_FixedStep
Descripción: Integra el sistema kármico con método de paso fijo (RK4),
             calcula divergencia exponencial, estima exponente de Lyapunov rápido
             y dimensión de correlación.
"""
# Configuración
warnings.filterwarnings('ignore')

# Parámetros del sistema
params = (0.1, 0.1, 0.1,  # alpha_I, alpha_A, alpha_V
          0.5, 0.6, 0.7,  # beta_IA, beta_AV, beta_VI
          0.3, 0.2)      # gamma, w

# Condiciones iniciales y discretización
y0 = np.array([0.4, 0.3, 0.2])
T = 200.0
N_pts = 80000  # número de pasos para integrador fijo (ajusta libremente)
assert N_pts > 38000, "N_pts debe ser mayor que 100 para un ajuste fiable"

dt = T / (N_pts - 1)

# Función del sistema
def karmic_vector(y, alpha_I, alpha_A, alpha_V,
                  beta_IA, beta_AV, beta_VI, gamma, w):
    I, A, V = y
    return np.array([
        alpha_I*I   + beta_IA*A*V - gamma*w*I,
        alpha_A*A   + beta_AV*V*I - gamma*w*A,
        alpha_V*V   + beta_VI*I*A - gamma*w*V
    ])

# Integrador RK4 de paso fijo
def integrate_rk4(y0, params, dt, N_steps):
    sol = np.zeros((N_steps, 3))
    y = y0.copy()
    sol[0] = y
    for i in range(1, N_steps):
        k1 = karmic_vector(y, *params)
        k2 = karmic_vector(y + 0.5*dt*k1, *params)
        k3 = karmic_vector(y + 0.5*dt*k2, *params)
        k4 = karmic_vector(y + dt*k3, *params)
        y = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        sol[i] = y
    return sol

# 1) Integración de las dos trayectorias
t = np.linspace(0, T, N_pts)
sol1 = integrate_rk4(y0, params, dt, N_pts)
# Perturbación determinista
delta = 1e-8
y0p = y0 + np.array([delta, 0.0, 0.0])
sol2 = integrate_rk4(y0p, params, dt, N_pts)

# 2) Divergencia y exponente de Lyapunov rápido
dist = np.linalg.norm(sol1 - sol2, axis=1)
ln_dist = np.log(dist)
# Filtrar valores finitos
t_mask = t[np.isfinite(ln_dist)]
ld_mask = ln_dist[np.isfinite(ln_dist)]
# Ventana dinámica para ajuste (10% de la simulación o mínimo 100 pts)
n_fit = max(100, int(0.1 * len(ld_mask)))
x_fit = t_mask[:n_fit]
y_fit = ld_mask[:n_fit]
# Ajuste lineal
def compute_lambda(x, y):
    coef = np.polyfit(x, y, 1)
    return coef[0]
lambda_est = compute_lambda(x_fit, y_fit)
print(f"Exponente de Lyapunov (rápido) ≈ {lambda_est:.4f}")

# 3) Estimación de dimensión de correlación
def estimate_correlation_dimension(data, eps_vals):
    M = data.shape[0]
    # Calcular distancias cuadradas
    D2 = np.sum((data[:, None, :] - data[None, :, :])**2, axis=2)
    C = np.zeros_like(eps_vals)
    for i, eps in enumerate(eps_vals):
        count = np.sum(D2 < eps**2) - M
        C[i] = count / (M*(M-1))
    # Ajuste lineal en región escalable central
    i1 = max(1, int(0.1 * len(eps_vals)))
    i2 = min(len(eps_vals)-1, int(0.4 * len(eps_vals)))
    log_eps = np.log(eps_vals[i1:i2])
    log_C = np.log(C[i1:i2])
    coef_dim = np.polyfit(log_eps, log_C, 1)
    return C, coef_dim[0]

# Tomar submuestra de 10% de puntos para GP (o máximo 2000)
M_sub = min(int(0.1 * N_pts), 2000)
C_vals, corr_dim = estimate_correlation_dimension(sol1[:M_sub],
                                                  np.logspace(-3, 0, 25))
print(f"Dimensión de correlación ≈ {corr_dim:.4f}")

# 4) Gráficas
plt.figure(figsize=(6,4))
plt.plot(t, ln_dist, label='ln(dist)')
plt.plot(x_fit, np.polyval(np.polyfit(x_fit, y_fit, 1), x_fit), 'r--',
         label=f'λ≈{lambda_est:.4f}')
plt.xlabel('Tiempo')
plt.ylabel('ln(dist)')
plt.title('Divergencia exponencial (RK4 paso fijo)')
plt.legend(); plt.tight_layout(); plt.show()

eps_vals = np.logspace(-3, 0, 25)
plt.figure(figsize=(6,4))
plt.loglog(eps_vals, C_vals, 'o-', label='C(ε)')
# recta de ajuste
i1 = max(1, int(0.1 * len(eps_vals)))
i2 = min(len(eps_vals)-1, int(0.4 * len(eps_vals)))
log_eps = np.log(eps_vals[i1:i2])
log_C = np.log(C_vals[i1:i2])
plt.loglog(eps_vals[i1:i2], np.exp(np.polyval(np.polyfit(log_eps, log_C, 1), log_eps)),
           'r--', label=f'D≈{corr_dim:.4f}')
plt.xlabel('ε'); plt.ylabel('C(ε)')
plt.title('Dimensión de correlación'); plt.legend(); plt.tight_layout(); plt.show()
