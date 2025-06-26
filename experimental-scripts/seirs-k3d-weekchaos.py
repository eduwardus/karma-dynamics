# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 23:52:08 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Función para construir matrices (igual que antes)
def build_matrices(params):
    alpha = np.array([
        [params['alpha_aa'], params['alpha_ar'], params['alpha_ad']],
        [params['alpha_ra'], params['alpha_rr'], params['alpha_rd']],
        [params['alpha_da'], params['alpha_dr'], params['alpha_dd']]
    ])
    kappa = np.array([
        [params['kappa_aa'], params['kappa_ar'], params['kappa_ad']],
        [params['kappa_ra'], params['kappa_rr'], params['kappa_rd']],
        [params['kappa_da'], params['kappa_dr'], params['kappa_dd']]
    ])
    sigma = np.array([params['sigma_a'], params['sigma_r'], params['sigma_d']])
    delta = np.array([params['delta_a'], params['delta_r'], params['delta_d']])
    omega = np.array([params['omega_a'], params['omega_r'], params['omega_d']])
    gamma = params['gamma']
    return alpha, kappa, sigma, delta, omega, gamma

# Modelo kármico revisado (igual que antes)
def karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma):
    Sa, Sr, Sd, Ea, Er, Ed, Ia, Ir, Id, Ra, Rr, Rd = y
    S = np.array([Sa, Sr, Sd])
    E = np.array([Ea, Er, Ed])
    I = np.array([Ia, Ir, Id])
    R = np.array([Ra, Rr, Rd])
    
    # 1. ACTIVACIÓN NO-LINEAL
    activation = alpha @ I
    activation = 2.0 / (1 + np.exp(-2 * activation)) - 1.0
    
    # 2. INTERFERENCIA NO-LINEAL
    interference = kappa @ R
    interference = np.tanh(interference)
    
    # 3. SINERGIA NO-ADITIVA
    total_I = np.sum(I)
    synergistic = 1 + gamma * np.tanh(total_I)
    
    # 4. RETROALIMENTACIÓN
    feedback = np.array([
        Ia * (Ra - 0.5),
        Ir * (Rr - 0.5),
        Id * (Rd - 0.5)
    ])
    
    # Ecuaciones
    dSdt = -S * activation * synergistic + omega * R
    dEdt = S * activation * synergistic - sigma * E + 0.1 * feedback
    dIdt = sigma * E - delta * I - I * interference - 0.2 * feedback
    dRdt = delta * I - omega * R + I * interference + 0.1 * feedback
    
    return np.array([dSdt[0], dSdt[1], dSdt[2], 
                     dEdt[0], dEdt[1], dEdt[2],
                     dIdt[0], dIdt[1], dIdt[2],
                     dRdt[0], dRdt[1], dRdt[2]])

# Función para calcular Lyapunov con método de re-normalización
def calculate_lyapunov(y0, params, t_span, max_time=1000, perturbation=1e-6, renormalize_time=10):
    """Cálculo robusto de Lyapunov con re-normalización periódica"""
    # Construir matrices
    alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
    
    # Tiempos de integración
    t_eval = np.arange(0, max_time, renormalize_time)
    
    # Estado de referencia
    y_ref = y0.copy()
    
    # Estado perturbado
    y_pert = y0 + np.random.normal(0, perturbation, len(y0))
    
    # Lista para almacenar los factores de expansión
    lyapunov_sums = []
    divergence_history = []
    
    # Evolución del sistema
    for i in range(len(t_eval)-1):
        # Integrar ambos sistemas
        sol_ref = solve_ivp(
            lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
            [t_eval[i], t_eval[i+1]], y_ref, method='LSODA', rtol=1e-8
        )
        sol_pert = solve_ivp(
            lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
            [t_eval[i], t_eval[i+1]], y_pert, method='LSODA', rtol=1e-8
        )
        
        # Obtener estados finales
        y_ref = sol_ref.y[:, -1]
        y_pert_final = sol_pert.y[:, -1]
        
        # Calcular divergencia
        delta_y = y_pert_final - y_ref
        distance = norm(delta_y)
        divergence_history.append(distance)
        
        # Factor de expansión en este intervalo
        if distance > 0:
            expansion = np.log(distance / perturbation)
            lyapunov_sums.append(expansion)
            
            # Re-normalizar la perturbación
            y_pert = y_ref + (perturbation / distance) * delta_y
        
    # Calcular exponente de Lyapunov
    if lyapunov_sums:
        total_time = t_eval[-1] - t_eval[0]
        lyap_exp = sum(lyapunov_sums) / total_time
        return lyap_exp, divergence_history
    return -np.inf, []

# Parámetros extremadamente no-lineales
params = {
    'alpha_aa': 3.5, 'alpha_ar': 3.0, 'alpha_ad': 3.0,
    'alpha_ra': 2.5, 'alpha_rr': 2.0, 'alpha_rd': 2.0,
    'alpha_da': 2.5, 'alpha_dr': 2.0, 'alpha_dd': 2.0,
    'kappa_aa': -2.5, 'kappa_ar': 2.0, 'kappa_ad': 2.0,
    'kappa_ra': 2.0, 'kappa_rr': -2.5, 'kappa_rd': 2.0,
    'kappa_da': 2.0, 'kappa_dr': 2.0, 'kappa_dd': -2.5,
    'sigma_a': 2.5, 'sigma_r': 3.0, 'sigma_d': 3.0,
    'delta_a': 0.5, 'delta_r': 1.0, 'delta_d': 1.0,
    'omega_a': 0.02, 'omega_r': 0.05, 'omega_d': 0.05,
    'gamma': 3.0
}

# Condiciones iniciales desequilibradas
y0 = np.array([0.01, 0.01, 0.01,  # S
               0.3, 0.4, 0.2,      # E
               0.5, 0.6, 0.7,      # I
               0.19, 0.19, 0.19])  # R

# Simulación extendida
t_span = [0, 2000]
t_eval = np.linspace(0, 2000, 5000)
alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)

sol = solve_ivp(
    lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
    t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8
)

# Visualización de la dinámica
plt.figure(figsize=(14, 10))

# Evolución temporal
plt.subplot(2, 2, 1)
plt.plot(t_eval, sol.y[6], 'r-', label='Avidya (Ignorancia)', alpha=0.7)
plt.plot(t_eval, sol.y[7], 'g-', label='Raga (Deseo)', alpha=0.7)
plt.plot(t_eval, sol.y[8], 'b-', label='Dvesha (Aversión)', alpha=0.7)
plt.title('Dinámica de los Tres Venenos')
plt.xlabel('Tiempo Kármico')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True)

# Espacio de fases 3D
ax = plt.subplot(2, 2, 2, projection='3d')
ax.plot(sol.y[6], sol.y[7], sol.y[8], 'm-', lw=0.5, alpha=0.7)
ax.set_xlabel('Avidya')
ax.set_ylabel('Raga')
ax.set_zlabel('Dvesha')
plt.title('Espacio de Fases 3D')

# Cálculo de Lyapunov con método robusto
lyap_exp, div_history = calculate_lyapunov(y0, params, [0, 1500])
print(f"\nExponente de Lyapunov calculado: {lyap_exp:.4f}")

# Visualización de la divergencia
if div_history:
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(len(div_history)) * 10, div_history, 'c-')
    plt.yscale('log')
    plt.xlabel('Tiempo')
    plt.ylabel('Divergencia (log)')
    plt.title('Evolución de la Divergencia')
    plt.grid(True)

# Espectro de potencias
plt.subplot(2, 2, 4)
for i, color, label in zip([6,7,8], ['r','g','b'], ['Avidya','Raga','Dvesha']):
    signal = sol.y[i] - np.mean(sol.y[i])
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), t_eval[1]-t_eval[0])
    plt.plot(freqs[freqs>0], fft[freqs>0], color, label=label, alpha=0.7)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frecuencia (log)')
plt.ylabel('Amplitud (log)')
plt.title('Espectro de Frecuencias')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('karmic_chaos_analysis.png', dpi=150)
plt.show()

# Diagnóstico final
if lyap_exp > 0.05:
    print("¡COMPORTAMIENTO CAÓTICO DETECTADO!")
    print("Características:")
    print("- Exponente de Lyapunov positivo")
    print("- Espectro de frecuencias continuo")
    print("- Trayectoria irregular pero acotada")
elif lyap_exp > 0:
    print("Comportamiento débilmente caótico o transicional")
else:
    print("Comportamiento periódico o estable")