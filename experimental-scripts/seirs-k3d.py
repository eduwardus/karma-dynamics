# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 22:13:32 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import pandas as pd
from sklearn.decomposition import PCA

# =============================================
# PARÁMETROS DEL MODELO KÁRMICO (TRES VENENOS)
# =============================================
# Matriz de activación (α)
alpha = np.array([
    [0.9, 0.7, 0.8],  # Avidya (ignorancia) se auto-activa y activa a los demás
    [0.6, 0.4, 0.5],  # Raga (apego) es activado principalmente por Avidya
    [0.7, 0.5, 0.6]   # Dvesha (aversión) es activado por Avidya
])

# Matriz de interferencia (κ) - Sabiduría que contrarresta los venenos
kappa = np.array([
    [-0.4, 0.3, 0.2],   # Sabiduría de Avidya reduce Raga y Dvesha
    [0.25, -0.3, 0.15],  # Sabiduría de Raga reduce Avidya pero puede aumentar Dvesha
    [0.2, 0.15, -0.35]   # Sabiduría de Dvesha reduce los otros venenos
])

# Tasa de manifestación (σ): Cómo las intenciones se convierten en acciones
sigma = np.array([0.3, 0.4, 0.35])  # Avidya, Raga, Dvesha

# Tasa de resolución (δ): Cómo las acciones se transforman en sabiduría
delta = np.array([0.2, 0.25, 0.22])  # Avidya es más difícil de resolver

# Tasa de recaída (ω): Sabiduría que vuelve a estado latente
omega = np.array([0.1, 0.15, 0.12])  # Raga tiene mayor tendencia a recaer

# Parámetro de sinergia (γ)
gamma = 0.25  # Los venenos se refuerzan mutuamente

# Condiciones iniciales (karma total normalizado a 1 por veneno)
S0 = np.array([0.4, 0.3, 0.35])  # Latente
E0 = np.array([0.2, 0.25, 0.2])  # Intención
I0 = np.array([0.3, 0.35, 0.3])  # Activo
R0 = np.array([0.1, 0.1, 0.15])  # Resuelto

y0 = np.concatenate([S0, E0, I0, R0])
y0_pert = y0 + np.random.normal(0, 1e-6, len(y0))

# Tiempo de simulación
t_span = [0, 500]
t_eval = np.linspace(0, 500, 5000)

# =============================================
# SISTEMA DE ECUACIONES DIFERENCIALES
# =============================================
def karmic_model(t, y):
    # Descomponer el vector de estado
    Sa, Sr, Sd = y[0], y[1], y[2]   # Latente (Avidya, Raga, Dvesha)
    Ea, Er, Ed = y[3], y[4], y[5]   # Intención
    Ia, Ir, Id = y[6], y[7], y[8]   # Activo
    Ra, Rr, Rd = y[9], y[10], y[11] # Resuelto
    
    S = np.array([Sa, Sr, Sd])
    E = np.array([Ea, Er, Ed])
    I = np.array([Ia, Ir, Id])
    R = np.array([Ra, Rr, Rd])
    
    # Fuerza de activación sinérgica
    activation = alpha @ I
    total_I = np.sum(I)
    synergistic = (1 + gamma * total_I)
    
    # Término de interferencia
    interference = kappa @ R
    
    # Ecuaciones diferenciales
    dSdt = -S * activation * synergistic + omega * R
    dEdt = S * activation * synergistic - sigma * E
    dIdt = sigma * E - delta * I - I * interference
    dRdt = delta * I - omega * R + I * interference
    
    return np.concatenate([dSdt, dEdt, dIdt, dRdt])

# =============================================
# SIMULACIÓN Y ANÁLISIS DE ATRACTORES
# =============================================
# Función para calcular exponente de Lyapunov
def calculate_lyapunov(t, divergence, transient=100):
    mask = (t > transient) & (divergence > 1e-12)
    if np.sum(mask) < 10:
        return -np.inf
    
    log_div = np.log(divergence[mask])
    t_valid = t[mask]
    
    if len(log_div) > 2:
        coeffs = np.polyfit(t_valid, log_div, 1)
        return coeffs[0]
    return -np.inf

# Simulación
sol = solve_ivp(karmic_model, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-6)
sol_pert = solve_ivp(karmic_model, t_span, y0_pert, t_eval=t_eval, method='LSODA', rtol=1e-6)

# Divergencia entre trayectorias
divergence = norm(sol.y - sol_pert.y, axis=0)
lyapunov_exp = calculate_lyapunov(t_eval, divergence)

# Clasificación de atractores
def classify_attractor(lyap_exp, trajectories):
    I_a, I_r, I_d = trajectories[6:9]
    
    # Calcular propiedades estadísticas
    std_dev = np.std(I_a[-1000:])
    mean_val = np.mean(I_a[-1000:])
    
    # Análisis de Fourier
    spectrum = np.abs(np.fft.rfft(I_a - np.mean(I_a)))
    freq = np.fft.rfftfreq(len(I_a), t_eval[1]-t_eval[0])
    peak_ratio = np.max(spectrum[1:]) / np.mean(spectrum[1:]) if len(spectrum) > 1 else 0
    
    # Clasificación
    if lyap_exp > 0.05:
        return "Caótico (Atractor extraño)"
    elif lyap_exp > 0.01:
        if peak_ratio > 5:
            return "Caos débil (Ciclo-caótico)"
        return "Caótico (Transición)"
    elif abs(lyap_exp) < 0.01:
        if std_dev > 0.05:
            return "Periódico (Ciclo límite)"
        return "Punto fijo"
    else:
        return "Convergente"

# Clasificar el atractor
attractor_type = classify_attractor(lyapunov_exp, sol.y)

# =============================================
# VISUALIZACIÓN DE RESULTADOS
# =============================================
plt.figure(figsize=(16, 12))

# 1. Evolución temporal de los venenos activos
plt.subplot(2, 2, 1)
plt.plot(t_eval, sol.y[6], 'r-', label='Avidya (Ignorancia)')
plt.plot(t_eval, sol.y[7], 'g-', label='Raga (Apego)')
plt.plot(t_eval, sol.y[8], 'b-', label='Dvesha (Aversión)')
plt.title('Evolución de los Tres Venenos Activos')
plt.xlabel('Tiempo Kármico')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True)

# 2. Espacio de fases 3D
ax = plt.subplot(2, 2, 2, projection='3d')
ax.plot(sol.y[6], sol.y[7], sol.y[8], 'm-', lw=0.5, alpha=0.7)
ax.set_xlabel('Avidya')
ax.set_ylabel('Raga')
ax.set_zlabel('Dvesha')
plt.title('Espacio de Fases de los Tres Venenos')

# 3. Divergencia de trayectorias
plt.subplot(2, 2, 3)
plt.semilogy(t_eval, divergence, 'c-')
plt.title('Divergencia entre Trayectorias Kármicas')
plt.xlabel('Tiempo')
plt.ylabel('log(||δy||)')
plt.annotate(f'λ = {lyapunov_exp:.4f}', (0.7, 0.9), xycoords='axes fraction', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True)

# 4. Proyección PCA para reducir dimensionalidad
pca = PCA(n_components=2)
state_matrix = sol.y[6:9].T  # Usamos solo los venenos activos
pca_proj = pca.fit_transform(state_matrix)

plt.subplot(2, 2, 4)
plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=t_eval, cmap='viridis', s=2)
plt.colorbar(label='Tiempo')
plt.xlabel('Componente PCA 1')
plt.ylabel('Componente PCA 2')
plt.title('Proyección PCA del Atractor')
plt.grid(True)

# Panel de diagnóstico
plt.figtext(0.1, 0.02, 
            f"TIPO DE ATRACTOR: {attractor_type}\n"
            f"Exponente de Lyapunov: {lyapunov_exp:.4f}\n"
            f"Sinergia (γ): {gamma}\n"
            f"Auto-activación Avidya (α_aa): {alpha[0,0]}\n"
            f"Interferencia Sabiduría (κ_aa): {kappa[0,0]}", 
            fontsize=14, bbox=dict(facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('karmic_attractors.png', dpi=300)
plt.show()

# =============================================
# ANÁLISIS ADICIONAL DE BIFURCACIONES
# =============================================
# Análisis de bifurcación variando γ (sinergia)
gamma_values = np.linspace(0, 0.5, 100)
max_Ia = []

print("Calculando diagrama de bifurcación...")
for g in gamma_values:
    gamma = g
    sol = solve_ivp(karmic_model, [0, 1000], y0, t_eval=np.linspace(500, 1000, 500), method='LSODA')
    max_Ia.append(np.max(sol.y[6][-100:]))  # Máximos de Avidya en estado estacionario

# Gráfico de bifurcación
plt.figure(figsize=(10, 6))
plt.scatter(gamma_values, max_Ia, s=2, alpha=0.7, c='purple')
plt.title('Diagrama de Bifurcación - Variando Sinergia (γ)')
plt.xlabel('Parámetro de Sinergia (γ)')
plt.ylabel('Máxima Intensidad de Avidya')
plt.grid(True)
plt.savefig('karmic_bifurcation.png', dpi=300)
plt.show()

# =============================================
# CLASIFICACIÓN DE ATRACTORES DETECTADOS
# =============================================
def detect_attractor_type(lyap_exp, max_vals, pca_proj):
    # 1. Punto fijo estable
    if lyap_exp < -0.05 and np.std(max_vals) < 0.01:
        return {
            "type": "Punto Fijo Estable",
            "stability": "Alta",
            "description": "El sistema converge a un estado kármico equilibrado sin oscilaciones"
        }
    
    # 2. Ciclo límite periódico
    elif -0.05 <= lyap_exp <= 0.05 and np.std(max_vals) > 0.05:
        return {
            "type": "Ciclo Límite Periódico",
            "stability": "Media",
            "description": "Comportamiento oscilatorio regular entre estados kármicos"
        }
    
    # 3. Atractor caótico
    elif lyap_exp > 0.05:
        # Calcular dimensión de correlación (proxy para fractalidad)
        distances = pca_proj[:, 0]**2 + pca_proj[:, 1]**2
        hist, bins = np.histogram(distances, bins=50)
        entropy = -np.sum(hist[hist>0] * np.log(hist[hist>0]))
        
        return {
            "type": "Atractor Caótico",
            "stability": "Baja",
            "description": f"Comportamiento impredecible con estructura fractal (entropía: {entropy:.2f})"
        }
    
    # 4. Punto fijo inestable
    else:
        return {
            "type": "Punto Fijo Inestable",
            "stability": "Crítica",
            "description": "Sistema sensible a perturbaciones, puede evolucionar a diferentes estados"
        }

# Clasificación detallada
attractor_class = detect_attractor_type(lyapunov_exp, max_Ia, pca_proj)

# Resultados finales
print("\n" + "="*60)
print("ANÁLISIS COMPLETO DEL SISTEMA KÁRMICO")
print("="*60)
print(f"Tipo de atractor detectado: {attractor_class['type']}")
print(f"Estabilidad: {attractor_class['stability']}")
print(f"Exponente de Lyapunov: {lyapunov_exp:.4f}")
print(f"Descripción: {attractor_class['description']}")
print("="*60)