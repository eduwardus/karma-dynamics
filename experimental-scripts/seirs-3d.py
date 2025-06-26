# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 21:10:55 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# PARÁMETROS DEL MODELO
# =====================
Λ = np.array([0.1, 0.1, 0.1])       # Reclutamiento de susceptibles
Γ = np.array([0.01, 0.01, 0.01])    # Mortalidad natural
Σ = np.array([0.3, 0.3, 0.3])       # Tasa de progresión E→I (1/periodo incubación)
Δ = np.array([0.2, 0.2, 0.2])       # Tasa de recuperación (1/duracion infección)
Ω = np.array([0.05, 0.05, 0.05])    # Pérdida de inmunidad
α = 0.15                             # Fuerza de sinergia entre virus

# Matriz de transmisión (alto contagio cruzado)
β = np.array([
    [0.8, 0.6, 0.4],  # Humano: contagio propio + refuerzo de mono y vaca
    [0.5, 0.7, 0.3],  # Mono:   contagio propio + refuerzo de humano
    [0.3, 0.2, 0.9]   # Vaca:   contagio propio + refuerzo de humano
])

# Matriz de inmunidad cruzada dual (positiva/negativa)
κ = np.array([
    [ 0.0, -0.4,  0.2],  # Humano: inmunidad negativa vs mono (-40%), positiva vs vaca (+20%)
    [ 0.3,  0.0, -0.1],  # Mono:   positiva vs humano (+30%), negativa vs vaca (-10%)
    [-0.2,  0.3,  0.0]   # Vaca:   negativa vs humano (-20%), positiva vs mono (+30%)
])

# Condiciones iniciales (población normalizada)
S0 = np.array([0.7, 0.7, 0.7])
E0 = np.array([0.1, 0.1, 0.1])
I0 = np.array([0.05, 0.05, 0.05])
R0 = np.array([0.15, 0.15, 0.15])
y0 = np.concatenate([S0, E0, I0, R0])

# Perturbación inicial para Lyapunov
y0_pert = y0 + np.random.normal(0, 1e-6, len(y0))

# Tiempo de simulación
t_span = [0, 500]
t_eval = np.linspace(0, 500, 5000)

# =========================================
# DEFINICIÓN DEL SISTEMA DE ECUACIONES
# =========================================
def seirs_vector(t, y):
    n_virus = 3
    S, E, I, R = y[0:3], y[3:6], y[6:9], y[9:12]
    
    # Fuerza de infección sinérgica (β·I con refuerzo por prevalencia total)
    I_total = np.sum(I)
    infection_force = β @ I * (1 + α * I_total)
    
    # Término de inmunidad cruzada dual
    cross_immunity = κ @ R
    
    # Ecuaciones diferenciales
    dS = Λ - S * infection_force + Ω * R - Γ * S
    dE = S * infection_force - Σ * E - Γ * E
    dI = Σ * E - Δ * I - Γ * I - I * cross_immunity
    dR = Δ * I - Ω * R - Γ * R + I * cross_immunity
    
    return np.concatenate([dS, dE, dI, dR])

# ===========================
# SIMULACIÓN Y ANÁLISIS
# ===========================
# Simulación del sistema
sol = solve_ivp(seirs_vector, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-6)
sol_pert = solve_ivp(seirs_vector, t_span, y0_pert, t_eval=t_eval, method='LSODA', rtol=1e-6)

# Divergencia entre trayectorias (norma euclídea)
divergence = np.linalg.norm(sol.y - sol_pert.y, axis=0)

# =========================================
# CÁLCULO ROBUSTO DEL EXPONENTE DE LYAPUNOV
# =========================================
def calculate_lyapunov(t, divergence, transient=100):
    """Calcula el exponente de Lyapunov con filtrado robusto"""
    # Ignorar fase transitoria inicial y valores muy pequeños
    mask = (t > transient) & (divergence > 1e-12)
    if np.sum(mask) < 10:  # Mínimo de puntos requeridos
        return -np.inf
    
    log_div = np.log(divergence[mask])
    t_valid = t[mask]
    
    # Ajuste lineal solo si hay suficientes puntos
    if len(log_div) > 2:
        coeffs = np.polyfit(t_valid, log_div, 1)
        return coeffs[0]
    return -np.inf

lyapunov_exp = calculate_lyapunov(t_eval, divergence)

# =========================================
# CLASIFICACIÓN DEL COMPORTAMIENTO
# =========================================
def classify_dynamics(lyap_exp, I_trajectories):
    """Clasifica el comportamiento dinámico basado en Lyapunov y trayectorias"""
    I_h, I_m, I_b = I_trajectories
    
    # 1. Verificar divergencia exponencial
    if lyap_exp > 0.01:
        # 2. Análisis espectral para confirmar caos
        spectrum = np.abs(np.fft.rfft(I_h[-1000:]))
        peaks = np.sort(spectrum)[-3:]
        if peaks[2]/peaks[1] < 2.0:  # Múltiples frecuencias significativas
            return "CAOS DETERMINISTA (Atractor extraño)"
        return "OSCILACIONES COMPLEJAS (Ciclo límite)"
    
    # 3. Comportamiento periódico
    elif abs(lyap_exp) < 0.01:
        std_dev = np.std(I_h[-1000:])
        if std_dev > 0.01: 
            return "OSCILACIONES PERIÓDICAS (Ciclo límite)"
        return "PUNTO FIJO ESTABLE"
    
    return "CONVERGENCIA A EQUILIBRIO"

# Obtener clasificación
chaos_type = classify_dynamics(lyapunov_exp, sol.y[6:9])

# ===========================
# VISUALIZACIÓN
# ===========================
plt.figure(figsize=(14, 10))

# Trayectorias de infectados
plt.subplot(2, 2, 1)
plt.plot(t_eval, sol.y[6], 'r-', label='I_humano')
plt.plot(t_eval, sol.y[7], 'g-', label='I_mono')
plt.plot(t_eval, sol.y[8], 'b-', label='I_vaca')
plt.title('Evolución Temporal de Infectados')
plt.xlabel('Tiempo')
plt.ylabel('Proporción Infectada')
plt.legend()
plt.grid(True)

# Divergencia exponencial
plt.subplot(2, 2, 2)
plt.semilogy(t_eval, divergence, 'm-')
plt.title('Divergencia entre Trayectorias')
plt.xlabel('Tiempo')
plt.ylabel('||δy(t)|| (log)')
plt.annotate(f'λ = {lyapunov_exp:.4f}', (0.7, 0.9), xycoords='axes fraction', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True)

# Atractor en espacio 3D
ax = plt.subplot(2, 2, 3, projection='3d')
ax.plot(sol.y[6], sol.y[7], sol.y[8], 'b-', lw=0.5, alpha=0.7)
ax.set_xlabel('I_humano')
ax.set_ylabel('I_mono')
ax.set_zlabel('I_vaca')
plt.title('Atractor en Espacio de Fases')

# Espectro de frecuencias
plt.subplot(2, 2, 4)
I_h = sol.y[6][-2000:]  # Usar solo el estado estacionario
freq = np.fft.rfftfreq(len(I_h), t_eval[1]-t_eval[0])
spectrum = np.abs(np.fft.rfft(I_h - np.mean(I_h)))
plt.plot(freq[freq>0], spectrum[freq>0], 'r-')
plt.title('Espectro de Frecuencias de I_humano')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.xlim([0, 0.5])
plt.grid(True)

# Panel de diagnóstico
plt.figtext(0.1, 0.02, 
            f"DIAGNÓSTICO: {chaos_type}\n"
            f"Parámetros: α={α}, β_hm={β[0,1]}, β_mb={β[1,2]}, κ_hm={κ[0,1]}, κ_mb={κ[1,2]}", 
            fontsize=14, bbox=dict(facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('seirs_vectorial_caos.png', dpi=300)
plt.show()