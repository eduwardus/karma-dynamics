# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 23:02:40 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import time
from tqdm import tqdm
import os

# =============================================
# FUNCIONES BASE (MODELO Y CÁLCULOS)
# =============================================
def build_matrices(params):
    """Construye matrices a partir de parámetros"""
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

def karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma):
    Sa, Sr, Sd, Ea, Er, Ed, Ia, Ir, Id, Ra, Rr, Rd = y
    S = np.array([Sa, Sr, Sd])
    E = np.array([Ea, Er, Ed])
    I = np.array([Ia, Ir, Id])
    R = np.array([Ra, Rr, Rd])
    
    # Activación no-lineal
    activation = alpha @ I
    activation = 2.0 / (1 + np.exp(-2 * activation)) - 1.0
    
    # Interferencia no-lineal
    interference = kappa @ R
    interference = np.tanh(interference)
    
    # Sinergia no-aditiva
    total_I = np.sum(I)
    synergistic = 1 + gamma * np.tanh(total_I)
    
    # Retroalimentación
    feedback = np.array([
        Ia * (Ra - 0.5),
        Ir * (Rr - 0.5),
        Id * (Rd - 0.5)
    ])
    
    # Ecuaciones diferenciales
    dSdt = -S * activation * synergistic + omega * R
    dEdt = S * activation * synergistic - sigma * E + 0.1 * feedback
    dIdt = sigma * E - delta * I - I * interference - 0.2 * feedback
    dRdt = delta * I - omega * R + I * interference + 0.1 * feedback
    
    return np.array([dSdt[0], dSdt[1], dSdt[2], 
                     dEdt[0], dEdt[1], dEdt[2],
                     dIdt[0], dIdt[1], dIdt[2],
                     dRdt[0], dRdt[1], dRdt[2]])

def calculate_lyapunov(y0, params, max_time=500, perturbation=1e-6, renormalize_time=10):
    """Cálculo robusto de Lyapunov con re-normalización periódica"""
    # Construir matrices
    alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
    
    # Tiempos de integración
    steps = int(max_time / renormalize_time)
    t_eval = np.linspace(0, max_time, steps+1)
    
    # Estado de referencia
    y_ref = y0.copy()
    
    # Estado perturbado
    y_pert = y0 + np.random.normal(0, perturbation, len(y0))
    
    # Acumulador para Lyapunov
    lyapunov_sum = 0
    
    for i in range(steps):
        # Intervalo actual
        t_interval = [t_eval[i], t_eval[i+1]]
        
        # Integrar sistema de referencia
        sol_ref = solve_ivp(
            lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
            t_interval, y_ref, method='LSODA', rtol=1e-6
        )
        y_ref = sol_ref.y[:, -1]
        
        # Integrar sistema perturbado
        sol_pert = solve_ivp(
            lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
            t_interval, y_pert, method='LSODA', rtol=1e-6
        )
        y_pert = sol_pert.y[:, -1]
        
        # Calcular divergencia y re-normalizar
        delta_y = y_pert - y_ref
        distance = np.linalg.norm(delta_y)
        
        if distance > 0:
            # Factor de expansión en este intervalo
            expansion = np.log(distance / perturbation)
            lyapunov_sum += expansion
            
            # Re-normalizar la perturbación
            y_pert = y_ref + (perturbation / distance) * delta_y
    
    # Calcular exponente de Lyapunov
    if steps > 0:
        return lyapunov_sum / max_time
    return -np.inf

def classify_dynamics(lyap_exp):
    """Clasifica el tipo de dinámica basado en Lyapunov"""
    if lyap_exp > 0.05:
        return "Caos fuerte"
    elif lyap_exp > 0.01:
        return "Caos débil"
    elif abs(lyap_exp) < 0.01:
        return "Periódico"
    elif lyap_exp < -0.05:
        return "Punto fijo estable"
    else:
        return "Punto fijo inestable"

# =============================================
# ESCANEO MONTE CARLO DEL ESPACIO KÁRMICO
# =============================================
# Configuración
NUM_SIMULATIONS = 5000
OUTPUT_FILE = "karmic_space_scan.csv"
PARAM_RANGES = {
    'alpha_aa': (1.0, 4.0), 'alpha_ar': (0.5, 3.5), 'alpha_ad': (0.5, 3.5),
    'alpha_ra': (0.5, 3.0), 'alpha_rr': (0.5, 3.0), 'alpha_rd': (0.5, 3.0),
    'alpha_da': (0.5, 3.0), 'alpha_dr': (0.5, 3.0), 'alpha_dd': (0.5, 3.0),
    'kappa_aa': (-3.0, -0.5), 'kappa_ar': (0.1, 3.0), 'kappa_ad': (0.1, 3.0),
    'kappa_ra': (0.1, 3.0), 'kappa_rr': (-3.0, -0.5), 'kappa_rd': (0.1, 3.0),
    'kappa_da': (0.1, 3.0), 'kappa_dr': (0.1, 3.0), 'kappa_dd': (-3.0, -0.5),
    'sigma_a': (0.5, 3.0), 'sigma_r': (0.5, 3.5), 'sigma_d': (0.5, 3.5),
    'delta_a': (0.1, 1.0), 'delta_r': (0.1, 1.5), 'delta_d': (0.1, 1.5),
    'omega_a': (0.001, 0.1), 'omega_r': (0.001, 0.2), 'omega_d': (0.001, 0.2),
    'gamma': (0.5, 4.0)
}

def generate_random_configuration():
    """Genera una configuración aleatoria de parámetros y estado inicial"""
    # Parámetros aleatorios
    params = {}
    for param, (low, high) in PARAM_RANGES.items():
        params[param] = np.random.uniform(low, high)
    
    # Estado inicial aleatorio
    y0 = np.zeros(12)
    for i in range(3):
        # Generar valores positivos y normalizar
        vals = np.random.exponential(1.0, size=4)
        vals /= vals.sum()
        y0[i*4: (i+1)*4] = vals
    
    return params, y0

def run_karmic_scan(num_simulations):
    """Ejecuta el escaneo del espacio kármico"""
    # Preparar archivo de salida
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            # Cabecera
            header = list(PARAM_RANGES.keys()) + [f"y0_{i}" for i in range(12)] + [
                "lyapunov_exp", "dynamics_type", "computation_time"
            ]
            f.write(",".join(header) + "\n")
    
    # Ejecutar simulaciones
    for i in tqdm(range(num_simulations), desc="Escaneando espacio kármico"):
        start_time = time.time()
        
        # Generar configuración aleatoria
        params, y0 = generate_random_configuration()
        
        # Calcular exponente de Lyapunov
        lyap_exp = calculate_lyapunov(y0, params)
        
        # Clasificar dinámica
        dyn_type = classify_dynamics(lyap_exp)
        comp_time = time.time() - start_time
        
        # Guardar resultados
        with open(OUTPUT_FILE, 'a') as f:
            # Parámetros
            param_vals = [str(params[p]) for p in PARAM_RANGES.keys()]
            # Estado inicial
            y0_vals = [str(v) for v in y0]
            # Resultados
            results = [str(lyap_exp), dyn_type, str(comp_time)]
            
            line = ",".join(param_vals + y0_vals + results)
            f.write(line + "\n")

def analyze_results():
    """Analiza y resume los resultados del escaneo"""
    if not os.path.exists(OUTPUT_FILE):
        print("No se encontraron resultados para analizar")
        return
    
    # Cargar datos
    df = pd.read_csv(OUTPUT_FILE)
    
    # Análisis básico
    print("\n=== RESUMEN DEL ESCANEO KÁRMICO ===")
    print(f"Total simulaciones: {len(df)}")
    print(f"Exponente Lyapunov promedio: {df['lyapunov_exp'].mean():.4f}")
    
    # Distribución de tipos de dinámica
    dyn_dist = df['dynamics_type'].value_counts(normalize=True) * 100
    print("\nDistribución de dinámicas:")
    print(dyn_dist)
    
    # Parámetros más correlacionados con caos
    chaos_mask = df['dynamics_type'].str.contains('Caos')
    if chaos_mask.sum() > 0:
        print("\nParámetros asociados con caos:")
        for param in PARAM_RANGES.keys():
            chaos_mean = df[chaos_mask][param].mean()
            all_mean = df[param].mean()
            print(f"{param}: {chaos_mean:.3f} (vs promedio {all_mean:.3f})")
    
    # Guardar regiones interesantes
    df[chaos_mask].to_csv("chaotic_regions.csv", index=False)
    df[df['dynamics_type'] == "Periódico"].to_csv("periodic_regions.csv", index=False)
    
    print("\nResultados guardados en:")
    print(f"- {OUTPUT_FILE} (todos los datos)")
    print("- chaotic_regions.csv (regiones caóticas)")
    print("- periodic_regions.csv (regiones periódicas)")

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("=== INICIO DEL ESCANEO DEL ESPACIO KÁRMICO ===")
    print(f"Simulaciones programadas: {NUM_SIMULATIONS}")
    print(f"Rangos paramétricos: {len(PARAM_RANGES)} dimensiones")
    
    # Ejecutar escaneo
    run_karmic_scan(NUM_SIMULATIONS)
    
    # Analizar resultados
    analyze_results()
    
    print("\n¡Escaneo completado con éxito!")