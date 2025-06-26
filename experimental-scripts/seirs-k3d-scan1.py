# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 22:31:34 2025

@author: eggra
"""
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import time
from tqdm import tqdm

# Configuración de la simulación Monte Carlo
NUM_SIMULATIONS = 10000  # Número de simulaciones
OUTPUT_FILE = "karmic_space_scan.csv"
TRANSIENT_TIME = 300     # Tiempo para eliminar transitorios
LYAPUNOV_TIME = 200      # Tiempo para cálculo de Lyapunov

# Rangos de parámetros (distribución uniforme)
PARAM_RANGES = {
    # Matriz de activación (α)
    'alpha_aa': (0.7, 1.0), 'alpha_ar': (0.5, 0.8), 'alpha_ad': (0.5, 0.8),
    'alpha_ra': (0.4, 0.7), 'alpha_rr': (0.3, 0.6), 'alpha_rd': (0.3, 0.6),
    'alpha_da': (0.4, 0.7), 'alpha_dr': (0.3, 0.6), 'alpha_dd': (0.3, 0.6),
    
    # Matriz de interferencia (κ)
    'kappa_aa': (-0.5, -0.2), 'kappa_ar': (0.1, 0.4), 'kappa_ad': (0.1, 0.4),
    'kappa_ra': (0.1, 0.4), 'kappa_rr': (-0.5, -0.2), 'kappa_rd': (0.1, 0.4),
    'kappa_da': (0.1, 0.4), 'kappa_dr': (0.1, 0.4), 'kappa_dd': (-0.5, -0.2),
    
    # Tasas de manifestación (σ)
    'sigma_a': (0.2, 0.5), 'sigma_r': (0.3, 0.6), 'sigma_d': (0.3, 0.6),
    
    # Tasas de resolución (δ)
    'delta_a': (0.1, 0.3), 'delta_r': (0.2, 0.4), 'delta_d': (0.2, 0.4),
    
    # Tasas de recaída (ω)
    'omega_a': (0.05, 0.15), 'omega_r': (0.1, 0.2), 'omega_d': (0.08, 0.18),
    
    # Parámetro de sinergia
    'gamma': (0.1, 0.4)
}

def initialize_random_parameters():
    """Genera parámetros aleatorios dentro de los rangos definidos"""
    params = {}
    for param, (low, high) in PARAM_RANGES.items():
        params[param] = np.random.uniform(low, high)
    return params

def initialize_random_state():
    """Genera un estado inicial aleatorio normalizado"""
    state = np.random.rand(12)  # 4 estados × 3 venenos
    # Normalizar cada veneno para sumar 1
    for i in range(3):
        start_idx = i*4
        end_idx = start_idx + 4
        state[start_idx:end_idx] /= state[start_idx:end_idx].sum()
    return state

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
    
    return np.array([dSdt[0], dSdt[1], dSdt[2], 
                     dEdt[0], dEdt[1], dEdt[2],
                     dIdt[0], dIdt[1], dIdt[2],
                     dRdt[0], dRdt[1], dRdt[2]])

def calculate_lyapunov(y0, params, t_span, t_eval):
    """Calcula el exponente de Lyapunov con método de dos trayectorias"""
    # Construir matrices
    alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
    
    # Simulación de referencia
    sol_ref = solve_ivp(
        lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
        t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-6
    )
    
    # Trayectoria perturbada
    perturbation = np.random.normal(0, 1e-6, len(y0))
    y0_pert = y0 + perturbation
    sol_pert = solve_ivp(
        lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
        t_span, y0_pert, t_eval=t_eval, method='LSODA', rtol=1e-6
    )
    
    # Calcular divergencia
    divergence = np.linalg.norm(sol_ref.y - sol_pert.y, axis=0)
    
    # Filtrar valores pequeños y calcular Lyapunov
    valid_idx = (t_eval > t_span[0] + TRANSIENT_TIME) & (divergence > 1e-10)
    if np.sum(valid_idx) < 5:
        return -np.inf, divergence
    
    t_valid = t_eval[valid_idx]
    log_div = np.log(divergence[valid_idx])
    
    # Ajuste lineal
    coeffs = np.polyfit(t_valid, log_div, 1)
    return coeffs[0], divergence

def classify_attractor(lyap_exp, state_history):
    """Clasifica el tipo de atractor basado en Lyapunov y estadísticas"""
    # Obtener solo los componentes activos (I_a, I_r, I_d)
    I_active = state_history[6:9, -100:].T  # Últimos 100 puntos
    
    # Calcular estadísticas
    mean_intensity = np.mean(I_active, axis=0)
    std_intensity = np.std(I_active, axis=0)
    max_intensity = np.max(I_active, axis=0)
    
    # Clasificación basada en Lyapunov
    if lyap_exp > 0.05:
        return "Caótico"
    elif lyap_exp > 0.01:
        # Verificar si hay periodicidad débil
        if np.max(std_intensity) > 0.1:
            return "Caos débil (Ciclo-caótico)"
        return "Caótico (Transición)"
    elif abs(lyap_exp) < 0.01:
        if np.max(std_intensity) > 0.05:
            return "Periódico"
        return "Punto fijo"
    else:
        if np.max(mean_intensity) < 0.1:
            return "Equilibrio kármico"
        return "Punto fijo inestable"

def run_monte_carlo_simulation(num_simulations):
    """Ejecuta la simulación Monte Carlo"""
    # Preparar archivo de salida
    header = list(PARAM_RANGES.keys()) + \
             [f"init_{s}{v}" for s in ['a', 'r', 'd'] for v in ['S', 'E', 'I', 'R']] + \
             ["lyapunov_exp", "attractor_type", "simulation_time"]
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(",".join(header) + "\n")
    
    # Bucle de simulaciones
    for i in tqdm(range(num_simulations)):
        start_time = time.time()
        
        # Inicialización aleatoria
        params = initialize_random_parameters()
        y0 = initialize_random_state()
        
        # Construir matrices
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        # Tiempos para Lyapunov (después de transitorios)
        t_span = [0, TRANSIENT_TIME + LYAPUNOV_TIME]
        t_eval = np.linspace(TRANSIENT_TIME, TRANSIENT_TIME + LYAPUNOV_TIME, 500)
        
        # Calcular exponente de Lyapunov
        lyap_exp, divergence = calculate_lyapunov(y0, params, t_span, t_eval)
        
        # Simulación completa para estadísticas
        full_t_eval = np.linspace(0, TRANSIENT_TIME + LYAPUNOV_TIME, 1000)
        sol = solve_ivp(
            lambda t, y: karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma),
            [0, TRANSIENT_TIME + LYAPUNOV_TIME], y0, t_eval=full_t_eval, method='LSODA', rtol=1e-6
        )
        
        # Clasificar atractor
        attractor_type = classify_attractor(lyap_exp, sol.y)
        sim_time = time.time() - start_time
        
        # Preparar datos para guardar
        param_values = [params[k] for k in PARAM_RANGES.keys()]
        init_values = list(y0)
        result_values = [lyap_exp, attractor_type, sim_time]
        
        # Guardar en archivo
        with open(OUTPUT_FILE, 'a') as f:
            line = ",".join([str(x) for x in param_values + init_values + result_values])
            f.write(line + "\n")
    
    print(f"\nSimulación completada. Resultados guardados en {OUTPUT_FILE}")

def analyze_results():
    """Analiza los resultados de la simulación Monte Carlo"""
    df = pd.read_csv(OUTPUT_FILE)
    
    # Análisis básico
    print("\nResumen estadístico del espacio kármico:")
    print(f"Total simulaciones: {len(df)}")
    print(f"Media exponente Lyapunov: {df['lyapunov_exp'].mean():.4f}")
    
    # Distribución de tipos de atractores
    print("\nDistribución de atractores:")
    print(df['attractor_type'].value_counts(normalize=True) * 100)
    
    # Correlaciones con Lyapunov
    print("\nParámetros más correlacionados con Lyapunov:")
    param_cols = list(PARAM_RANGES.keys())
    correlations = df[param_cols].corrwith(df['lyapunov_exp']).abs().sort_values(ascending=False)
    print(correlations.head(10))
    
    # Identificar regiones del espacio de parámetros
    chaotic_df = df[df['attractor_type'].str.contains('Caótico')]
    periodic_df = df[df['attractor_type'].str.contains('Periódico')]
    fixed_df = df[df['attractor_type'] == 'Punto fijo']
    
    print(f"\nRegiones identificadas:")
    print(f"- Caóticas: {len(chaotic_df)} puntos ({len(chaotic_df)/len(df)*100:.1f}%)")
    print(f"- Periódicas: {len(periodic_df)} puntos ({len(periodic_df)/len(df)*100:.1f}%)")
    print(f"- Punto fijo: {len(fixed_df)} puntos ({len(fixed_df)/len(df)*100:.1f}%)")
    
    # Guardar análisis regional
    chaotic_df.to_csv("chaotic_region.csv", index=False)
    periodic_df.to_csv("periodic_region.csv", index=False)
    fixed_df.to_csv("fixed_point_region.csv", index=False)

# Ejecutar simulación Monte Carlo
run_monte_carlo_simulation(NUM_SIMULATIONS)

# Analizar resultados
analyze_results()