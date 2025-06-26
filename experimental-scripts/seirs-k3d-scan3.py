# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 02:33:52 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm
import warnings

# Suprimir advertencias específicas de integración
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate._ivp.lsoda")

# =============================================
# FUNCIONES BASE (MODELO Y CÁLCULOS OPTIMIZADOS)
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
    
    # Activación no-lineal con estabilización numérica
    activation = alpha @ I
    activation = np.exp(np.clip(activation, -10, 10))  # Limitar rango para estabilidad
    
    # Interferencia con estabilización
    interference = kappa @ R
    interference = np.arctan(interference * 2)
    
    # Sinergia con estabilización numérica
    total_I = np.sum(I)
    synergistic = 1 + gamma * np.clip(total_I, 0, 10)**2  # Limitar rango
    
    # Retroalimentación con estabilización
    feedback = np.array([
        np.clip(Ia, 0, 1)**2 * np.clip(Ra - 0.3, -1, 1),
        np.clip(Ir, 0, 1)**2 * np.clip(Rr - 0.3, -1, 1),
        np.clip(Id, 0, 1)**2 * np.clip(Rd - 0.3, -1, 1)
    ])
    
    # Ecuaciones diferenciales con estabilización
    dSdt = -S * activation * synergistic + omega * R
    dEdt = S * activation * synergistic - sigma * E + 0.2 * feedback
    dIdt = sigma * E - delta * I - I * interference - 0.3 * feedback
    dRdt = delta * I - omega * R + I * interference + 0.1 * feedback
    
    # Limitar valores extremos
    return np.clip(np.array([
        dSdt[0], dSdt[1], dSdt[2], 
        dEdt[0], dEdt[1], dEdt[2],
        dIdt[0], dIdt[1], dIdt[2],
        dRdt[0], dRdt[1], dRdt[2]
    ]), -100, 100)  # Limitar derivadas extremas

def robust_integration(model_func, t_span, y0, method='BDF', max_step=0.1):
    """Integración robusta con manejo de errores y múltiples intentos"""
    # Primer intento con tolerancias estándar
    try:
        sol = solve_ivp(
            model_func, t_span, y0, 
            method=method, 
            rtol=1e-4, 
            atol=1e-7,
            max_step=max_step
        )
        return sol
    except Exception as e:
        print(f"Primer intento fallido: {str(e)}")
    
    # Segundo intento con tolerancias más relajadas
    try:
        sol = solve_ivp(
            model_func, t_span, y0, 
            method=method, 
            rtol=1e-2, 
            atol=1e-4,
            max_step=max_step
        )
        return sol
    except Exception as e:
        print(f"Segundo intento fallido: {str(e)}")
    
    # Tercer intento con método diferente
    new_method = 'Radau' if method == 'BDF' else 'BDF'
    try:
        sol = solve_ivp(
            model_func, t_span, y0, 
            method=new_method, 
            rtol=1e-3, 
            atol=1e-5,
            max_step=max_step
        )
        return sol
    except Exception as e:
        print(f"Tercer intento fallido: {str(e)}")
        return None

def calculate_lyapunov(y0, params, max_time=500, perturbation=1e-6, renormalize_time=5):
    """Cálculo robusto de Lyapunov con manejo de errores"""
    try:
        # Construir matrices
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        # Tiempos de integración
        steps = int(max_time / renormalize_time)
        if steps < 10:  # Mínimo de pasos
            return -np.inf
            
        t_eval = np.linspace(0, max_time, steps+1)
        
        # Estado de referencia
        y_ref = y0.copy()
        
        # Estado perturbado
        y_pert = y0 + np.random.normal(0, perturbation, len(y0))
        
        # Acumulador para Lyapunov
        lyapunov_sum = 0
        valid_steps = 0
        
        for i in range(steps):
            # Intervalo actual
            t_interval = [t_eval[i], t_eval[i+1]]
            
            # Crear función de modelo para este paso
            def model_func(t, y):
                return karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma)
            
            # Integrar sistema de referencia
            sol_ref = robust_integration(model_func, t_interval, y_ref, max_step=renormalize_time/2)
            if sol_ref is None or not sol_ref.success:
                continue
            y_ref = sol_ref.y[:, -1]
            
            # Integrar sistema perturbado
            sol_pert = robust_integration(model_func, t_interval, y_pert, max_step=renormalize_time/2)
            if sol_pert is None or not sol_pert.success:
                continue
            y_pert = sol_pert.y[:, -1]
            
            # Calcular divergencia y re-normalizar
            delta_y = y_pert - y_ref
            distance = norm(delta_y)
            
            if distance > 1e-12:  # Evitar distancias demasiado pequeñas
                # Factor de expansión en este intervalo
                expansion = np.log(distance / perturbation)
                lyapunov_sum += expansion
                valid_steps += 1
                
                # Re-normalizar la perturbación
                y_pert = y_ref + (perturbation / distance) * delta_y
        
        # Calcular exponente de Lyapunov solo si hay suficientes pasos válidos
        if valid_steps > 10:
            return lyapunov_sum / (valid_steps * renormalize_time)
        return -np.inf
    except Exception as e:
        print(f"Error en cálculo de Lyapunov: {str(e)}")
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
# ESCANEO MONTE CARLO DEL ESPACIO KÁRMICO (OPTIMIZADO)
# =============================================
# Configuración
NUM_SIMULATIONS = 8000
OUTPUT_FILE = "karmic_space_scan_robust.csv"
PARAM_RANGES = {
    'alpha_aa': (1.5, 3.5), 'alpha_ar': (1.0, 3.0), 'alpha_ad': (1.0, 3.0),
    'alpha_ra': (1.0, 2.5), 'alpha_rr': (1.0, 2.5), 'alpha_rd': (1.0, 2.5),
    'alpha_da': (1.0, 2.5), 'alpha_dr': (1.0, 2.5), 'alpha_dd': (1.0, 2.5),
    'kappa_aa': (-3.0, -0.5), 'kappa_ar': (0.5, 2.5), 'kappa_ad': (0.5, 2.5),
    'kappa_ra': (0.5, 2.5), 'kappa_rr': (-3.0, -0.5), 'kappa_rd': (0.5, 2.5),
    'kappa_da': (0.5, 2.5), 'kappa_dr': (0.5, 2.5), 'kappa_dd': (-3.0, -0.5),
    'sigma_a': (0.5, 2.5), 'sigma_r': (0.5, 3.0), 'sigma_d': (0.5, 3.0),
    'delta_a': (0.1, 0.8), 'delta_r': (0.1, 1.0), 'delta_d': (0.1, 1.0),
    'omega_a': (0.005, 0.1), 'omega_r': (0.005, 0.15), 'omega_d': (0.005, 0.15),
    'gamma': (1.5, 4.0)
}

def generate_random_configuration():
    """Genera una configuración aleatoria de parámetros y estado inicial"""
    # Parámetros aleatorios con distribución más estable
    params = {}
    for param, (low, high) in PARAM_RANGES.items():
        # Mezcla de distribuciones uniforme y normal para evitar extremos
        if np.random.rand() > 0.7:
            params[param] = np.random.uniform(low, high)
        else:
            mean = (low + high) / 2
            std = (high - low) / 6
            params[param] = np.clip(np.random.normal(mean, std), low, high)
    
    # Estado inicial aleatorio más estable
    y0 = np.zeros(12)
    for i in range(3):
        # Distribución más uniforme para evitar desequilibrios extremos
        vals = np.random.dirichlet(np.ones(4))
        y0[i*4: (i+1)*4] = vals
    
    return params, y0

def run_karmic_scan(num_simulations):
    """Ejecuta el escaneo del espacio kármico con manejo robusto"""
    # Preparar archivo de salida
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            # Cabecera
            header = list(PARAM_RANGES.keys()) + [f"y0_{i}" for i in range(12)] + [
                "lyapunov_exp", "dynamics_type", "computation_time"
            ]
            f.write(",".join(header) + "\n")
    
    # Ejecutar simulaciones con barra de progreso
    progress_bar = tqdm(total=num_simulations, desc="Escaneando espacio kármico")
    completed = 0
    
    while completed < num_simulations:
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
        
        completed += 1
        progress_bar.update(1)
    
    progress_bar.close()

def analyze_results():
    """Analiza y resume los resultados del escaneo"""
    if not os.path.exists(OUTPUT_FILE):
        print("No se encontraron resultados para analizar")
        return
    
    try:
        # Cargar datos
        df = pd.read_csv(OUTPUT_FILE)
        
        # Verificar que existe la columna necesaria
        if 'dynamics_type' not in df.columns:
            print("Error: Columna 'dynamics_type' no encontrada")
            print("Columnas disponibles:", df.columns.tolist())
            return
        
        # Análisis básico
        print("\n=== RESUMEN DEL ESCANEO KÁRMICO ===")
        print(f"Total simulaciones: {len(df)}")
        print(f"Exponente Lyapunov promedio: {df['lyapunov_exp'].mean():.4f}")
        
        # Distribución de tipos de dinámica
        dyn_dist = df['dynamics_type'].value_counts(normalize=True) * 100
        print("\nDistribución de dinámicas:")
        print(dyn_dist)
        
        # Parámetros más correlacionados con caos
        if 'Caos' in dyn_dist.index:
            chaos_mask = df['dynamics_type'].str.contains('Caos', na=False)
            print("\nParámetros asociados con caos:")
            for param in PARAM_RANGES.keys():
                if param in df.columns:
                    chaos_mean = df.loc[chaos_mask, param].mean()
                    all_mean = df[param].mean()
                    print(f"{param}: {chaos_mean:.3f} (vs promedio {all_mean:.3f})")
        
        # Identificar simulaciones con alto Lyapunov
        if not df.empty:
            # Guardar todas las simulaciones
            df.to_csv("all_simulations.csv", index=False)
            
            # Filtrar y guardar simulaciones interesantes
            if 'lyapunov_exp' in df.columns:
                high_lyap = df[df['lyapunov_exp'] > 0.05]
                if not high_lyap.empty:
                    high_lyap.to_csv("high_lyapunov_simulations.csv", index=False)
                    print(f"\nSimulaciones con Lyapunov > 0.05: {len(high_lyap)}")
            
            print("\nResultados guardados en:")
            print(f"- {OUTPUT_FILE} (datos completos)")
            print("- all_simulations.csv (backup completo)")
            if not high_lyap.empty:
                print("- high_lyapunov_simulations.csv (simulaciones caóticas)")
        else:
            print("\nNo hay datos válidos para guardar")
            
    except Exception as e:
        print(f"Error en análisis de resultados: {str(e)}")

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("=== INICIO DEL ESCANEO ROBUSTO DEL ESPACIO KÁRMICO ===")
    print(f"Simulaciones programadas: {NUM_SIMULATIONS}")
    print("Configuración mejorada para manejar sistemas stiff")
    
    # Ejecutar escaneo
    run_karmic_scan(NUM_SIMULATIONS)
    
    # Analizar resultados
    analyze_results()
    
    print("\n¡Proceso completado con éxito!")