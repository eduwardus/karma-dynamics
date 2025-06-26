# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:52:00 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Simulador KARMIC - Búsqueda de Caos (Configuración Extrema)
Created on Sat Jun 15 09:45:00 2024
@author: DeepSeek
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import time
from tqdm import tqdm
import os
from numpy.linalg import norm
import warnings
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import logging
import gc

# Configurar logging
logging.basicConfig(filename='karmic_chaos_extreme.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================
# CONFIGURACIÓN PRINCIPAL (15,000 ITERACIONES)
# =============================================
NUM_SIMULATIONS = 15000
OUTPUT_FILE = "karmic_chaos_extreme_scan.csv"
CHECKPOINT_INTERVAL = 1000
LOG_INTERVAL = 100

# =============================================
# FUNCIONES BASE (MODELO ORIGINAL)
# =============================================
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

def karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma):
    Sa, Sr, Sd, Ea, Er, Ed, Ia, Ir, Id, Ra, Rr, Rd = y
    S = np.array([Sa, Sr, Sd])
    E = np.array([Ea, Er, Ed])
    I = np.array([Ia, Ir, Id])
    R = np.array([Ra, Rr, Rd])
    
    # Activación no-lineal
    activation = np.exp(alpha @ I)
    
    # Interferencia
    interference = kappa @ R
    
    # Sinergia
    total_I = np.sum(I)
    synergistic = 1 + gamma * total_I**2
    
    # Retroalimentación
    feedback = np.array([
        Ia**2 * (Ra - 0.3),
        Ir**2 * (Rr - 0.3),
        Id**2 * (Rd - 0.3)
    ])
    
    # Ecuaciones diferenciales
    dSdt = -S * activation * synergistic + omega * R
    dEdt = S * activation * synergistic - sigma * E + 0.2 * feedback
    dIdt = sigma * E - delta * I - I * interference - 0.3 * feedback
    dRdt = delta * I - omega * R + I * interference + 0.1 * feedback
    
    return np.array([
        dSdt[0], dSdt[1], dSdt[2], 
        dEdt[0], dEdt[1], dEdt[2],
        dIdt[0], dIdt[1], dIdt[2],
        dRdt[0], dRdt[1], dRdt[2]
    ])

def robust_integration(model_func, t_span, y0):
    try:
        sol = solve_ivp(
            model_func, t_span, y0, 
            method='Radau', 
            rtol=1e-8, 
            atol=1e-10,
            max_step=0.5
        )
        return sol if sol.success else None
    except Exception as e:
        logger.error(f"Error en integración: {str(e)}")
        return None

def calculate_lyapunov(y0, params, max_time=400, perturbation=1e-8, renormalize_time=1.5):
    try:
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        steps = int(max_time / renormalize_time)
        if steps < 10:
            return -np.inf
            
        t_eval = np.linspace(0, max_time, steps + 1)
        y_ref = y0.copy()
        y_pert = y0 + perturbation * (np.random.random(len(y0)) - 0.5)
        
        lyapunov_sum = 0
        valid_steps = 0
        
        def model_func(t, y):
            return karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma)
        
        for i in range(steps):
            t_interval = [t_eval[i], t_eval[i+1]]
            
            sol_ref = robust_integration(model_func, t_interval, y_ref)
            if sol_ref is None: break
            y_ref = sol_ref.y[:, -1]
            
            sol_pert = robust_integration(model_func, t_interval, y_pert)
            if sol_pert is None: break
            y_pert = sol_pert.y[:, -1]
            
            delta_y = y_pert - y_ref
            distance = norm(delta_y)
            
            if distance > 1e-15:
                expansion = np.log(distance / perturbation)
                lyapunov_sum += expansion
                valid_steps += 1
                y_pert = y_ref + (perturbation / distance) * delta_y
        
        return lyapunov_sum / (valid_steps * renormalize_time) if valid_steps > 10 else -np.inf
    except Exception as e:
        logger.error(f"Error en Lyapunov: {str(e)}")
        return -np.inf

def classify_dynamics(lyap_exp):
    if lyap_exp > 0.05:
        return "Caos fuerte"
    elif lyap_exp > 0.01:
        return "Caos débil"
    elif lyap_exp > 0.001:
        return "Periódico complejo"
    elif lyap_exp > -0.01:
        return "Periódico simple"
    elif lyap_exp > -0.05:
        return "Punto fijo estable"
    else:
        return "Punto fijo inestable"

# =============================================
# ESCANEO CON PARÁMETROS EXTREMOS
# =============================================
PARAM_RANGES = {
    'alpha_aa': (3.2, 4.5), 'alpha_ar': (2.0, 3.5), 'alpha_ad': (2.0, 3.5),
    'alpha_ra': (2.0, 3.0), 'alpha_rr': (3.2, 4.5), 'alpha_rd': (2.0, 3.0),
    'alpha_da': (2.0, 3.0), 'alpha_dr': (2.0, 3.0), 'alpha_dd': (3.2, 4.5),
    'kappa_aa': (-1.5, -0.8), 'kappa_ar': (0.5, 1.5), 'kappa_ad': (0.5, 1.5),
    'kappa_ra': (0.5, 1.5), 'kappa_rr': (-1.5, -0.8), 'kappa_rd': (0.5, 1.5),
    'kappa_da': (0.5, 1.5), 'kappa_dr': (0.5, 1.5), 'kappa_dd': (-1.5, -0.8),
    'sigma_a': (0.8, 1.5), 'sigma_r': (0.8, 1.5), 'sigma_d': (0.8, 1.5),
    'delta_a': (0.02, 0.1), 'delta_r': (0.02, 0.1), 'delta_d': (0.02, 0.1),
    'omega_a': (0.0005, 0.005), 'omega_r': (0.0005, 0.005), 'omega_d': (0.0005, 0.005),
    'gamma': (4.5, 7.0)
}

def generate_chaos_configuration():
    params = {}
    
    # Parámetros con distribución sesgada hacia extremos
    for param, (low, high) in PARAM_RANGES.items():
        if param in ['gamma', 'alpha_aa', 'alpha_rr', 'alpha_dd']:
            # 80% de probabilidad de valores altos
            if np.random.rand() < 0.8:
                params[param] = np.random.uniform(0.8 * high, high)
            else:
                params[param] = np.random.uniform(low, high)
        elif 'kappa' in param and 'aa' in param or 'rr' in param or 'dd' in param:
            # Interferencia inhibitoria menos extrema
            params[param] = np.random.uniform(low, high)
        else:
            params[param] = np.random.uniform(low, high)
    
    # Estado inicial con desequilibrios controlados
    y0 = np.zeros(12)
    for i in range(3):
        # Perfiles epidemiológicos distintos por grupo
        group_type = np.random.choice(["outbreak", "controlled", "recovered"], p=[0.6, 0.3, 0.1])
        
        if group_type == "outbreak":
            # Grupo con brote activo
            base_vals = np.array([0.05, 0.20, 0.70, 0.05])  # S, E, I, R
        elif group_type == "controlled":
            # Grupo con epidemia controlada
            base_vals = np.array([0.40, 0.10, 0.10, 0.40])
        else:  # recovered
            # Grupo con mayoría recuperada
            base_vals = np.array([0.10, 0.05, 0.05, 0.80])
            
        # Aplicar ruido controlado
        noise = 0.15 * np.random.randn(4)
        noise = np.clip(noise, -0.14, 0.14)
        vals = np.clip(base_vals + noise, 0.01, 0.99)
        vals /= vals.sum()  # Normalizar
        y0[i*4: (i+1)*4] = vals
    
    return params, y0

def run_chaos_scan(num_simulations):
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a' if file_exists else 'w') as f:
        if not file_exists:
            header = list(PARAM_RANGES.keys()) + [f"y0_{i}" for i in range(12)] + [
                "lyapunov_exp", "dynamics_type", "computation_time"
            ]
            f.write(",".join(header) + "\n")
    
    completed = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            completed = sum(1 for line in f) - 1
    
    logger.info(f"Iniciando escaneo extremo con {num_simulations} iteraciones")
    logger.info(f"Simulaciones ya completadas: {completed}")
    
    progress_bar = tqdm(total=num_simulations, initial=completed, 
                        desc="Escaneo Extremo", unit="sim")
    
    start_time = time.time()
    last_checkpoint = completed
    
    while completed < num_simulations:
        iter_start = time.time()
        params, y0 = generate_chaos_configuration()
        
        lyap_exp = calculate_lyapunov(y0, params)
        dyn_type = classify_dynamics(lyap_exp)
        comp_time = time.time() - iter_start
        
        with open(OUTPUT_FILE, 'a') as f:
            param_vals = [str(params[p]) for p in PARAM_RANGES.keys()]
            y0_vals = [str(v) for v in y0]
            results = [str(lyap_exp), dyn_type, str(comp_time)]
            f.write(",".join(param_vals + y0_vals + results) + "\n")
        
        completed += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"Lyap": f"{lyap_exp:.4f}", "Type": dyn_type[:10]})
        
        # Guardar checkpoint
        if completed % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            sims_done = completed - last_checkpoint
            rate = sims_done / elapsed if elapsed > 0 else 0
            remaining = (num_simulations - completed) / rate if rate > 0 else float('inf')
            
            logger.info(f"Checkpoint: {completed}/{num_simulations} completadas")
            logger.info(f"Tasa: {rate:.2f} sim/s - Restante: {remaining/3600:.2f} horas")
            
            with open("scan_progress.txt", 'w') as pf:
                pf.write(f"Completadas: {completed}\n")
                pf.write(f"Tasa: {rate:.2f} sim/s\n")
                pf.write(f"Restante: {remaining/3600:.2f} horas\n")
            
            last_checkpoint = completed
            start_time = time.time()
            gc.collect()
        
        # Log periódico
        if completed % LOG_INTERVAL == 0:
            logger.info(f"Sim {completed}: Lyap={lyap_exp:.4f}, Tipo={dyn_type}, T={comp_time:.2f}s")
            logger.debug(f"Params: gamma={params.get('gamma',0):.2f}, alpha_aa={params.get('alpha_aa',0):.2f}")
    
    progress_bar.close()
    logger.info("Escaneo extremo completado")

# =============================================
# FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# =============================================

def analyze_chaos_results():
    """Analiza los resultados del escaneo y extrae configuraciones caóticas"""
    if not os.path.exists(OUTPUT_FILE):
        print("No se encontraron resultados para analizar.")
        return

    try:
        # Leer el archivo de resultados
        df = pd.read_csv(OUTPUT_FILE)
        
        print("\n=== RESULTADOS DEL ESCANEO EXTREMO ===")
        print(f"Total de simulaciones: {len(df)}")
        
        # Verificar si hay datos
        if len(df) == 0:
            print("No hay datos en el archivo de resultados.")
            return
        
        # Estadísticas básicas del exponente de Lyapunov
        if 'lyapunov_exp' in df.columns:
            max_lyap = df['lyapunov_exp'].max()
            min_lyap = df['lyapunov_exp'].min()
            mean_lyap = df['lyapunov_exp'].mean()
            print(f"\nExponente de Lyapunov máximo: {max_lyap:.6f}")
            print(f"Exponente de Lyapunov mínimo: {min_lyap:.6f}")
            print(f"Exponente de Lyapunov promedio: {mean_lyap:.6f}")
            
            # Clasificación de dinámicas
            if 'dynamics_type' in df.columns:
                dyn_counts = df['dynamics_type'].value_counts()
                print("\nDistribución de dinámicas:")
                print(dyn_counts)
                
                # Filtrar configuraciones caóticas
                chaos_mask = df['dynamics_type'].str.contains('Caos')
                chaos_df = df[chaos_mask]
                
                if len(chaos_df) > 0:
                    print(f"\nSe encontraron {len(chaos_df)} configuraciones caóticas.")
                    # Guardar configuraciones caóticas en un archivo
                    chaos_df.to_csv("chaotic_configurations.csv", index=False)
                    print("Configuraciones caóticas guardadas en 'chaotic_configurations.csv'.")
                else:
                    print("\nNo se encontraron configuraciones caóticas.")
            else:
                print("Advertencia: No se encontró la columna 'dynamics_type' en los resultados.")
        else:
            print("Advertencia: No se encontró la columna 'lyapunov_exp' en los resultados.")
            
        # Guardar un resumen estadístico
        stats_file = "scan_statistics.txt"
        with open(stats_file, 'w') as sf:
            sf.write(f"Total simulaciones: {len(df)}\n")
            sf.write(f"Max Lyapunov: {max_lyap:.6f}\n")
            sf.write(f"Min Lyapunov: {min_lyap:.6f}\n")
            sf.write(f"Mean Lyapunov: {mean_lyap:.6f}\n")
            sf.write("\nDistribución de tipos:\n")
            sf.write(dyn_counts.to_string())
            
        print(f"\nResumen estadístico guardado en '{stats_file}'.")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        logger.error(f"Error en analyze_chaos_results: {str(e)}")

def plot_chaotic_trajectory(params, y0, t_span=(0, 500)):
    """Genera gráficos para una configuración dada"""
    try:
        # Construir matrices de parámetros
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        # Función para el modelo
        def model_func(t, y):
            return karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma)
        
        # Resolver el sistema
        sol = robust_integration(model_func, t_span, y0)
        if sol is None:
            print("Error: La integración falló.")
            return
            
        t = sol.t
        S = sol.y[0:3]   # [Sa, Sr, Sd]
        E = sol.y[3:6]   # [Ea, Er, Ed]
        I = sol.y[6:9]   # [Ia, Ir, Id]
        R = sol.y[9:12]  # [Ra, Rr, Rd]
        
        # Configurar gráficos
        plt.figure(figsize=(15, 12))
        
        # Gráfico 1: Trayectoria 3D de los infectados
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        ax1.plot(I[0], I[1], I[2], 'b-', lw=0.8)
        ax1.set_title("Trayectoria 3D de Infectados")
        ax1.set_xlabel("Ia")
        ax1.set_ylabel("Ir")
        ax1.set_zlabel("Id")
        ax1.grid(True)
        
        # Gráfico 2: Serie temporal de infectados
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(t, I[0], 'r-', label="Ia")
        ax2.plot(t, I[1], 'g-', label="Ir")
        ax2.plot(t, I[2], 'b-', label="Id")
        ax2.set_title("Evolución de Infectados")
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Población")
        ax2.legend()
        ax2.grid(True)
        
        # Gráfico 3: Comportamiento de S, E, I, R para el primer grupo
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(t, S[0], 'b-', label="Sa")
        ax3.plot(t, E[0], 'y-', label="Ea")
        ax3.plot(t, I[0], 'r-', label="Ia")
        ax3.plot(t, R[0], 'g-', label="Ra")
        ax3.set_title("Dinámica Grupo A")
        ax3.set_xlabel("Tiempo")
        ax3.set_ylabel("Población")
        ax3.legend()
        ax3.grid(True)
        
        # Gráfico 4: Parámetros clave
        ax4 = plt.subplot(2, 2, 4)
        key_params = {
            'gamma': params['gamma'],
            'alpha_aa': params['alpha_aa'],
            'kappa_aa': params['kappa_aa'],
            'delta_a': params['delta_a'],
            'omega_a': params['omega_a']
        }
        ax4.bar(key_params.keys(), key_params.values(), color='purple')
        ax4.set_title("Parámetros Clave")
        ax4.set_ylabel("Valor")
        ax4.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig("chaotic_trajectory.png")
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Error en la visualización: {str(e)}")
        logger.error(f"Error en plot_chaotic_trajectory: {str(e)}")

def find_and_visualize_chaos():
    """Busca las configuraciones más caóticas y genera gráficos"""
    chaotic_configs_file = "chaotic_configurations.csv"
    if not os.path.exists(chaotic_configs_file):
        print("No se encontraron configuraciones caóticas.")
        return
        
    try:
        # Leer configuraciones caóticas
        chaos_df = pd.read_csv(chaotic_configs_file)
        if len(chaos_df) == 0:
            print("El archivo de configuraciones caóticas está vacío.")
            return
            
        # Ordenar por exponente de Lyapunov (descendente)
        chaos_df = chaos_df.sort_values('lyapunov_exp', ascending=False)
        
        # Tomar las 3 configuraciones más caóticas
        top_configs = chaos_df.head(3)
        
        for idx, row in top_configs.iterrows():
            print(f"\nVisualizando configuración {idx} con Lyapunov={row['lyapunov_exp']:.6f}")
            # Extraer parámetros
            params = {param: row[param] for param in PARAM_RANGES.keys()}
            # Extraer estado inicial
            y0 = [row[f'y0_{i}'] for i in range(12)]
            # Generar gráficos
            plot_chaotic_trajectory(params, y0)
            
    except Exception as e:
        print(f"Error al visualizar configuraciones caóticas: {str(e)}")
        logger.error(f"Error en find_and_visualize_chaos: {str(e)}")

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("=== DETECTOR DE CAOS KÁRMICO - CONFIGURACIÓN EXTREMA ===")
    print(f"Ejecutando {NUM_SIMULATIONS} iteraciones con parámetros optimizados para caos")
    print("Parámetros clave:")
    print(f" - gamma: {PARAM_RANGES['gamma']}")
    print(f" - alpha_aa: {PARAM_RANGES['alpha_aa']}")
    print(f" - kappa_aa: {PARAM_RANGES['kappa_aa']}")
    
    # Paso 1: Ejecutar el escaneo
    run_chaos_scan(NUM_SIMULATIONS)
    
    # Paso 2: Analizar los resultados
    analyze_chaos_results()
    
    # Paso 3: Visualizar las configuraciones caóticas
    find_and_visualize_chaos()
    
    print("\n¡Proceso completo! Busca resultados en chaotic_configurations.csv")