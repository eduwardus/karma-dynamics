# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 11:50:03 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Simulador KARMIC - Versión Caótica (15,000 iteraciones)
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
import gc  # Garbage Collector para gestión de memoria

# Configurar logging
logging.basicConfig(filename='karmic_chaos_scan.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================
# CONFIGURACIÓN PRINCIPAL (15,000 ITERACIONES)
# =============================================
NUM_SIMULATIONS = 15000  # 15,000 iteraciones
OUTPUT_FILE = "karmic_chaos_scan_15000.csv"
CHECKPOINT_INTERVAL = 1000  # Guardar cada 1000 simulaciones
LOG_INTERVAL = 100  # Loggear cada 100 simulaciones

# =============================================
# FUNCIONES BASE (MODELO ORIGINAL SIN ESTABILIZACIONES)
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
    """Formulación ORIGINAL sin estabilizaciones artificiales"""
    Sa, Sr, Sd, Ea, Er, Ed, Ia, Ir, Id, Ra, Rr, Rd = y
    S = np.array([Sa, Sr, Sd])
    E = np.array([Ea, Er, Ed])
    I = np.array([Ia, Ir, Id])
    R = np.array([Ra, Rr, Rd])
    
    # Activación no-lineal (ORIGINAL)
    activation = alpha @ I
    activation = np.exp(activation)  # SIN CLIP
    
    # Interferencia (ORIGINAL)
    interference = kappa @ R  # SIN ARCTAN
    
    # Sinergia (ORIGINAL)
    total_I = np.sum(I)
    synergistic = 1 + gamma * total_I**2  # SIN CLIP
    
    # Retroalimentación (ORIGINAL)
    feedback = np.array([
        Ia**2 * (Ra - 0.3),
        Ir**2 * (Rr - 0.3),
        Id**2 * (Rd - 0.3)
    ])
    
    # Ecuaciones diferenciales (SIN ACOTAMIENTO)
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
    """Integración adaptativa para sistemas stiff"""
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

def calculate_lyapunov(y0, params, max_time=200, perturbation=1e-8, renormalize_time=2):
    """Cálculo de Lyapunov optimizado para grandes volúmenes"""
    try:
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        # Configuración temporal optimizada
        steps = int(max_time / renormalize_time)
        if steps < 5:
            return -np.inf
            
        t_eval = np.linspace(0, max_time, steps + 1)
        
        # Estado de referencia y perturbado
        y_ref = y0.copy()
        y_pert = y0 + perturbation * (np.random.random(len(y0)) - 0.5)
        
        lyapunov_sum = 0
        valid_steps = 0
        
        def model_func(t, y):
            return karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma)
        
        for i in range(steps):
            t_interval = [t_eval[i], t_eval[i+1]]
            
            # Integrar sistema de referencia
            sol_ref = robust_integration(model_func, t_interval, y_ref)
            if sol_ref is None: 
                break
            y_ref = sol_ref.y[:, -1]
            
            # Integrar sistema perturbado
            sol_pert = robust_integration(model_func, t_interval, y_pert)
            if sol_pert is None: 
                break
            y_pert = sol_pert.y[:, -1]
            
            # Calcular divergencia
            delta_y = y_pert - y_ref
            distance = norm(delta_y)
            
            if distance > 1e-15:
                # Factor de expansión
                expansion = np.log(distance / perturbation)
                lyapunov_sum += expansion
                valid_steps += 1
                
                # Renormalización mínima
                y_pert = y_ref + (perturbation / distance) * delta_y
        
        return lyapunov_sum / (valid_steps * renormalize_time) if valid_steps > 5 else -np.inf
    except Exception as e:
        logger.error(f"Error en Lyapunov: {str(e)}")
        return -np.inf

def classify_dynamics(lyap_exp):
    """Clasificación mejorada para detección de caos"""
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
# ESCANEO PARA DETECCIÓN DE CAOS (OPTIMIZADO PARA 15K)
# =============================================

# Rangos optimizados para comportamiento caótico
PARAM_RANGES = {
    'alpha_aa': (2.8, 3.5), 'alpha_ar': (1.5, 2.8), 'alpha_ad': (1.5, 2.8),
    'alpha_ra': (1.5, 2.5), 'alpha_rr': (2.8, 3.5), 'alpha_rd': (1.5, 2.5),
    'alpha_da': (1.5, 2.5), 'alpha_dr': (1.5, 2.5), 'alpha_dd': (2.8, 3.5),
    'kappa_aa': (-2.5, -1.8), 'kappa_ar': (0.1, 0.5), 'kappa_ad': (0.1, 0.5),
    'kappa_ra': (0.1, 0.5), 'kappa_rr': (-2.5, -1.8), 'kappa_rd': (0.1, 0.5),
    'kappa_da': (0.1, 0.5), 'kappa_dr': (0.1, 0.5), 'kappa_dd': (-2.5, -1.8),
    'sigma_a': (0.7, 1.2), 'sigma_r': (0.7, 1.2), 'sigma_d': (0.7, 1.2),
    'delta_a': (0.05, 0.15), 'delta_r': (0.05, 0.15), 'delta_d': (0.05, 0.15),
    'omega_a': (0.001, 0.01), 'omega_r': (0.001, 0.01), 'omega_d': (0.001, 0.01),
    'gamma': (3.5, 5.0)
}

def generate_chaos_configuration():
    """Configuraciones optimizadas para caos con generación eficiente"""
    params = {}
    
    # Generación vectorizada de parámetros
    for param, (low, high) in PARAM_RANGES.items():
        # Para parámetros clave, usar distribución sesgada
        if param in ['alpha_aa', 'gamma', 'kappa_aa']:
            # 80% de probabilidad de estar en los extremos superiores
            if np.random.rand() < 0.8:
                params[param] = np.random.uniform(0.85 * high, high)
            else:
                params[param] = np.random.uniform(low, high)
        else:
            params[param] = np.random.uniform(low, high)
    
    # Estado inicial con desequilibrios controlados
    y0 = np.zeros(12)
    for i in range(3):
        # Mayor proporción en infectados
        base_vals = np.array([0.15, 0.05, 0.7, 0.1])  # S, E, I, R
        noise = 0.1 * np.random.randn(4)
        noise = np.clip(noise, -0.09, 0.09)
        vals = base_vals + noise
        vals = np.clip(vals, 0.01, 0.99)  # Evitar valores extremos
        vals /= vals.sum()  # Normalizar
        y0[i*4: (i+1)*4] = vals
    
    return params, y0

def run_chaos_scan(num_simulations):
    """Escaneo optimizado para grandes volúmenes con guardado periódico"""
    # Preparar archivo de salida
    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, 'a' if file_exists else 'w') as f:
        if not file_exists:
            header = list(PARAM_RANGES.keys()) + [f"y0_{i}" for i in range(12)] + [
                "lyapunov_exp", "dynamics_type", "computation_time"
            ]
            f.write(",".join(header) + "\n")
    
    # Verificar si ya hay simulaciones completadas
    completed = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            completed = sum(1 for line in f) - 1  # Restar encabezado
    
    # Mensaje inicial
    logger.info(f"Iniciando/continuando escaneo con {num_simulations} iteraciones")
    logger.info(f"Simulaciones ya completadas: {completed}")
    
    # Barra de progreso
    progress_bar = tqdm(total=num_simulations, initial=completed, 
                        desc="Escaneo Kármico", unit="sim")
    
    # Estadísticas de rendimiento
    start_time = time.time()
    last_checkpoint = completed
    
    # Bucle principal optimizado
    while completed < num_simulations:
        iter_start = time.time()
        
        # Generar configuración
        params, y0 = generate_chaos_configuration()
        
        # Calcular Lyapunov
        lyap_exp = calculate_lyapunov(y0, params)
        dyn_type = classify_dynamics(lyap_exp)
        comp_time = time.time() - iter_start
        
        # Guardar resultados
        with open(OUTPUT_FILE, 'a') as f:
            param_vals = [str(params[p]) for p in PARAM_RANGES.keys()]
            y0_vals = [str(v) for v in y0]
            results = [str(lyap_exp), dyn_type, str(comp_time)]
            f.write(",".join(param_vals + y0_vals + results) + "\n")
        
        completed += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"Lyap": f"{lyap_exp:.4f}", "Type": dyn_type[:10]})
        
        # Guardar checkpoint periódico
        if completed % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            sims_done = completed - last_checkpoint
            rate = sims_done / elapsed if elapsed > 0 else 0
            remaining = (num_simulations - completed) / rate if rate > 0 else float('inf')
            
            logger.info(f"Checkpoint: {completed}/{num_simulations} completadas")
            logger.info(f"Tasa: {rate:.2f} sim/s - Tiempo restante: {remaining/3600:.2f} horas")
            
            # Guardar estadísticas
            with open("scan_progress.txt", 'w') as pf:
                pf.write(f"Completadas: {completed}\n")
                pf.write(f"Tasa: {rate:.2f} sim/s\n")
                pf.write(f"Tiempo estimado restante: {remaining/3600:.2f} horas\n")
            
            last_checkpoint = completed
            start_time = time.time()
            gc.collect()  # Liberar memoria
        
        # Log periódico
        if completed % LOG_INTERVAL == 0:
            logger.info(f"Simulación {completed}: Lyap={lyap_exp:.4f}, Tipo={dyn_type}, Tiempo={comp_time:.2f}s")
    
    progress_bar.close()
    logger.info("Escaneo completado")

def analyze_chaos_results():
    """Análisis optimizado para grandes conjuntos de datos"""
    if not os.path.exists(OUTPUT_FILE):
        logger.error("No hay datos para analizar")
        return
    
    try:
        # Cargar datos en chunks para eficiencia
        chunks = pd.read_csv(OUTPUT_FILE, chunksize=5000)
        df = pd.concat(chunks)
        
        print("\n=== RESULTADOS DE DETECCIÓN DE CAOS ===")
        print(f"Simulaciones totales: {len(df)}")
        
        if 'lyapunov_exp' in df.columns:
            print(f"Exponente Lyapunov máximo: {df['lyapunov_exp'].max():.4f}")
            print(f"Exponente Lyapunov promedio: {df['lyapunov_exp'].mean():.4f}")
            print(f"Exponente Lyapunov mediano: {df['lyapunov_exp'].median():.4f}")
        
        if 'dynamics_type' in df.columns:
            dyn_dist = df['dynamics_type'].value_counts(normalize=True) * 100
            print("\nDistribución de dinámicas:")
            print(dyn_dist)
            
            # Filtrar configuraciones caóticas
            chaos_mask = df['dynamics_type'].str.contains('Caos', na=False)
            chaos_df = df[chaos_mask]
            
            if not chaos_df.empty:
                chaos_df.to_csv("chaotic_configurations.csv", index=False)
                print(f"\nConfiguraciones caóticas encontradas: {len(chaos_df)}")
                print("Guardadas en chaotic_configurations.csv")
                
                # Parámetros clave en configuraciones caóticas
                print("\nPromedios en configuraciones caóticas:")
                for param in ['alpha_aa', 'gamma', 'kappa_aa', 'delta_a', 'omega_a']:
                    if param in chaos_df.columns:
                        mean_val = chaos_df[param].mean()
                        std_val = chaos_df[param].std()
                        print(f"{param}: {mean_val:.3f} ± {std_val:.3f}")
            
            # Guardar top 10 configuraciones más caóticas
            if 'lyapunov_exp' in df.columns:
                top_chaos = df.nlargest(10, 'lyapunov_exp')
                top_chaos.to_csv("top_chaotic_configs.csv", index=False)
                print("\nTop 10 configuraciones más caóticas guardadas en top_chaotic_configs.csv")
        
        # Guardar análisis completo
        df.to_csv("full_analysis_results.csv", index=False)
        print("\nAnálisis completo guardado en full_analysis_results.csv")
        
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        print(f"Error en análisis: {str(e)}")

def plot_chaotic_trajectory(params, y0, t_span=(0, 200)):
    """Visualización optimizada de trayectorias"""
    try:
        alpha, kappa, sigma, delta, omega, gamma = build_matrices(params)
        
        def model_func(t, y):
            return karmic_model(t, y, alpha, kappa, sigma, delta, omega, gamma)
        
        sol = robust_integration(model_func, t_span, y0)
        if sol is None:
            print("Error en integración para visualización")
            return
        
        t = sol.t
        I = sol.y[6:9]  # Ia, Ir, Id
        
        plt.figure(figsize=(14, 10))
        
        # Trayectoria 3D
        ax1 = plt.subplot2grid((3, 2), (0, 0), projection='3d')
        ax1.plot(I[0], I[1], I[2], 'b-', lw=0.7)
        ax1.set_title("Espacio de Fases 3D (Infectados)", fontsize=12)
        ax1.set_xlabel("Ia", fontsize=10)
        ax1.set_ylabel("Ir", fontsize=10)
        ax1.set_zlabel("Id", fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # Serie temporal
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax2.plot(t, I[0], 'r-', label="Ia", alpha=0.8)
        ax2.plot(t, I[1], 'g-', label="Ir", alpha=0.8)
        ax2.plot(t, I[2], 'b-', label="Id", alpha=0.8)
        ax2.set_title("Evolución Temporal de Infectados", fontsize=12)
        ax2.set_xlabel("Tiempo", fontsize=10)
        ax2.set_ylabel("Población", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # Mapa de parámetros clave
        ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        param_keys = ['alpha_aa', 'gamma', 'kappa_aa', 'delta_a', 'omega_a']
        param_vals = [params[k] for k in param_keys]
        ax3.bar(param_keys, param_vals, color='purple', alpha=0.7)
        ax3.set_title("Parámetros Clave", fontsize=12)
        ax3.set_ylabel("Valor", fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.tick_params(axis='y', labelsize=8)
        
        # Histograma de valores
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        ax4.hist(I[0], bins=50, color='red', alpha=0.5, label="Ia")
        ax4.hist(I[1], bins=50, color='green', alpha=0.5, label="Ir")
        ax4.hist(I[2], bins=50, color='blue', alpha=0.5, label="Id")
        ax4.set_title("Distribución de Poblaciones Infectadas", fontsize=12)
        ax4.set_xlabel("Población", fontsize=10)
        ax4.set_ylabel("Frecuencia", fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        plt.savefig("chaotic_trajectory.png", dpi=150)
        plt.close()  # Cerrar figura para ahorrar memoria
        print("Visualización guardada en chaotic_trajectory.png")
        
    except Exception as e:
        logger.error(f"Error en visualización: {str(e)}")
        print(f"Error en visualización: {str(e)}")

def find_and_visualize_chaos():
    """Busca y visualiza configuraciones caóticas con manejo de memoria"""
    try:
        if not os.path.exists("chaotic_configurations.csv"):
            print("No se encontraron configuraciones caóticas")
            return
        
        # Cargar solo las columnas necesarias para ahorrar memoria
        df = pd.read_csv("chaotic_configurations.csv", usecols=['lyapunov_exp'] + 
                         list(PARAM_RANGES.keys()) + [f"y0_{i}" for i in range(12)])
        
        if df.empty:
            print("No hay configuraciones caóticas para visualizar")
            return
        
        # Seleccionar las 3 configuraciones más caóticas
        top_indices = df['lyapunov_exp'].nlargest(3).index
        
        for i, idx in enumerate(top_indices):
            print(f"\nVisualizando configuración {i+1} (Lyap_exp = {df.at[idx, 'lyapunov_exp']:.4f})")
            
            # Reconstruir parámetros y estado inicial
            params = {k: df.at[idx, k] for k in PARAM_RANGES.keys()}
            y0 = [df.at[idx, f"y0_{j}"] for j in range(12)]
            
            plot_chaotic_trajectory(params, y0)
            gc.collect()  # Liberar memoria entre visualizaciones
            
    except Exception as e:
        logger.error(f"Error en visualización: {str(e)}")
        print(f"Error en visualización: {str(e)}")

# =============================================
# EJECUCIÓN PRINCIPAL PARA 15,000 ITERACIONES
# =============================================
if __name__ == "__main__":
    print("=== DETECTOR DE CAOS KÁRMICO ===")
    print(f"Configuración para {NUM_SIMULATIONS} iteraciones")
    print("Tiempo estimado: 10-30 horas (dependiendo del hardware)")
    print("Se recomienda ejecutar en un servidor o durante la noche")
    print("Se guardará progreso periódicamente\n")
    
    # Paso 1: Escanear espacio paramétrico
    run_chaos_scan(NUM_SIMULATIONS)
    
    # Paso 2: Analizar resultados
    analyze_chaos_results()
    
    # Paso 3: Visualizar dinámicas caóticas
    find_and_visualize_chaos()
    
    print("\n¡Análisis completo con 15,000 iteraciones!")