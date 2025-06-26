# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:33:24 2025

@author: eggra
"""

import numpy as np
import os
import pickle
from scipy.stats import laplace, beta, poisson
from tqdm import tqdm
import time

# ===== PARÁMETROS KÁRMICOS =====
t_max = 100  # Tiempo máximo de simulación
dt = 0.1     # Paso temporal
n_sim = 15000  # Número total de simulaciones Monte Carlo
checkpoint_interval = 100  # Guardar cada 100 simulaciones
partial_save_interval = 10  # Guardar resultados parciales cada 10 simulaciones

# Condiciones iniciales (karma latente, activado, manifestándose, inmunidad)
X0 = np.array([[0.99, 0.01, 0.00, 0.00],  # avidyā 
               [0.99, 0.01, 0.00, 0.00],  # rāga
               [0.99, 0.01, 0.00, 0.00]]) # dveṣa

# Parámetros base (3 componentes)
sigma = np.array([0.7, 0.7, 0.7])     # Tasa base de maduración
gamma = np.array([0.2, 0.2, 0.2])     # Tasa base de purga
omega = np.array([0.1, 0.1, 0.1])     # Tasa base de recaída
kappa = 0.7                           # Factor de protección experiencial
lambda_g = 0.7                         # Factor de aprendizaje

# Matriz de acoplamiento kármico (α_ij)
alpha = np.array([[2.0, 0.1, 0.2], 
                  [0.1, 1.8, 0.3], 
                  [0.3, 0.2, 1.5]])

# Intensidades de ruido (D_beta, b_zeta, a_eta, lambda_xi)
noise_params = {
    'D_beta': np.array([0.2, 0.2, 0.2]),
    'b_zeta': np.array([0.3, 0.3, 0.3]),
    'a_eta': np.array([2.0, 2.0, 2.0]),
    'lambda_xi': np.array([0.3, 0.3, 0.3])
}

# ===== FUNCIONES AUXILIARES =====
def f_proteccion(I, kappa):
    """Función de protección por experiencia"""
    return 1 - kappa * np.clip(I, 0, 1/kappa)  # Asegura no negatividad

def g_aprendizaje(I, lambda_g):
    """Función de aprendizaje kármico"""
    return 1 - np.exp(-lambda_g * np.abs(I))

def load_checkpoint():
    """Cargar estado anterior si existe"""
    if os.path.exists('karmic_checkpoint.pkl'):
        with open('karmic_checkpoint.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def save_checkpoint(sim_count, results, start_time):
    """Guardar estado actual para poder reanudar"""
    checkpoint = {
        'sim_count': sim_count,
        'results': results,
        'start_time': start_time
    }
    with open('karmic_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

def save_partial_results(current_sim, results_array):
    """Guarda resultados parciales en un archivo .npz"""
    partial_file = f"karmic_partial_results_{current_sim}.npz"
    np.savez_compressed(
        partial_file,
        results=results_array[:current_sim],
        parameters={
            't_max': t_max,
            'dt': dt,
            'alpha': alpha,
            'noise_params': noise_params,
            'components': ['avidya', 'raga', 'dvesha'],
            'metrics': ['max_I', 't_max', 'area_I', 'n_peaks', 'lyapunov', 'apen', 'corr_dim']
        },
        simulation_range=(0, current_sim)
    )
    return partial_file

def initialize_simulation():
    """Inicializar o reanudar simulación"""
    checkpoint = load_checkpoint()
    
    if checkpoint:
        print(f"Reanudando simulación desde el punto {checkpoint['sim_count']}")
        return checkpoint['sim_count'], checkpoint['results'], checkpoint['start_time']
    
    # Crear estructura vacía para resultados
    results = np.zeros((n_sim, 3, 7))  # 7 indicadores por componente
    return 0, results, time.time()

# ===== FUNCIONES DE ANÁLISIS DE CAOS =====
def lyapunov_estimate(trajectory, min_points=100):
    """Estimación rápida de exponente de Lyapunov"""
    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    
    n = len(trajectory)
    if n < min_points: 
        return 0.0
    
    # Seleccionar puntos de referencia aleatorios
    ref_indices = np.random.choice(range(min_points, n-min_points), size=min(10, n//10), replace=False)
    divergences = []
    
    for i in ref_indices:
        ref_val = trajectory[i]
        # Comparar con puntos posteriores
        dists = np.abs(trajectory[i+1:i+51] - ref_val)
        valid_dists = dists[dists > 1e-10]
        if len(valid_dists) > 0:
            divergences.append(np.log(np.max(valid_dists)))
    
    return np.mean(divergences) if divergences else 0.0

def approximate_entropy(U, m=2, r=0.2):
    """Entropía aproximada (versión robusta)"""
    if len(U) < 100:
        return 0.0
    
    U = np.asarray(U)
    if np.ptp(U) < 1e-10:  # Evitar división por cero
        return 0.0
    
    # Normalización
    U_norm = (U - np.min(U)) / (np.ptp(U) + 1e-12)
    
    N = len(U_norm)
    if N <= m+1:
        return 0.0
    
    def _phi(m_val):
        x = np.lib.stride_tricks.sliding_window_view(U_norm, window_shape=m_val)
        if len(x) < 2:
            return 0.0
        dists = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        max_dists = np.max(dists, axis=2)
        C = np.sum(max_dists <= r, axis=1) / (len(x) - 1)
        return np.mean(np.log(C[C > 0]))
    
    apen = abs(_phi(m) - _phi(m+1))
    return apen if not np.isnan(apen) else 0.0

def correlation_dimension(traj, emb_dim=3, num_points=1000):
    """Dimensión de correlación (versión optimizada)"""
    traj = np.asarray(traj)
    if len(traj) < num_points:
        return 0.0
    
    # Reconstrucción del atractor
    attractor = np.lib.stride_tricks.sliding_window_view(traj, window_shape=emb_dim)
    if len(attractor) < 100:
        return 0.0
    
    # Muestreo aleatorio para eficiencia
    sample_indices = np.random.choice(len(attractor), size=min(1000, len(attractor)), replace=False)
    attractor = attractor[sample_indices]
    
    # Cálculo de distancias por pares
    dists = np.linalg.norm(attractor[:, np.newaxis] - attractor[np.newaxis, :], axis=2)
    upper_tri = np.triu_indices_from(dists, k=1)
    dists = dists[upper_tri]
    
    if len(dists) < 100:
        return 0.0
    
    # Rangos para el histograma
    r_vals = np.logspace(-4, 0, 20)
    C_r = np.array([np.sum(dists < r) / len(dists) for r in r_vals])
    
    # Regresión lineal en la región lineal
    valid = (C_r > 0) & (r_vals > 0)
    if np.sum(valid) < 5:
        return 0.0
    
    coeffs = np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)
    return coeffs[0]

# ===== SIMULACIÓN MONTE CARLO CON PERSISTENCIA =====
start_sim, results, start_time = initialize_simulation()

# Si ya hemos completado todas las simulaciones
if start_sim >= n_sim:
    print("¡Todas las simulaciones ya están completas!")
    print(f"Resultados guardados en: karmic_vector_stochastic_results.npy")
    exit()

# Calcular tiempo estimado restante
if start_sim > 0:
    elapsed = time.time() - start_time
    time_per_sim = elapsed / start_sim
    remaining = time_per_sim * (n_sim - start_sim)
    print(f"Tiempo estimado restante: {remaining/3600:.2f} horas")

# Bucle principal de simulación
try:
    for sim in tqdm(range(start_sim, n_sim), initial=start_sim, total=n_sim, 
                    desc="Simulaciones Kármicas", unit="sim", ncols=100):
        X = X0.copy()
        buffers = {i: [X0[i, 2], X0[i, 2], X0[i, 2]] for i in range(3)}
        metrics = np.zeros((3, 7))  # [max_I, t_max, area_I, n_picos, Lyapunov, ApEn, DimCorr]
        full_trajectory = np.zeros((int(t_max/dt), 3))  # Almacena I_i en cada paso
        
        for step in range(int(t_max/dt)):
            t = step * dt
            S, E, I, R = X.T
            
            # 1. Fuerza de activación (vectorizada)
            fuerza_activacion = alpha @ I
            
            # 2. Generación de ruidos vectorizada
            noise_beta = np.random.normal(0, np.sqrt(noise_params['D_beta']))
            noise_zeta = laplace.rvs(scale=noise_params['b_zeta'])
            noise_eta = 2 * beta(noise_params['a_eta'], noise_params['a_eta']).rvs() - 1
            noise_xi = poisson.rvs(noise_params['lambda_xi']) - noise_params['lambda_xi']
            
            # 3. Cálculo de derivadas (ecuaciones vectorizadas)
            dS = ( (omega + noise_xi) * f_proteccion(I, kappa) * R - 
                   S * (fuerza_activacion + noise_beta) ) * dt
            
            dE = ( S * (fuerza_activacion + noise_beta) - 
                   (sigma + noise_zeta) * E ) * dt
            
            dI = ( (sigma + noise_zeta) * E - 
                   (gamma + noise_eta) * I ) * dt
            
            dR = ( (gamma + noise_eta) * g_aprendizaje(I, lambda_g) * I - 
                   (omega + noise_xi) * R ) * dt
            
            # 4. Actualización con clip para no negatividad
            X = np.clip(X + np.vstack([dS, dE, dI, dR]).T, 0, None)
            
            # Almacenar trayectoria completa
            full_trajectory[step] = X[:,2]
            
            # 5. Cálculo de métricas básicas
            for i in range(3):
                # Máxima manifestación y tiempo
                if X[i, 2] > metrics[i, 0]:
                    metrics[i, 0] = X[i, 2]
                    metrics[i, 1] = t
                
                # Experiencia acumulada (área bajo curva)
                metrics[i, 2] += X[i, 2] * dt
                
                # Detección de picos (usando buffer de 3 puntos)
                buffers[i].pop(0)
                buffers[i].append(X[i, 2])
                if step > 1 and (buffers[i][1] > buffers[i][0] and buffers[i][1] > buffers[i][2]):
                    metrics[i, 3] += 1
        
        # 6. Cálculo de indicadores de caos (versión robusta)
        for i in range(3):
            traj = full_trajectory[:,i]
            
            # Exponente de Lyapunov
            metrics[i, 4] = lyapunov_estimate(traj)
            
            # Entropía Aproximada
            metrics[i, 5] = approximate_entropy(traj)
            
            # Dimensión de Correlación
            metrics[i, 6] = correlation_dimension(traj)
        
        results[sim] = metrics
        
        # Guardado parcial periódico
        if sim % partial_save_interval == 0:
            partial_file = save_partial_results(sim+1, results)
            # Análisis rápido
            current_lyapunov = results[:sim+1, :, 4]
            print(f"\nGuardado parcial {partial_file}")
            print(f"Lyapunov range: [{np.min(current_lyapunov):.3f}, {np.max(current_lyapunov):.3f}]")
            
        # Guardado de checkpoint principal
        if sim % checkpoint_interval == 0 or sim == n_sim - 1:
            save_checkpoint(sim + 1, results, start_time)
    
    # Guardar resultados finales
    np.save("karmic_vector_stochastic_results.npy", results)
    
    # Eliminar checkpoint si la simulación se completó
    if os.path.exists('karmic_checkpoint.pkl'):
        os.remove('karmic_checkpoint.pkl')
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    print(f"\n¡Simulaciones completadas en {total_time/3600:.2f} horas!")
    print("Resultados guardados en: karmic_vector_stochastic_results.npy")

except KeyboardInterrupt:
    print("\nSimulación interrumpida por el usuario. Guardando estado actual...")
    save_checkpoint(sim, results, start_time)
    partial_file = save_partial_results(sim, results)
    print(f"Estado guardado. Puede reanudar desde la simulación {sim}")
    print(f"Resultados parciales guardados en {partial_file}")
except Exception as e:
    print(f"\nError durante la simulación: {str(e)}")
    print("Guardando estado actual para poder reanudar...")
    save_checkpoint(sim, results, start_time)
    partial_file = save_partial_results(sim, results)
    print(f"Estado guardado en la simulación {sim}. Puede reanudar desde este punto.")
    print(f"Resultados parciales guardados en {partial_file}")