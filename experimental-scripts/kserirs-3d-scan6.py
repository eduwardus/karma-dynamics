# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:26:35 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Simulación Kármica con Parámetros Optimizados para Ciclos
"""

import numpy as np
import os
import pickle
from scipy.stats import laplace, beta, poisson
from tqdm import tqdm
import time
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# ===== PARÁMETROS OPTIMIZADOS PARA CICLOS =====
t_max = 500  # Tiempo de simulación
dt = 0.05    # Paso temporal
n_sim = 5000  # Número de simulaciones
checkpoint_interval = 100
partial_save_interval = 10

# Condiciones iniciales
X0 = np.array([[0.75, 0.25, 0.00, 0.00],  # avidyā 
               [0.65, 0.35, 0.00, 0.00],  # rāga
               [0.55, 0.45, 0.00, 0.00]]) # dveṣa

# Parámetros clave ajustados para ciclos:
sigma = np.array([0.9, 0.9, 0.9])      # Tasas de maduración
gamma = np.array([0.15, 0.15, 0.15])   # Tasas de purga reducidas
omega = np.array([0.20, 0.20, 0.20])   # Tasas de recaída aumentadas
kappa = 0.75                           # Factor de protección
lambda_g = 0.01                        # Factor de aprendizaje reducido

# Matriz de acoplamiento para ciclos:
alpha = np.array([
    [1.5, 1.2, 0.3],  
    [1.2, 1.5, 0.4],  
    [0.3, 0.4, 1.3]   
])

# Ruidos equilibrados:
noise_params = {
    'D_beta': np.array([0.3, 0.3, 0.3]),
    'b_zeta': np.array([0.4, 0.4, 0.4]),
    'a_eta': np.array([1.8, 1.8, 1.8]),
    'lambda_xi': np.array([0.3, 0.3, 0.3])
}

# ===== FUNCIONES AUXILIARES (ECUACIONES ORIGINALES) =====
def f_proteccion(I, kappa):
    """Función de protección por experiencia (original)"""
    return 1 - kappa * np.clip(I, 0, 1/kappa)  # Asegura no negatividad

def g_aprendizaje(I, lambda_g):
    """Función de aprendizaje kármico (original)"""
    return 1 - np.exp(-lambda_g * np.abs(I))

def load_checkpoint():
    """Cargar estado anterior si existe"""
    if os.path.exists('karmic_checkpoint.pkl'):
        with open('karmic_checkpoint.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def save_checkpoint(sim_count, results, start_time, behavior_counters):
    """Guardar estado actual para poder reanudar"""
    checkpoint = {
        'sim_count': sim_count,
        'results': results,
        'start_time': start_time,
        'behavior_counters': behavior_counters
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
        return (checkpoint['sim_count'], checkpoint['results'], 
                checkpoint['start_time'], checkpoint.get('behavior_counters', {}))
    
    # Crear estructura vacía para resultados
    results = np.zeros((n_sim, 3, 7))  # 7 indicadores por componente
    
    # Contadores para diferentes comportamientos
    behavior_counters = {
        'chaotic': 0,
        'cyclic_simple': 0,
        'cyclic_complex': 0,
        'extinct': 0
    }
    
    return 0, results, time.time(), behavior_counters

# ===== FUNCIONES DE ANÁLISIS DE CAOS (MANTENIDAS) =====
def lyapunov_estimate(trajectory, min_points=300):
    """Estimación mejorada de exponente de Lyapunov"""
    if len(trajectory) < min_points:
        return 0.0
    
    # Suavizado de la trayectoria para reducir ruido de alta frecuencia
    window_size = 5
    if len(trajectory) > window_size:
        smoothed = np.convolve(trajectory, np.ones(window_size)/window_size, mode='valid')
    else:
        smoothed = trajectory
    
    # Cálculo de divergencia en ventanas deslizantes
    divergences = []
    window_size = 100
    
    for i in range(len(smoothed) - window_size):
        ref = smoothed[i]
        segment = smoothed[i+1:i+window_size]
        dists = np.abs(segment - ref)
        valid_dists = dists[dists > 1e-8]
        
        if len(valid_dists) > 10:  # Suficientes puntos válidos
            log_dists = np.log(valid_dists)
            times = np.arange(1, len(valid_dists)+1) * dt
            slope, _ = np.polyfit(times, log_dists, 1)
            divergences.append(slope)
    
    return np.mean(divergences) if divergences else 0.0

def approximate_entropy(U, m=2, r=0.2, sample_size=200):
    """Entropía aproximada optimizada para memoria"""
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
    
    # Crear ventanas deslizantes para m y m+1
    x_m = np.lib.stride_tricks.sliding_window_view(U_norm, window_shape=m)
    x_m1 = np.lib.stride_tricks.sliding_window_view(U_norm, window_shape=m+1)
    
    # Muestreo aleatorio para eficiencia
    n_samples_m = min(sample_size, len(x_m))
    n_samples_m1 = min(sample_size, len(x_m1))
    
    sample_indices_m = np.random.choice(len(x_m), size=n_samples_m, replace=False)
    sample_indices_m1 = np.random.choice(len(x_m1), size=n_samples_m1, replace=False)
    
    def _phi(x, sample_indices):
        """Cálculo eficiente de phi con muestreo"""
        total = 0.0
        count = 0
        
        for i in sample_indices:
            # Distancia máxima entre el vector i y todos los demás
            dists = np.max(np.abs(x - x[i]), axis=1)
            # Contar vectores dentro del radio r (excluyendo el propio vector)
            within_r = np.sum(dists <= r) - 1  # -1 para excluir el vector i mismo
            
            if within_r > 0:
                C_i = within_r / (len(x) - 1)
                total += np.log(C_i)
                count += 1
        
        return total / count if count > 0 else 0.0
    
    phi_m = _phi(x_m, sample_indices_m)
    phi_m1 = _phi(x_m1, sample_indices_m1)
    
    apen = abs(phi_m - phi_m1)
    return apen if not np.isnan(apen) else 0.0

def correlation_dimension(traj, emb_dim=3, sample_size=1000):
    """Dimensión de correlación optimizada"""
    traj = np.asarray(traj)
    if len(traj) < 100:
        return 0.0
    
    # Reconstrucción del atractor
    attractor = np.lib.stride_tricks.sliding_window_view(traj, window_shape=emb_dim)
    if len(attractor) < 100:
        return 0.0
    
    # Muestreo aleatorio para eficiencia
    sample_indices = np.random.choice(len(attractor), size=min(sample_size, len(attractor)), replace=False)
    attractor = attractor[sample_indices]
    
    # Cálculo de distancias por pares en bloques
    n = len(attractor)
    block_size = 500  # Tamaño de bloque para procesamiento por partes
    r_vals = np.logspace(-4, 0, 20)
    C_r = np.zeros(len(r_vals))
    
    # Procesar en bloques para evitar matrices grandes
    for i in range(0, n, block_size):
        block_end = min(i + block_size, n)
        block = attractor[i:block_end]
        
        # Calcular distancias para este bloque contra todo el conjunto
        dists_block = np.linalg.norm(attractor[:, np.newaxis] - block[np.newaxis, :], axis=2)
        
        # Acumular conteos para cada r
        for idx, r in enumerate(r_vals):
            C_r[idx] += np.sum(dists_block < r) - np.sum(dists_block[i:block_end, i:block_end] < r)
    
    # Excluir auto-distancias
    total_pairs = n * (n - 1) / 2
    C_r = C_r / total_pairs
    
    # Regresión lineal en la región lineal
    valid = (C_r > 0) & (r_vals > 0)
    if np.sum(valid) < 5:
        return 0.0
    
    coeffs = np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)
    return coeffs[0]

# ===== DETECCIÓN DE COMPORTAMIENTOS =====
def detect_behavior_type(trajectory, metrics):
    """Detecta el tipo de comportamiento de la trayectoria"""
    # 1. Comportamiento extinto: todos los componentes terminan cerca de cero
    final_values = trajectory[-1, :]
    if np.all(final_values < 0.001):
        return 'extinct'
    
    # 2. Comportamiento caótico
    max_lyapunov = np.max(metrics[:, 4])
    if max_lyapunov > 0.2:
        return 'chaotic'
    
    # 3. Comportamiento cíclico simple
    n_peaks = metrics[:, 3]  # Número de picos por componente
    peaks_per_time = np.mean(n_peaks) / (t_max / 10)  # Normalizado
    
    # Criterios para ciclo simple: periodicidad y amplitud sostenida
    if peaks_per_time < 2.0 and np.all(n_peaks > 3) and np.all(trajectory[-100:,:].mean(axis=0) > 0.05):
        # Análisis FFT para periodicidad
        is_periodic = True
        for i in range(3):
            yf = fft(trajectory[:, i])
            xf = fftfreq(len(trajectory), dt)[:len(trajectory)//2]
            power_spectrum = 2.0/len(trajectory) * np.abs(yf[0:len(trajectory)//2])
            significant_peaks = find_peaks(power_spectrum, height=np.max(power_spectrum)*0.2, distance=10)[0]
            if len(significant_peaks) > 3:  # Demasiados picos -> no periódico simple
                is_periodic = False
                break
        if is_periodic:
            return 'cyclic_simple'
    
    # 4. Comportamiento cíclico complejo
    if peaks_per_time > 1.5 and np.any(n_peaks > 10) and np.all(trajectory[-100:,:].mean(axis=0) > 0.05):
        return 'cyclic_complex'
    
    return 'other'

# ===== SIMULACIÓN MONTE CARLO =====
start_sim, results, start_time, behavior_counters = initialize_simulation()

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
            
            # 1. Fuerza de activación (vectorizada) - ECUACIÓN ORIGINAL
            fuerza_activacion = alpha @ I
            
            # 2. Generación de ruidos vectorizada
            noise_beta = np.random.normal(0, np.sqrt(noise_params['D_beta']))
            noise_zeta = laplace.rvs(scale=noise_params['b_zeta'])
            noise_eta = 2 * beta(noise_params['a_eta'], noise_params['a_eta']).rvs() - 1
            noise_xi = poisson.rvs(noise_params['lambda_xi']) - noise_params['lambda_xi']
            
            # 3. Cálculo de derivadas (ECUACIONES ORIGINALES)
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
        
        # 6. Cálculo de indicadores de caos
        for i in range(3):
            traj = full_trajectory[:,i]
            metrics[i, 4] = lyapunov_estimate(traj)   # Exponente de Lyapunov
            metrics[i, 5] = approximate_entropy(traj)  # Entropía Aproximada
            metrics[i, 6] = correlation_dimension(traj) # Dimensión de Correlación
        
        results[sim] = metrics
        
        # 7. Detección de tipo de comportamiento
        behavior_type = detect_behavior_type(full_trajectory, metrics)
        
        # Actualizar contadores y guardar trayectorias representativas
        if behavior_type in behavior_counters:
            behavior_counters[behavior_type] += 1
            counter = behavior_counters[behavior_type]
            
            # Guardar hasta 5 trayectorias de cada tipo
            if behavior_type != 'other' and counter <= 5:
                filename = f"{behavior_type}_trajectory_sim_{sim}.npy"
                np.save(filename, full_trajectory)
                print(f"\nTrayectoria {behavior_type} detectada en simulación {sim} (contador: {counter})")
        
        # Guardado parcial periódico
        if sim % partial_save_interval == 0:
            partial_file = save_partial_results(sim+1, results)
            
        # Guardado de checkpoint principal
        if sim % checkpoint_interval == 0 or sim == n_sim - 1:
            save_checkpoint(sim + 1, results, start_time, behavior_counters)
    
    # Guardar resultados finales
    np.save("karmic_vector_stochastic_results.npy", results)
    
    # Eliminar checkpoint si la simulación se completó
    if os.path.exists('karmic_checkpoint.pkl'):
        os.remove('karmic_checkpoint.pkl')
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    print(f"\n¡Simulaciones completadas en {total_time/3600:.2f} horas!")
    print("Resultados guardados en: karmic_vector_stochastic_results.npy")
    print("\nResumen de comportamientos detectados:")
    for behavior, count in behavior_counters.items():
        print(f" - {behavior}: {count} trayectorias representativas guardadas")

except KeyboardInterrupt:
    print("\nSimulación interrumpida por el usuario. Guardando estado actual...")
    save_checkpoint(sim, results, start_time, behavior_counters)
    partial_file = save_partial_results(sim, results)
    print(f"Estado guardado. Puede reanudar desde la simulación {sim}")
    print(f"Resultados parciales guardados en {partial_file}")
except Exception as e:
    print(f"\nError durante la simulación: {str(e)}")
    import traceback
    traceback.print_exc()
    print("Guardando estado actual para poder reanudar...")
    save_checkpoint(sim, results, start_time, behavior_counters)
    partial_file = save_partial_results(sim, results)
    print(f"Estado guardado en la simulación {sim}. Puede reanudar desde este punto.")
    print(f"Resultados parciales guardados en {partial_file}")