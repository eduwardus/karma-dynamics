# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:00:05 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Análisis de Modelos Dinámicos y Atractores
@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, periodogram, correlate
from scipy.stats import linregress
import pandas as pd
import time
import os
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import json
from mpl_toolkits.mplot3d import Axes3D

# Configuración inicial
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =================================================================
# MODELO DINÁMICO - TRES RAÍCES (IGNORANCIA, APEGO, AVERSIÓN)
# =================================================================
def three_roots_model(t, y, params):
    """
    Modelo dinámico de las tres raíces del sufrimiento:
    - I: Ignorancia
    - A: Apego
    - V: Aversión
    
    Ecuaciones:
    dI/dt = αI*I + βIA*A*V - γI*w*I
    dA/dt = αA*A + βAV*V*I - γA*w*A
    dV/dt = αV*V + βVI*I*A - γV*w*V
    """
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

def explosion_event(t, y, params):
    """Detiene la simulación si los valores exceden un umbral"""
    return np.max(y) - 10  # Umbral bajo para detectar divergencia temprana
explosion_event.terminal = True
explosion_event.direction = 1

# =================================================================
# FUNCIONES AUXILIARES PARA ANÁLISIS
# =================================================================
def calculate_autocorrelation(signal, max_lag=500):
    """Calcula la autocorrelación normalizada de una señal"""
    n = len(signal)
    if n < 10:
        return np.zeros(max_lag)
    
    autocorr = correlate(signal, signal, mode='full', method='auto')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag]
    autocorr /= autocorr[0]  # Normalizar
    return autocorr

def calculate_lyapunov_exponent(t, y, params, delta=1e-7):
    """
    Calcula un exponente de Lyapunov aproximado para señales pequeñas
    usando una perturbación proporcional a la amplitud del sistema
    """
    if len(t) < 100:
        return 0, 0
    
    # Crear perturbación proporcional a la amplitud del sistema
    perturbation = delta * (np.max(y, axis=1) - np.min(y, axis=1))
    perturbation[perturbation == 0] = delta  # Evitar perturbación cero
    
    y0_perturbed = y[:, 0] + perturbation
    
    # Simular trayectoria perturbada
    sol_perturbed = solve_ivp(
        three_roots_model,
        [t[0], t[-1]],
        y0_perturbed,
        args=(params,),
        t_eval=t,
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not sol_perturbed.success or len(sol_perturbed.y) == 0:
        return 0, 0
    
    # Calcular divergencia entre trayectorias
    dist = np.linalg.norm(y - sol_perturbed.y, axis=0)
    valid_idx = (dist > 1e-10) & (dist < 1e5)  # Filtrar valores extremos
    
    if np.sum(valid_idx) < 20:
        return 0, 0
    
    dist = dist[valid_idx]
    t_valid = t[valid_idx]
    
    # Regresión lineal en escala logarítmica
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r_value, p_value, std_err = linregress(t_valid, np.log(dist))
    
    # Filtrar resultados no físicos
    if abs(slope) > 10 or r_value < 0.5:
        return 0, 0
    
    return slope, r_value**2

def calculate_entropy(signal):
    """Calcula la entropía aproximada de una señal usando histograma"""
    hist, bin_edges = np.histogram(signal, bins=50, density=True)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # Eliminar bins vacíos
    entropy = -np.sum(prob * np.log(prob))
    return entropy

# =================================================================
# CLASIFICACIÓN AVANZADA DE ATRACTORES
# =================================================================
def classify_attractor(t, y, params):
    """
    Clasifica el tipo de atractor usando múltiples técnicas:
    - Análisis de estabilidad
    - Exponente de Lyapunov
    - Análisis espectral
    - Autocorrelación
    - Detección de periodicidad
    """
    if len(t) < 100 or y.size == 0:
        return "Divergencia", {}
    
    I, A, V = y
    metadata = {
        'max_value': np.max(y),
        'min_value': np.min(y),
        'mean_value': np.mean(y, axis=1),
        'std_value': np.std(y, axis=1)
    }
    
    # Detectar divergencia (umbral bajo para señales pequeñas)
    if np.max(y) > 10 or np.any(np.isnan(y)):
        return "Divergencia", metadata
    
    # Usar los últimos 3/4 de la simulación (estado estacionario)
    start_idx = len(t) // 4
    if start_idx >= len(t):
        return "Transitorio largo", metadata
    
    I_ss = I[start_idx:]
    A_ss = A[start_idx:]
    V_ss = V[start_idx:]
    t_ss = t[start_idx:]
    
    # Calcular amplitud para ajustar umbrales dinámicamente
    amplitude = np.max(V_ss) - np.min(V_ss)
    if amplitude == 0:
        amplitude = 1e-5  # Evitar división por cero
    
    # 1. Detectar punto fijo con criterio relativo a la amplitud
    std_dev = np.std(V_ss)
    mean_value = np.mean(V_ss)
    fixed_point_threshold = max(0.01 * mean_value, 0.005 * amplitude)
    
    if std_dev < fixed_point_threshold:
        center = np.mean(y[:, start_idx:], axis=1)
        metadata['center'] = center
        return "Punto fijo estable", metadata
    
    # 2. Calcular exponente de Lyapunov
    lyap_exp, lyap_r2 = calculate_lyapunov_exponent(t, y, params, delta=1e-7)
    metadata['lyapunov'] = lyap_exp
    metadata['lyapunov_r2'] = lyap_r2
    
    # 3. Análisis espectral mejorado
    spectral_entropy = 0
    if len(t_ss) > 10 and (t_ss[1] - t_ss[0]) > 0:
        fs = 1/(t_ss[1]-t_ss[0])  # Frecuencia de muestreo
        f, Pxx = periodogram(V_ss, fs=fs, scaling='density', nfft=2048)
        
        if len(Pxx) > 0:
            # Calcular entropía espectral normalizada
            Pxx_normalized = Pxx / np.sum(Pxx)
            spectral_entropy = -np.sum(Pxx_normalized * np.log(Pxx_normalized + 1e-10))
            metadata['spectral_entropy'] = spectral_entropy
            
            # Detección de frecuencias dominantes
            noise_floor = np.percentile(Pxx, 70)
            dominant_peaks = Pxx > 5 * noise_floor
            if np.any(dominant_peaks):
                dominant_freq = f[np.argmax(Pxx)]
                n_spectral_peaks = np.sum(dominant_peaks)
                metadata['dominant_freq'] = dominant_freq
                metadata['n_spectral_peaks'] = n_spectral_peaks
            else:
                metadata['dominant_freq'] = 0
                metadata['n_spectral_peaks'] = 0
        else:
            metadata['spectral_entropy'] = 0
            metadata['dominant_freq'] = 0
            metadata['n_spectral_peaks'] = 0
    else:
        metadata['spectral_entropy'] = 0
        metadata['dominant_freq'] = 0
        metadata['n_spectral_peaks'] = 0
    
    # 4. Autocorrelación
    autocorr = calculate_autocorrelation(V_ss)
    if len(autocorr) > 20:
        autocorr_decay = autocorr[10] if len(autocorr) > 10 else 1
        autocorr_periodic = np.max(autocorr[10:20]) > 0.7  # Periodicidad
        metadata['autocorr_decay'] = autocorr_decay
        metadata['autocorr_periodic'] = autocorr_periodic
    else:
        autocorr_decay = 1
        autocorr_periodic = False
    
    # 5. Entropía de la señal
    entropy = calculate_entropy(V_ss)
    metadata['entropy'] = entropy
    
    # 6. Detección de picos con umbral relativo
    peak_height = np.mean(V_ss) + 0.2 * amplitude
    n_peaks = 0
    try:
        peaks, properties = find_peaks(
            V_ss, 
            height=peak_height, 
            prominence=0.1 * amplitude, 
            distance=max(3, int(0.05 * len(V_ss))))
        n_peaks = len(peaks)
        
        if n_peaks > 2:
            periods = np.diff(t_ss[peaks])
            period_mean = np.mean(periods)
            period_std = np.std(periods)
            period_cv = period_std / period_mean if period_mean > 0 else 0
            
            metadata.update({
                'n_peaks': n_peaks,
                'period_mean': period_mean,
                'period_std': period_std,
                'period_cv': period_cv
            })
    except Exception as e:
        metadata['peak_error'] = str(e)
    
    # =================================================================
    # CLASIFICACIÓN HEURÍSTICA MEJORADA
    # =================================================================
    # Umbrales adaptativos basados en la amplitud
    chaos_threshold = max(0.01, 0.05 * amplitude)
    entropy_threshold = max(1.0, 0.5 * np.log(len(V_ss)))
    
    # 1. Punto fijo inestable (oscilaciones crecientes)
    if n_peaks < 2 and metadata.get('n_spectral_peaks', 0) < 2 and lyap_exp > 0.01:
        return "Punto fijo inestable", metadata
    
    # 2. Ciclo límite (criterio más flexible)
    if n_peaks >= 3 and metadata.get('period_cv', 1) < 0.25 and autocorr_periodic:
        return "Ciclo límite", metadata
    
    # 3. Caos determinista (criterio menos estricto)
    if (lyap_exp > chaos_threshold and lyap_r2 > 0.6 and 
        spectral_entropy > entropy_threshold and autocorr_decay < 0.5):
        return "Caos determinista", metadata
    
    # 4. Comportamiento cuasiperiódico
    if (metadata.get('n_spectral_peaks', 0) >= 3 and 
        spectral_entropy > entropy_threshold * 0.8 and 
        spectral_entropy < entropy_threshold * 1.5 and 
        autocorr_decay > 0.6):
        return "Cuasiperiódico", metadata
    
    # 5. Comportamiento periódico simple
    if n_peaks >= 3 and metadata.get('period_cv', 1) < 0.1:
        return "Periódico", metadata
    
    # 6. Comportamiento transitorio largo
    if len(t_ss) < len(t)//2:  # No se alcanzó estado estacionario
        return "Transitorio largo", metadata
    
    # 7. Si no cumple otros criterios pero tiene oscilaciones
    if n_peaks >= 3:
        return "Oscilatorio complejo", metadata
    
    # Comportamiento indefinido
    return "Comportamiento complejo", metadata

# =================================================================
# SIMULACIÓN MONTE CARLO MEJORADA
# =================================================================
def monte_carlo_simulation(n_simulations=100, t_span=(0, 1000)):
    """
    Realiza un barrido aleatorio de parámetros y condiciones iniciales
    para explorar el espacio de fases del modelo
    """
    results = []
    
    # Rangos de parámetros optimizados para reducir divergencias
    param_ranges = {
        'alpha_I': (0.0001, 0.01),   # Tasas de crecimiento muy bajas
        'alpha_A': (0.0001, 0.01),
        'alpha_V': (0.0001, 0.01),
        'beta_IA': (0.05, 0.5),       # Interacciones débiles
        'beta_AV': (0.05, 0.5),
        'beta_VI': (0.05, 0.5),
        'gamma_I': (0.1, 0.5),        # Disipación moderada
        'gamma_A': (0.1, 0.5),
        'gamma_V': (0.1, 0.5),
        'w': (0.1, 0.9)               # Amplio rango de sabiduría
    }
    
    for i in tqdm(range(n_simulations), desc="Simulando"):
        # Generar parámetros aleatorios con distribución uniforme
        params = {key: np.random.uniform(low, high) for key, (low, high) in param_ranges.items()}
        
        # Generar condición inicial con distribución Dirichlet (normalizada)
        y0 = np.random.dirichlet([1, 1, 1]) * np.random.uniform(0.001, 0.01)
        
        try:
            # Simulación completa con parámetros de tolerancia ajustados
            sol = solve_ivp(
                three_roots_model,
                t_span,
                y0,
                args=(params,),
                method='LSODA',
                events=[explosion_event],
                max_step=0.5,
                t_eval=np.linspace(t_span[0], t_span[1], 5000),  # Más puntos para mejor resolución
                rtol=1e-8,
                atol=1e-10
            )
            
            # Si explotó durante la simulación
            if sol.t_events[0].size > 0:
                results.append({
                    'params': params,
                    'y0': y0,
                    'attractor_type': 'Divergencia',
                    'simulation_time': sol.t,
                    'simulation_y': sol.y
                })
                continue
            
            # Clasificar atractor
            attractor_type, metadata = classify_attractor(sol.t, sol.y, params)
            
            # Guardar resultados
            result = {
                'params': params,
                'y0': y0,
                'attractor_type': attractor_type,
                'simulation_time': sol.t,
                'simulation_y': sol.y
            }
            result.update(metadata)
            results.append(result)
            
        except Exception as e:
            # Guardar error con más información
            results.append({
                'params': params,
                'y0': y0,
                'attractor_type': f'Error: {str(e)}',
                'simulation_time': np.array([]),
                'simulation_y': np.array([])
            })
    
    return results

# =================================================================
# ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS
# =================================================================
def analyze_results(results):
    """Analiza y visualiza los resultados del Monte Carlo"""
    if not results:
        print("No hay resultados para analizar")
        return None
    
    # Convertir a DataFrame
    df = pd.DataFrame(results)
    
    # Expandir los parámetros en columnas separadas
    if 'params' in df.columns and len(df) > 0:
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
    
    # Estadísticas básicas
    print("\nDistribución de tipos de atractores:")
    print(df['attractor_type'].value_counts())
    
    # Visualización
    plot_attractor_distribution(df)
    
    # Solo intentar otras visualizaciones si hay suficientes datos
    if len(df) > 10:
        plot_parameter_space(df)
        if 'lyapunov' in df.columns:
            plot_lyapunov_distribution(df)
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("results", exist_ok=True)
    filename = f"results/monte_carlo_results_{timestamp}.pkl"
    df.to_pickle(filename)
    
    # Guardar también en JSON para fácil inspección
    json_filename = f"results/monte_carlo_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        f.write(df.to_json(orient='records', indent=4))
    
    print(f"\nResultados guardados en {filename} y {json_filename}")
    
    return df

def plot_attractor_distribution(df):
    """Visualiza la distribución de tipos de atractores"""
    if len(df) == 0:
        print("No hay datos para visualizar distribución")
        return
    
    type_counts = df['attractor_type'].value_counts()
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(type_counts.index, type_counts.values, color='skyblue')
    
    # Añadir etiquetas
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height} ({height/len(df)*100:.1f}%)',
                 ha='center', va='bottom', rotation=45, fontsize=9)
    
    plt.title('Distribución de Tipos de Atractores', fontsize=16)
    plt.xlabel('Tipo de Atractor', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/attractor_distribution.png', dpi=150)
    plt.show()

def plot_parameter_space(df):
    """Visualiza el espacio de parámetros usando PCA"""
    if len(df) < 2:
        return
    
    # Preparar datos para PCA
    param_cols = ['alpha_I', 'alpha_A', 'alpha_V', 
                 'beta_IA', 'beta_AV', 'beta_VI',
                 'gamma_I', 'gamma_A', 'gamma_V', 'w']
    
    # Verificar que existen las columnas
    missing_cols = [col for col in param_cols if col not in df.columns]
    if missing_cols:
        print(f"Advertencia: Columnas faltantes para PCA: {missing_cols}")
        return
    
    # Filtrar casos válidos
    valid_df = df.dropna(subset=param_cols)
    valid_df = valid_df[~valid_df['attractor_type'].str.startswith(('Error', 'Divergencia'))]
    
    if len(valid_df) < 10:
        print("No hay suficientes datos válidos para PCA")
        return
    
    # Matriz de parámetros
    X = valid_df[param_cols].values
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Varianza explicada por componentes: {pca.explained_variance_ratio_}")
    
    # Colores por tipo de atractor
    unique_types = valid_df['attractor_type'].unique()
    color_map = cm.get_cmap('tab20', len(unique_types))
    
    plt.figure(figsize=(14, 10))
    for i, atype in enumerate(unique_types):
        idx = valid_df['attractor_type'] == atype
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                    color=color_map(i), 
                    label=atype, 
                    alpha=0.7,
                    s=60)
    
    plt.title('Espacio de Parámetros (PCA)', fontsize=16)
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/parameter_space_pca.png', dpi=150)
    plt.show()

def plot_lyapunov_distribution(df):
    """Visualiza la distribución de exponentes de Lyapunov"""
    if 'lyapunov' not in df.columns:
        return
    
    # Filtrar valores válidos
    valid_df = df[df['lyapunov'].notna() & (df['lyapunov_r2'] > 0.5)]
    if len(valid_df) == 0:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Histograma general
    plt.hist(valid_df['lyapunov'], bins=30, alpha=0.7, color='blue', label='Todos')
    
    # Resaltar caos
    if 'attractor_type' in valid_df.columns:
        chaos_df = valid_df[valid_df['attractor_type'] == 'Caos determinista']
        if not chaos_df.empty:
            plt.hist(chaos_df['lyapunov'], bins=30, alpha=0.7, color='red', label='Caos')
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.5, label='Estabilidad')
    plt.title('Distribución de Exponentes de Lyapunov', fontsize=16)
    plt.xlabel('Exponente de Lyapunov', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lyapunov_distribution.png', dpi=150)
    plt.show()

# =================================================================
# VISUALIZACIÓN DE SIMULACIONES INDIVIDUALES
# =================================================================
def plot_simulation_result(result):
    """Visualiza los resultados de una simulación individual"""
    t = result['simulation_time']
    y = result['simulation_y']
    attractor_type = result['attractor_type']
    params = result.get('params', {})
    
    if len(t) == 0 or y.size == 0:
        print("No hay datos de simulación")
        return
    
    I, A, V = y
    
    # Serie temporal
    plt.figure(figsize=(14, 8))
    plt.plot(t, I, 'b-', alpha=0.7, label='Ignorancia (I)')
    plt.plot(t, A, 'g-', alpha=0.7, label='Apego (A)')
    plt.plot(t, V, 'r-', alpha=0.7, label='Aversión (V)')
    
    # Línea de estado estacionario
    if len(t) > 4:
        start_idx = len(t) // 4
        plt.axvline(t[start_idx], color='gray', linestyle='--', alpha=0.5, label='Inicio estado est.')
    
    # Título con parámetros clave
    param_str = "αI={:.4f} αA={:.4f} αV={:.4f} βIA={:.4f} βAV={:.4f} βVI={:.4f} γI={:.4f} γA={:.4f} γV={:.4f} w={:.4f}".format(
        params.get('alpha_I', 0), params.get('alpha_A', 0), params.get('alpha_V', 0),
        params.get('beta_IA', 0), params.get('beta_AV', 0), params.get('beta_VI', 0),
        params.get('gamma_I', 0), params.get('gamma_A', 0), params.get('gamma_V', 0),
        params.get('w', 0)
    )
    
    plt.title(f'Dinámica Temporal - {attractor_type}\n{param_str}', fontsize=14)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Intensidad', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    safe_attractor_name = attractor_type.replace(' ', '_').replace(':', '').replace('/', '')
    plt.savefig(f'results/time_series_{safe_attractor_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Espacio de fases 3D
    if len(t) > 100:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(I, A, V, 'b-', alpha=0.5, linewidth=0.8)
        ax.scatter(I[0], A[0], V[0], c='green', s=80, label='Inicio')
        ax.scatter(I[-1], A[-1], V[-1], c='red', s=80, label='Fin')
        ax.set_title(f'Espacio de Fases - {attractor_type}', fontsize=16)
        ax.set_xlabel('Ignorancia (I)', fontsize=12)
        ax.set_ylabel('Apego (A)', fontsize=12)
        ax.set_zlabel('Aversión (V)', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'results/phase_space_{safe_attractor_name}.png', dpi=150)
        plt.show()
    
    # Análisis de frecuencias (si no es punto fijo)
    if attractor_type not in ["Punto fijo estable", "Punto fijo inestable"] and len(t) > 4:
        start_idx = len(t) // 4
        V_ss = V[start_idx:]
        t_ss = t[start_idx:]
        
        if len(t_ss) > 10:
            fs = 1/(t_ss[1]-t_ss[0])
            f, Pxx = periodogram(V_ss, fs=fs, scaling='density')
            
            plt.figure(figsize=(12, 6))
            plt.semilogy(f, Pxx, 'b-')
            plt.title('Espectro de Frecuencias (Aversión)', fontsize=16)
            plt.xlabel('Frecuencia', fontsize=12)
            plt.ylabel('Densidad espectral', fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'results/spectrum_{safe_attractor_name}.png', dpi=150)
            plt.show()

# =================================================================
# FUNCIÓN PARA EJECUTAR SIMULACIÓN A GRAN ESCALA
# =================================================================
def run_large_simulation(n_simulations=10000):
    """Ejecuta una simulación a gran escala y guarda resultados intermedios"""
    results = []
    batch_size = 1000
    n_batches = n_simulations // batch_size
    
    for i in range(n_batches):
        print(f"\nEjecutando lote {i+1}/{n_batches} ({batch_size} simulaciones)")
        batch_results = monte_carlo_simulation(batch_size)
        results.extend(batch_results)
        
        # Guardar resultados intermedios
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"results/batch_{i}_{timestamp}.pkl"
        pd.DataFrame(batch_results).to_pickle(filename)
        print(f"Lote guardado en {filename}")
    
    return results

# =================================================================
# EJECUCIÓN PRINCIPAL
# =================================================================
if __name__ == "__main__":
    # Crear directorio de resultados
    os.makedirs("results", exist_ok=True)
    
    # Ejecutar simulación de prueba
    print("Iniciando simulación Monte Carlo mejorada...")
    test_results = monte_carlo_simulation(n_simulations=500)
    
    # Analizar resultados
    results_df = analyze_results(test_results)
    
    # Mostrar casos interesantes
    if results_df is not None:
        print("\nCasos interesantes encontrados:")
        
        # Buscar diferentes tipos de atractores (excluyendo divergencias y errores)
        interesting_types = [t for t in results_df['attractor_type'].unique() 
                            if not t.startswith(('Divergencia', 'Error', 'Comportamiento complejo'))]
        
        for attractor_type in interesting_types:
            print(f"\nMostrando ejemplo de {attractor_type}:")
            # Filtrar casos que tengan datos de simulación
            filtered = results_df[results_df['attractor_type'] == attractor_type]
            if not filtered.empty:
                # Tomar un caso con datos de simulación no vacíos
                valid_cases = filtered[filtered['simulation_y'].apply(lambda x: x.size > 0)]
                if not valid_cases.empty:
                    example = valid_cases.iloc[0].to_dict()
                    plot_simulation_result(example)
                else:
                    print(f"No hay casos válidos para {attractor_type}")
            else:
                print(f"No hay casos de {attractor_type}")
        
        # Opción para ejecutar simulación a gran escala
        run_large = input("\n¿Ejecutar simulación a gran escala? (s/n): ")
        if run_large.lower() == 's':
            n_simulations = int(input("Número de simulaciones (ej. 10000): "))
            large_results = run_large_simulation(n_simulations)
            analyze_results(large_results)