# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 23:12:23 2025
@author: eggra
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
import warnings
import time
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

# Ignorar warnings
warnings.filterwarnings('ignore')

# Modelo de Tres Raíces Kármicas
def three_roots_model(y, params):
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# Jacobiano
def jacobian(y, params):
    I, A, V = y
    J = np.zeros((3, 3))
    J[0,0] = params['alpha_I'] - params['gamma_I']*params['w'] + params['beta_IA']*A*V
    J[0,1] = params['beta_IA']*V
    J[0,2] = params['beta_IA']*A
    J[1,0] = params['beta_AV']*V
    J[1,1] = params['alpha_A'] - params['gamma_A']*params['w'] + params['beta_AV']*V*I
    J[1,2] = params['beta_AV']*I
    J[2,0] = params['beta_VI']*A
    J[2,1] = params['beta_VI']*I
    J[2,2] = params['alpha_V'] - params['gamma_V']*params['w'] + params['beta_VI']*I*A
    return J

# Generar parámetros aleatorios
def generate_random_params():
    return {
        'alpha_I': np.random.uniform(0.01, 1.0),
        'alpha_A': np.random.uniform(0.01, 1.0),
        'alpha_V': np.random.uniform(0.01, 1.0),
        'beta_IA': np.random.uniform(0.05, 1.5),
        'beta_AV': np.random.uniform(0.05, 1.5),
        'beta_VI': np.random.uniform(0.05, 1.5),
        'gamma_I': np.random.uniform(0.1, 1.5),
        'gamma_A': np.random.uniform(0.1, 1.5),
        'gamma_V': np.random.uniform(0.1, 1.5),
        'w': np.random.uniform(0.05, 1.2)
    }

# Búsqueda extendida
def extended_saddle_search(n_param_sets=1500, n_samples_per_set=600, 
                          search_range=(-1.0, 3.0)):
    all_saddle_points = []
    all_eigenvalues = []
    all_eigenvectors = []
    all_params = []
    
    print(f"Búsqueda exhaustiva: {n_param_sets}×{n_samples_per_set} = {n_param_sets*n_samples_per_set:,} puntos")
    start_time = time.time()
    
    for _ in tqdm(range(n_param_sets), desc="Conjuntos paramétricos"):
        params = generate_random_params()
        
        for __ in range(n_samples_per_set):
            y0 = np.random.uniform(search_range[0], search_range[1], 3)
            
            try:
                fp = fsolve(lambda y: three_roots_model(y, params), y0)
                
                if np.any(fp < search_range[0]-0.5) or np.any(fp > search_range[1]+0.5):
                    continue
                    
                J = jacobian(fp, params)
                eigvals, eigvecs = np.linalg.eig(J)
                
                positive_real = np.sum(np.real(eigvals) > 1e-4)
                negative_real = np.sum(np.real(eigvals) < -1e-4)
                
                if positive_real >= 1 and negative_real >= 1:
                    is_duplicate = False
                    for sp in all_saddle_points:
                        if np.linalg.norm(fp - sp) < 0.08:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_saddle_points.append(fp)
                        all_eigenvalues.append(eigvals)
                        all_eigenvectors.append(eigvecs)
                        all_params.append(params)
                        
            except:
                continue
                
    elapsed = time.time() - start_time
    print(f"Tiempo total: {elapsed/60:.2f} minutos")
    return all_saddle_points, all_eigenvalues, all_eigenvectors, all_params

# Visualización
def visualize_extended_saddles(saddle_points, eigenvalues):
    if not saddle_points:
        print("No se encontraron puntos silla")
        return
    
    sp_array = np.array(saddle_points)
    eig_array = np.array(eigenvalues)
    
    max_real_eig = np.max(np.real(eig_array), axis=1)
    min_real_eig = np.min(np.real(eig_array), axis=1)
    stability_index = max_real_eig / (max_real_eig - min_real_eig + 1e-8)
    
    fig = plt.figure(figsize=(18, 14))
    
    ax1 = fig.add_subplot(221, projection='3d')
    sc1 = ax1.scatter(sp_array[:,0], sp_array[:,1], sp_array[:,2], 
                     c=stability_index, cmap='jet', s=30, alpha=0.7)
    ax1.set_title('Puntos Silla: Espacio Kármico')
    ax1.set_xlabel('Ignorancia (I)')
    ax1.set_ylabel('Apego (A)')
    ax1.set_zlabel('Aversión (V)')
    plt.colorbar(sc1, ax=ax1, label='Índice de Inestabilidad')
    
    ax2 = fig.add_subplot(222)
    hb = ax2.hexbin(sp_array[:,0], sp_array[:,1], 
                   gridsize=30, cmap='viridis', bins='log')
    ax2.set_title('Densidad: Ignorancia vs Apego')
    ax2.set_xlabel('Ignorancia (I)')
    ax2.set_ylabel('Apego (A)')
    plt.colorbar(hb, ax=ax2, label='Log(Densidad)')
    
    ax3 = fig.add_subplot(223)
    sc3 = ax3.scatter(sp_array[:,2], max_real_eig, 
                     c=np.abs(sp_array[:,0]), cmap='coolwarm', s=40)
    ax3.set_title('Aversión vs Máxima Inestabilidad')
    ax3.set_xlabel('Aversión (V)')
    ax3.set_ylabel('Máximo Autovalor Real')
    plt.colorbar(sc3, ax=ax3, label='|Ignorancia|')
    
    ax4 = fig.add_subplot(224, projection='3d')
    hist, xedges, yedges = np.histogram2d(sp_array[:,0], sp_array[:,1], bins=20)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.5 * (xedges[1]-xedges[0])
    dz = hist.ravel()
    ax4.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, alpha=0.7)
    ax4.set_title('Distribución Ignorancia-Apego')
    ax4.set_xlabel('Ignorancia (I)')
    ax4.set_ylabel('Apego (A)')
    ax4.set_zlabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig('exhaustive_saddle_search.png', dpi=150)
    plt.show()
    
    return sp_array

# Interpretación budista ajustada a los seis reinos
def interpret_cluster(centroid, eigenvalues, eigenvectors):
    I, A, V = centroid
    interpretation = []
    
    # 1. Tipo de silla (dinámica)
    real_parts = np.real(eigenvalues)
    n_positive = np.sum(real_parts > 0.01)
    n_negative = np.sum(real_parts < -0.01)
    
    if n_positive == 1 and n_negative == 2:
        saddle_type = "Silla Atractor"
    elif n_positive == 2 and n_negative == 1:
        saddle_type = "Silla Repulsor"
    else:
        saddle_type = "Silla Mixto"
    interpretation.append(saddle_type)
    
    # 2. Dirección de inestabilidad (klesha dominante en transición)
    unstable_idx = np.argmax(real_parts)
    unstable_vec = np.real(eigenvectors[:, unstable_idx])
    unstable_vec /= np.linalg.norm(unstable_vec)
    
    dir_components = []
    if abs(unstable_vec[0]) > 0.4:
        dir_components.append(f"Ignorancia ({unstable_vec[0]:.2f})")
    if abs(unstable_vec[1]) > 0.4:
        dir_components.append(f"Apego ({unstable_vec[1]:.2f})")
    if abs(unstable_vec[2]) > 0.4:
        dir_components.append(f"Aversión ({unstable_vec[2]:.2f})")
    
    if dir_components:
        interpretation.append("Inestabilidad: " + "/".join(dir_components))
    
    # 3. Reino de transición según reinterpretación budista
    realm = ""
    # Primero identificar el veneno dominante en la dirección
    dominant_idx = np.argmax(np.abs(unstable_vec))
    dominant_value = unstable_vec[dominant_idx]
    
    if dominant_idx == 0 and dominant_value > 0:  # Ignorancia positiva
        realm = "Animal (Ignorancia dominante)"
    elif dominant_idx == 1 and dominant_value > 0:  # Apego positivo
        if unstable_vec[2] > 0.3:  # Con aversión significativa
            realm = "Preta (Apego con aversión)"
        else:
            realm = "Humano (Apego primario)"
    elif dominant_idx == 2 and dominant_value > 0:  # Aversión positiva
        realm = "Naraka (Aversión pura)"
    elif dominant_idx == 1 and dominant_value < 0:  # Apego negativo (desapego)
        realm = "Deva (Apego sutil)"
    elif dominant_idx == 1 and dominant_value > 0.5 and unstable_vec[2] > 0.4:
        realm = "Asura (Apego-aversión)"
    
    if realm:
        interpretation.append(f"Transición: {realm}")
    
    # 4. Características del estado según valores del centroide
    if I > 1.0:
        interpretation.append("Confusión profunda")
    elif I < -0.5:
        interpretation.append("Sabiduría desequilibrada")
    
    if A > 1.0:
        interpretation.append("Apego compulsivo")
    elif A < -0.5:
        interpretation.append("Desapego patológico")
    
    if V > 1.0:
        interpretation.append("Aversión destructiva")
    elif V < -0.5:
        interpretation.append("Aceptación indiscriminada")
    
    # 5. Patrones especiales
    if I > 0 and A > 0 and V > 0:
        interpretation.append("Caída desde reinos superiores")
    elif I < 0 and V > 0:
        interpretation.append("Caída desde pseudo-sabiduría")
    
    return " | ".join(interpretation)

# Análisis de clusters
def cluster_analysis(saddle_points, eigenvalues, eigenvectors):
    if not saddle_points:
        print("No hay puntos para clustering.")
        return pd.DataFrame()
    
    sp_array = np.array(saddle_points)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    sp_2d = tsne.fit_transform(sp_array)
    
    clusterer = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(sp_2d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    scatter = ax1.scatter(sp_2d[:,0], sp_2d[:,1], c=labels, 
                         cmap='Spectral', s=30, alpha=0.8)
    ax1.set_title('t-SNE de Puntos Silla')
    ax1.set_xlabel('Componente t-SNE 1')
    ax1.set_ylabel('Componente t-SNE 2')
    legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(sp_array[:,0], sp_array[:,1], sp_array[:,2], 
                     c=labels, cmap='Spectral', s=30, alpha=0.7)
    ax2.set_title('Clusters en Espacio Kármico')
    ax2.set_xlabel('Ignorancia (I)')
    ax2.set_ylabel('Apego (A)')
    ax2.set_zlabel('Aversión (V)')
    
    plt.tight_layout()
    plt.savefig('advanced_cluster_analysis.png', dpi=150)
    plt.show()
    
    unique_labels = np.unique(labels)
    cluster_stats = []
    
    for label in unique_labels:
        if label == -1:
            continue
            
        cluster_points = sp_array[labels == label]
        cluster_indices = np.where(labels == label)[0]
        centroid = np.mean(cluster_points, axis=0)
        
        cluster_eigvals = np.mean([eigenvalues[i] for i in cluster_indices], axis=0)
        cluster_eigvecs = np.mean([eigenvectors[i] for i in cluster_indices], axis=0)
        
        cluster_stats.append({
            'cluster': label,
            'size': len(cluster_points),
            'centroid': centroid,
            'ignorance_mean': centroid[0],
            'attachment_mean': centroid[1],
            'aversion_mean': centroid[2],
            'character': interpret_cluster(centroid, cluster_eigvals, cluster_eigvecs)
        })
    
    return pd.DataFrame(cluster_stats)

# Guardar resultados
def save_results(saddle_points, eigenvalues, eigenvectors, params_list):
    if not saddle_points:
        print("No hay puntos para guardar.")
        return None
    
    data = []
    for i, point in enumerate(saddle_points):
        record = {
            'I': point[0],
            'A': point[1],
            'V': point[2],
            'lambda1_real': np.real(eigenvalues[i][0]),
            'lambda1_imag': np.imag(eigenvalues[i][0]),
            'lambda2_real': np.real(eigenvalues[i][1]),
            'lambda2_imag': np.imag(eigenvalues[i][1]),
            'lambda3_real': np.real(eigenvalues[i][2]),
            'lambda3_imag': np.imag(eigenvalues[i][2]),
            'max_real_eig': np.max(np.real(eigenvalues[i])),
            'min_real_eig': np.min(np.real(eigenvalues[i])),
            'unstable_direction_I': np.real(eigenvectors[i][0, np.argmax(np.real(eigenvalues[i]))]),
            'unstable_direction_A': np.real(eigenvectors[i][1, np.argmax(np.real(eigenvalues[i]))]),
            'unstable_direction_V': np.real(eigenvectors[i][2, np.argmax(np.real(eigenvalues[i]))])
        }
        for key, value in params_list[i].items():
            record[f'param_{key}'] = value
        data.append(record)
    
    df = pd.DataFrame(data)
    filename = 'exhaustive_saddle_points.csv'
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename} ({len(df)} puntos)")
    return df

# Función principal
if __name__ == "__main__":
    print("=== BÚSQUEDA EXHAUSTIVA DE PUNTOS SILLA (COSMOLOGÍA BUDISTA) ===")
    saddle_points, eigenvalues, eigenvectors, params_list = extended_saddle_search(
        n_param_sets=8000,
        n_samples_per_set=800,
        search_range=(-3.0, 3.0))
    
    print(f"\nPuntos silla encontrados: {len(saddle_points)}")
    
    if saddle_points:
        sp_array = visualize_extended_saddles(saddle_points, eigenvalues)
        cluster_df = cluster_analysis(saddle_points, eigenvalues, eigenvectors)
        df = save_results(saddle_points, eigenvalues, eigenvectors, params_list)
        
        print("\n=== INTERPRETACIÓN KÁRMICA DE CLUSTERS ===")
        if not cluster_df.empty:
            for _, row in cluster_df.iterrows():
                print(f"\nCluster #{row['cluster']} (Tamaño: {row['size']})")
                print(f"Centroide: I={row['ignorance_mean']:.2f}, A={row['attachment_mean']:.2f}, V={row['aversion_mean']:.2f}")
                print(f"Carácter kármico: {row['character']}")
        else:
            print("No se encontraron clusters significativos.")
    else:
        print("No se encontraron puntos silla")