import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
import warnings

# Ignorar warnings para una salida más limpia
warnings.filterwarnings('ignore')

# Modelo de Tres Raíces Kármicas
def three_roots_model(y, params):
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# Jacobiano del sistema
def jacobian(y, params):
    I, A, V = y
    J = np.zeros((3, 3))
    
    J[0, 0] = params['alpha_I'] - params['gamma_I']*params['w'] + params['beta_IA']*A*V
    J[0, 1] = params['beta_IA']*V
    J[0, 2] = params['beta_IA']*A
    
    J[1, 0] = params['beta_AV']*V
    J[1, 1] = params['alpha_A'] - params['gamma_A']*params['w'] + params['beta_AV']*V*I
    J[1, 2] = params['beta_AV']*I
    
    J[2, 0] = params['beta_VI']*A
    J[2, 1] = params['beta_VI']*I
    J[2, 2] = params['alpha_V'] - params['gamma_V']*params['w'] + params['beta_VI']*I*A
    
    return J

# Generar parámetros aleatorios en rangos ampliados
def generate_random_params():
    params = {
        'alpha_I': np.random.uniform(0.01, 0.5),
        'alpha_A': np.random.uniform(0.01, 0.4),
        'alpha_V': np.random.uniform(0.01, 0.6),
        'beta_IA': np.random.uniform(0.05, 0.8),
        'beta_AV': np.random.uniform(0.05, 0.7),
        'beta_VI': np.random.uniform(0.05, 0.9),
        'gamma_I': np.random.uniform(0.1, 0.8),
        'gamma_A': np.random.uniform(0.1, 0.7),
        'gamma_V': np.random.uniform(0.1, 0.9),
        'w': np.random.uniform(0.1, 0.9)
    }
    return params

# Detección de puntos silla con exploración paramétrica extendida
def extended_saddle_search(n_param_sets=200, n_samples_per_set=300):
    all_saddle_points = []   # Cada elemento: [I, A, V]
    all_eigenvalues = []     # Lista de arrays de autovalores
    all_params = []          # Lista de diccionarios de parámetros
    
    print(f"Explorando {n_param_sets} conjuntos paramétricos con {n_samples_per_set} puntos cada uno...")
    
    for _ in tqdm(range(n_param_sets)):
        params = generate_random_params()
        
        for __ in range(n_samples_per_set):
            # Generar punto inicial aleatorio en rango ampliado
            y0 = np.random.uniform(0, 2, 3)
            
            try:
                # Encontrar punto fijo
                fp = fsolve(lambda y: three_roots_model(y, params), y0)
                
                # Verificar si es punto fijo válido
                if np.any(fp < -0.5) or np.any(fp > 2.5):
                    continue
                    
                # Calcular Jacobiano y autovalores
                J = jacobian(fp, params)
                eigvals = np.linalg.eigvals(J)
                
                # Contar autovalores con parte real positiva/negativa
                positive_real = np.sum(np.real(eigvals) > 1e-4)
                negative_real = np.sum(np.real(eigvals) < -1e-4)
                
                # Criterio para punto silla: al menos 1 autovalor positivo y 1 negativo
                if positive_real >= 1 and negative_real >= 1:
                    # Evitar duplicados
                    is_duplicate = False
                    for sp in all_saddle_points:
                        if np.linalg.norm(fp - sp) < 0.05:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_saddle_points.append(fp)
                        all_eigenvalues.append(eigvals)
                        all_params.append(params)
                        
            except:
                continue
                
    return all_saddle_points, all_eigenvalues, all_params

# Visualización avanzada de puntos silla (CORREGIDA)
def visualize_extended_saddles(saddle_points, eigenvalues):
    if not saddle_points:
        print("No se encontraron puntos silla en el espacio explorado.")
        return
    
    # Convertir a arrays para procesamiento
    sp_array = np.array(saddle_points)
    eig_array = np.array(eigenvalues)
    
    # Calcular valores para colorear
    max_real_eig = np.max(np.real(eig_array), axis=1)
    min_real_eig = np.min(np.real(eig_array), axis=1)
    norm_points = np.linalg.norm(sp_array, axis=1)
    
    # Crear figura
    fig = plt.figure(figsize=(16, 12))
    
    # Gráfico 3D principal
    ax1 = fig.add_subplot(221, projection='3d')
    sc1 = ax1.scatter(sp_array[:, 0], sp_array[:, 1], sp_array[:, 2], 
                     c=max_real_eig, cmap='viridis', s=50, alpha=0.8)
    ax1.set_title('Puntos Silla en Espacio Kármico')
    ax1.set_xlabel('Ignorancia (I)')
    ax1.set_ylabel('Apego (A)')
    ax1.set_zlabel('Aversión (V)')
    fig.colorbar(sc1, ax=ax1, label='Máx. Parte Real Autovalor')
    
    # Gráfico 2D: I vs A
    ax2 = fig.add_subplot(222)
    sc2 = ax2.scatter(sp_array[:, 0], sp_array[:, 1], 
                     c=min_real_eig, cmap='plasma', s=40)
    ax2.set_title('Proyección Ignorancia vs Apego')
    ax2.set_xlabel('Ignorancia (I)')
    ax2.set_ylabel('Apego (A)')
    ax2.grid(alpha=0.2)
    fig.colorbar(sc2, ax=ax2, label='Mín. Parte Real Autovalor')
    
    # Gráfico 2D: Norma del punto vs Máximo autovalor real
    ax3 = fig.add_subplot(223)
    sc3 = ax3.scatter(norm_points, max_real_eig, 
                     c=min_real_eig, cmap='coolwarm', s=50)
    ax3.set_title('Norma del Punto vs Máximo Autovalor Real')
    ax3.set_xlabel('Norma del Punto (||(I,A,V)||)')
    ax3.set_ylabel('Máximo Autovalor Real')
    ax3.grid(alpha=0.2)
    fig.colorbar(sc3, ax=ax3, label='Mín. Parte Real Autovalor')
    
    # Gráfico de distribución de autovalores
    ax4 = fig.add_subplot(224)
    all_real_parts = np.real(np.concatenate(eigenvalues))
    ax4.hist(all_real_parts, bins=50, color='teal', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--', label='Estabilidad (Re(λ)=0')
    ax4.set_title('Distribución de Partes Reales de Autovalores')
    ax4.set_xlabel('Parte Real de λ')
    ax4.set_ylabel('Frecuencia')
    ax4.legend()
    ax4.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('extended_saddle_points.png', dpi=150)
    plt.show()
    
    return sp_array

# Análisis de clusters de puntos silla
def cluster_analysis(saddle_points):
    from sklearn.cluster import DBSCAN
    
    if not saddle_points:
        return []
    
    print("\nRealizando análisis de clusters...")
    sp_array = np.array(saddle_points)
    
    # Normalizar los puntos
    normalized = (sp_array - np.mean(sp_array, axis=0)) / np.std(sp_array, axis=0)
    
    # Clustering con DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(normalized)
    labels = clustering.labels_
    
    # Contar clusters (excluyendo ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters identificados: {n_clusters}")
    
    # Visualización de clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Ruido - puntos negros
            col = 'k'
            size = 20
            alpha = 0.3
        else:
            size = 50
            alpha = 0.8
            
        class_member_mask = (labels == k)
        xyz = sp_array[class_member_mask]
        
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                   c=[col], s=size, alpha=alpha, edgecolor='k',
                   label=f'Cluster {k}' if k != -1 else 'Ruido')
    
    ax.set_title(f'Clusters de Puntos Silla ({n_clusters} grupos)')
    ax.set_xlabel('Ignorancia (I)')
    ax.set_ylabel('Apego (A)')
    ax.set_zlabel('Aversión (V)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('saddle_clusters.png', dpi=150)
    plt.show()
    
    return labels

# Guardar resultados en CSV
def save_results(saddle_points, eigenvalues, params_list):
    data = []
    for i, point in enumerate(saddle_points):
        params = params_list[i]
        eigvals = eigenvalues[i]
        
        # Crear registro de datos
        record = {
            'I': point[0],
            'A': point[1],
            'V': point[2],
            'lambda1_real': np.real(eigvals[0]),
            'lambda1_imag': np.imag(eigvals[0]),
            'lambda2_real': np.real(eigvals[1]),
            'lambda2_imag': np.imag(eigvals[1]),
            'lambda3_real': np.real(eigvals[2]),
            'lambda3_imag': np.imag(eigvals[2]),
            'max_real_eig': np.max(np.real(eigvals)),
            'min_real_eig': np.min(np.real(eigvals))
        }
        
        # Añadir parámetros
        for key, value in params.items():
            record[f"param_{key}"] = value
            
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv('saddle_points_analysis.csv', index=False)
    print(f"Resultados guardados en saddle_points_analysis.csv ({len(df)} puntos)")
    return df

# Ejecutar análisis extendido
if __name__ == "__main__":
    # Búsqueda extendida
    saddle_points, eigenvalues, params_list = extended_saddle_search(
        n_param_sets=300, 
        n_samples_per_set=400
    )
    
    print(f"\nTotal de puntos silla encontrados: {len(saddle_points)}")
    
    if saddle_points:
        # Visualización
        sp_array = visualize_extended_saddles(saddle_points, eigenvalues)
        
        # Análisis de clusters
        cluster_labels = cluster_analysis(saddle_points)
        
        # Guardar resultados
        df = save_results(saddle_points, eigenvalues, params_list)
        
        # Mostrar algunos resultados interesantes
        print("\nPuntos silla más significativos:")
        top_saddles = df.sort_values('max_real_eig', ascending=False).head(3)
        for i, row in top_saddles.iterrows():
            print(f"\nPunto Silla #{i}:")
            print(f"  Coordenadas: I={row['I']:.4f}, A={row['A']:.4f}, V={row['V']:.4f}")
            print(f"  Autovalor máximo: {row['max_real_eig']:.4f}")
            print(f"  Parámetros clave: w={row['param_w']:.3f}, β_IA={row['param_beta_IA']:.3f}")
    else:
        print("No se encontraron puntos silla en el espacio explorado.")