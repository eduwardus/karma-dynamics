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

# Generar parámetros aleatorios con más variación
def generate_random_params():
    return {
        'alpha_I': np.random.uniform(0.01, 0.8),
        'alpha_A': np.random.uniform(0.01, 0.7),
        'alpha_V': np.random.uniform(0.01, 1.0),
        'beta_IA': np.random.uniform(0.05, 1.2),
        'beta_AV': np.random.uniform(0.05, 1.0),
        'beta_VI': np.random.uniform(0.05, 1.5),
        'gamma_I': np.random.uniform(0.1, 1.2),
        'gamma_A': np.random.uniform(0.1, 1.0),
        'gamma_V': np.random.uniform(0.1, 1.3),
        'w': np.random.uniform(0.1, 1.2)
    }

# Búsqueda extendida con parámetros ampliados
def extended_saddle_search(n_param_sets=1500, n_samples_per_set=600, 
                          search_range=(-1.0, 3.0)):
    all_saddle_points = []
    all_eigenvalues = []
    all_params = []
    
    print(f"Búsqueda exhaustiva: {n_param_sets}×{n_samples_per_set} = {n_param_sets*n_samples_per_set:,} puntos")
    start_time = time.time()
    
    for _ in tqdm(range(n_param_sets), desc="Conjuntos paramétricos"):
        params = generate_random_params()
        
        for __ in range(n_samples_per_set):
            y0 = np.random.uniform(search_range[0], search_range[1], 3)
            
            try:
                fp = fsolve(lambda y: three_roots_model(y, params), y0)
                
                # Criterio de validación más amplio
                if np.any(fp < search_range[0]-0.5) or np.any(fp > search_range[1]+0.5):
                    continue
                    
                J = jacobian(fp, params)
                eigvals = np.linalg.eigvals(J)
                
                positive_real = np.sum(np.real(eigvals) > 1e-4)
                negative_real = np.sum(np.real(eigvals) < -1e-4)
                
                if positive_real >= 1 and negative_real >= 1:
                    # Tolerancia más flexible para duplicados
                    is_duplicate = False
                    for sp in all_saddle_points:
                        if np.linalg.norm(fp - sp) < 0.08:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_saddle_points.append(fp)
                        all_eigenvalues.append(eigvals)
                        all_params.append(params)
                        
            except:
                continue
                
    elapsed = time.time() - start_time
    print(f"Tiempo total: {elapsed/60:.2f} minutos")
    return all_saddle_points, all_eigenvalues, all_params

# Visualización mejorada para más clusters
def visualize_extended_saddles(saddle_points, eigenvalues):
    if not saddle_points:
        print("No se encontraron puntos silla")
        return
    
    sp_array = np.array(saddle_points)
    eig_array = np.array(eigenvalues)
    
    # Calcular métricas para colorear
    max_real_eig = np.max(np.real(eig_array), axis=1)
    min_real_eig = np.min(np.real(eig_array), axis=1)
    stability_index = max_real_eig / (max_real_eig - min_real_eig + 1e-8)
    
    fig = plt.figure(figsize=(18, 14))
    
    # Gráfico 3D principal con colores por cluster
    ax1 = fig.add_subplot(221, projection='3d')
    sc1 = ax1.scatter(sp_array[:,0], sp_array[:,1], sp_array[:,2], 
                     c=stability_index, cmap='jet', s=30, alpha=0.7)
    ax1.set_title('Puntos Silla: Espacio Kármico (Color = Índice de Inestabilidad)')
    ax1.set_xlabel('Ignorancia (I)')
    ax1.set_ylabel('Apego (A)')
    ax1.set_zlabel('Aversión (V)')
    plt.colorbar(sc1, ax=ax1, label='Índice de Inestabilidad')
    
    # Proyección 2D con densidad
    ax2 = fig.add_subplot(222)
    hb = ax2.hexbin(sp_array[:,0], sp_array[:,1], 
                   gridsize=30, cmap='viridis', bins='log')
    ax2.set_title('Densidad de Puntos Silla: Ignorancia vs Apego')
    ax2.set_xlabel('Ignorancia (I)')
    ax2.set_ylabel('Apego (A)')
    plt.colorbar(hb, ax=ax2, label='Log(Densidad)')
    
    # Distribución de aversión vs inestabilidad
    ax3 = fig.add_subplot(223)
    sc3 = ax3.scatter(sp_array[:,2], max_real_eig, 
                     c=np.abs(sp_array[:,0]), cmap='coolwarm', s=40)
    ax3.set_title('Aversión vs Máxima Inestabilidad')
    ax3.set_xlabel('Aversión (V)')
    ax3.set_ylabel('Máximo Autovalor Real')
    plt.colorbar(sc3, ax=ax3, label='|Ignorancia|')
    
    # Histograma 3D de distribución
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

# Análisis de clusters mejorado
def cluster_analysis(saddle_points):
    if not saddle_points:
        print("No hay puntos para clustering.")
        return pd.DataFrame()
    
    sp_array = np.array(saddle_points)
    
    # Reducción de dimensionalidad
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    sp_2d = tsne.fit_transform(sp_array)
    
    # Clustering con HDBSCAN (mejor para densidad variable)
    clusterer = HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(sp_2d)
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Gráfico t-SNE
    scatter = ax1.scatter(sp_2d[:,0], sp_2d[:,1], c=labels, 
                         cmap='Spectral', s=30, alpha=0.8)
    ax1.set_title('t-SNE de Puntos Silla')
    ax1.set_xlabel('Componente t-SNE 1')
    ax1.set_ylabel('Componente t-SNE 2')
    legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)
    
    # Gráfico 3D original con colores de cluster
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(sp_array[:,0], sp_array[:,1], sp_array[:,2], 
                     c=labels, cmap='Spectral', s=30, alpha=0.7)
    ax2.set_title('Clusters en Espacio Kármico Original')
    ax2.set_xlabel('Ignorancia (I)')
    ax2.set_ylabel('Apego (A)')
    ax2.set_zlabel('Aversión (V)')
    
    plt.tight_layout()
    plt.savefig('advanced_cluster_analysis.png', dpi=150)
    plt.show()
    
    # Análisis de cada cluster
    unique_labels = np.unique(labels)
    cluster_stats = []
    
    for label in unique_labels:
        if label == -1:
            continue  # Saltar ruido
            
        cluster_points = sp_array[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        spread = np.std(cluster_points, axis=0)
        
        cluster_stats.append({
            'cluster': label,
            'size': len(cluster_points),
            'centroid': centroid,
            'spread': spread,
            'ignorance_mean': centroid[0],
            'attachment_mean': centroid[1],
            'aversion_mean': centroid[2],
            'character': interpret_cluster(centroid)
        })
    
    return pd.DataFrame(cluster_stats)

# Interpretación budista de clusters
def interpret_cluster(centroid):
    I, A, V = centroid
    interpretation = []
    
    # Interpretación basada en posición en el espacio kármico
    if I > 1.0:
        interpretation.append("Estados de confusión profunda")
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
    
    # Carácter general del cluster
    if len(interpretation) == 0:
        interpretation.append("Transiciones moderadas")
    
    if I > 0 and A > 0 and V > 0:
        interpretation.append("Caída desde reinos superiores")
    elif I < 0 and V > 0:
        interpretation.append("Caída desde pseudo-sabiduría")
    
    return " | ".join(interpretation)

# Guardar resultados en CSV (FUNCIÓN FALTANTE)
def save_results(saddle_points, eigenvalues, params_list):
    if not saddle_points:
        print("No hay puntos para guardar.")
        return None
    
    # Crear un DataFrame con los puntos silla
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
            'min_real_eig': np.min(np.real(eigenvalues[i]))
        }
        # Añadir los parámetros
        for key, value in params_list[i].items():
            record[f'param_{key}'] = value
        data.append(record)
    
    df = pd.DataFrame(data)
    filename = 'exhaustive_saddle_points.csv'
    df.to_csv(filename, index=False)
    print(f"Resultados guardados en {filename} ({len(df)} puntos)")
    return df

# Función principal ejecutable
if __name__ == "__main__":
    print("=== BÚSQUEDA EXHAUSTIVA DE PUNTOS SILLA ===")
    saddle_points, eigenvalues, params_list = extended_saddle_search(
        n_param_sets=1500,
        n_samples_per_set=600,
        search_range=(-1.0, 3.0))
    
    print(f"\nPuntos silla encontrados: {len(saddle_points)}")
    
    if saddle_points:
        # Visualización mejorada
        sp_array = visualize_extended_saddles(saddle_points, eigenvalues)
        
        # Análisis de clusters avanzado
        cluster_df = cluster_analysis(saddle_points)
        
        # Guardar resultados
        df = save_results(saddle_points, eigenvalues, params_list)
        
        # Mostrar interpretación de clusters
        print("\n=== INTERPRETACIÓN DE CLUSTERS ===")
        if not cluster_df.empty:
            for _, row in cluster_df.iterrows():
                print(f"\nCluster #{row['cluster']} (Tamaño: {row['size']})")
                print(f"Centroide: I={row['ignorance_mean']:.2f}, A={row['attachment_mean']:.2f}, V={row['aversion_mean']:.2f}")
                print(f"Carácter kármico: {row['character']}")
        else:
            print("No se encontraron clusters significativos.")
    else:
        print("No se encontraron puntos silla")
        
"""
## Interpretación Budista Detallada de los Clusters de Puntos Silla

### **Cluster #1: El Asceta Desequilibrado**
- **Centroide**: I=0.70, A=-0.59, V=-0.65
- **Carácter kármico**: Desapego patológico | Aceptación indiscriminada
- **Interpretación**:
  Representa estados de **falsa renuncia** donde el desapego extremo (A negativo) se convierte en una forma de aversión disfrazada. La alta ignorancia (I=0.70) combinada con aceptación compulsiva (V negativo) sugiere una espiritualidad basada en el autoengaño. 
  - **Kármicamente**: Transición a reinos de aislamiento espiritual (ciertos estados divinos estériles) o renacimiento como asceta atrapado en su propio dogmatismo.
  - **Símil budista**: Como un monje que medita en el bosque pero desprecia a quienes no siguen su camino.

### **Cluster #2: El Intelectual Espiritual**
- **Centroide**: I=-0.62, A=0.65, V=-0.62
- **Carácter kármico**: Sabiduría desequilibrada | Aceptación indiscriminada
- **Interpretación**:
  Estado de **sabiduría intelectualizada** (I negativo) con apego al conocimiento (A alto) y aceptación acrítica de doctrinas (V negativo). 
  - **Kármicamente**: Caída a reinos humanos donde se privilegia el intelecto sobre la compasión, o renacimiento como erudito atrapado en debates filosóficos estériles.
  - **Referencia**: Similar a los brahmanes en los sutras que debaten con Buda pero no practican.

### **Cluster #3: El Falso Iluminado**
- **Centroide**: I=-0.42, A=0.62, V=0.65
- **Carácter kármico**: Transiciones moderadas | Caída desde pseudo-sabiduría
- **Interpretación**:
  **Sabiduría arrogante** (I negativo) combinada con apego al estatus espiritual (A alto) y aversión sutil hacia quienes no comprenden "su verdad" (V alto). 
  - **Kármicamente**: Transición a reinos asúricos (competitividad espiritual) o renacimiento como gurú que comercia con enseñanzas.
  - **Peligro**: Estado típico previo a grandes caídas kármicas por abuso de autoridad espiritual.

### **Cluster #4: La Indiferencia Peligrosa**
- **Centroide**: I=-0.40, A=0.00, V=0.01
- **Carácter kármico**: Transiciones moderadas | Caída desde pseudo-sabiduría
- **Interpretación**:
  **Sabiduría fría** (I negativo) sin conexión emocional (A neutro) ni compromiso (V neutro). Representa el peligro del desapego sin compasión. 
  - **Kármicamente**: Renacimiento en estados liminares (como ciertos reinos animales superiores) donde hay conciencia pero no engagement con el sufrimiento ajeno.
  - **Enseñanza**: Ilustra el error de buscar solo la sabiduría sin desarrollar el corazón compasivo.

### **Cluster #5: La Tolerancia Tóxica**
- **Centroide**: I=0.14, A=0.06, V=-0.57
- **Carácter kármico**: Aceptación indiscriminada
- **Interpretación**:
  Estado de **pasividad kármica** donde la aceptación extrema (V negativo) permite el abuso y la injusticia. La baja ignorancia (I=0.14) sugiere conciencia del problema pero inacción. 
  - **Kármicamente**: Renacimiento en situaciones de opresión donde se "acepta" el sufrimiento como destino.
  - **Paradoja**: Muestra cómo la "no-aversión" mal entendida puede perpetuar el sufrimiento.

### **Cluster #6: La Tormenta Kármica**
- **Centroide**: I=1.11, A=1.12, V=1.16
- **Carácter kármico**: Estados de confusión profunda | Apego compulsivo | Aversión destructiva | Caída desde reinos superiores
- **Interpretación**:
  **Triple intoxicación extrema**. Representa la caída catastrófica de aquellos que tuvieron privilegios kármicos (devas) pero los malgastaron. 
  - **Kármicamente**: Transición abrupta a reinos inferiores (narakas o estados animales inferiores).
  - **Símil**: Como un deva que abusa de su longevidad y poderes, acumulando karma negativo masivo.

### **Cluster #7: La Estancación Espiritual**
- **Centroide**: I=0.70, A=-0.01, V=-0.01
- **Carácter kármico**: Transiciones moderadas
- **Interpretación**:
  **Ignorancia cómoda** sin apego ni aversión activos. Estado de estancamiento donde se evitan tanto los placeres como los desafíos. 
  - **Kármicamente**: Renacimiento en reinos de pura supervivencia (ciertos estados animales) o vidas humanas sin crecimiento espiritual.
  - **Peligro**: La "zona de confort" como trampa kármica que impide el despertar.

### **Cluster #8: La Adicción Samsárica**
- **Centroide**: I=0.52, A=0.46, V=0.45
- **Carácter kármico**: Transiciones moderadas | Caída desde reinos superiores
- **Interpretación**:
  **Desequilibrio armonioso** donde las tres raíces están presentes pero compensadas. Representa el estado típico del samsara "confortable". 
  - **Kármicamente**: Renacimiento repetido en reinos humanos o devas bajos con ciclos de placer-sufrimiento moderados.
  - **Paradoja**: El peligro de un samsara "aceptable" que no motiva la búsqueda de liberación.

### **Cluster #9: El Misántropo Espiritual**
- **Centroide**: I=0.94, A=-0.75, V=0.86
- **Carácter kármico**: Desapego patológico
- **Interpretación**:
  **Desapego con aversión** - estado de quien renuncia al mundo por desprecio (no por sabiduría). Alta ignorancia (I=0.94) combinada con rechazo a lo humano (V alto). 
  - **Kármicamente**: Renacimiento como espíritu solitario (ciertos reinos pretas) o eremita atormentado.
  - **Enseñanza**: Muestra cómo el desapego sin sabiduría se convierte en otra forma de sufrimiento.

### **Cluster #10: El Equilibrio Engañoso**
- **Centroide**: I=0.30, A=-0.33, V=0.33
- **Carácter kármico**: Transiciones moderadas
- **Interpretación**:
  **Falso equilibrio** entre aversión (V positivo) y desapego (A negativo). Estado de aparente balance que oculta conflicto interno. 
  - **Kármicamente**: Renacimiento en familias disfuncionales o entornos donde se simula armonía pero hay tensión latente.
  - **Sutil**: Peligroso por su apariencia de estabilidad que en realidad es estancamiento.

## Patrones Cosmológicos Emergentes

1. **La Trampa de los Extremos**:
   - Clusters 1, 2, 5 y 9 muestran cómo cualquier cualidad llevada al extremo (incluso el desapego o aceptación) se convierte en veneno kármico.

2. **La Caída de los "Espirituales"**:
   - Clusters 2, 3 y 4 revelan que los mayores peligros están en estados de pseudo-sabiduría, no en la ignorancia burda.

3. **Dinámica de los Reinos Superiores**:
   - Clusters 6 y 8 muestran dos patrones de caída desde reinos devas: por abuso (6) o por complacencia (8).

4. **La Paradoja del Desapego**:
   - Clusters 1, 7 y 9 demuestran que el desapego sin compasión conduce a estados kármicamente peligrosos.

5. **Equilibrios Engañosos**:
   - Clusters 7 y 10 revelan que no todo equilibrio es saludable; algunos son formas de estancamiento.

## Conclusión Filosófica

Estos clusters muestran que **el camino medio budista no es un punto sino una dinámica consciente**. El verdadero equilibrio no es la neutralidad (Cluster 7) ni la compensación de opuestos (Cluster 10), sino la sabiduría que disuelve los extremos. 

La presencia de múltiples clusters con sabiduría negativa (I<0 en Clusters 2-4) confirma la advertencia budista sobre el "conocimiento espiritual" como obstáculo. Mientras que el Cluster 6 (triple intoxicación) ejemplifica el samsara en su expresión más cruda, los clusters más peligrosos son aquellos que simulan liberación pero refuerzan el ego espiritual.

La enseñanza profunda: **Toda posición fija en el espacio kármico es un punto silla**, y solo el movimiento constante hacia el desapego genuino (no la posición de "no-apego") conduce al Nirvana.
"""