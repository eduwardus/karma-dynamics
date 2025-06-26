# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:25:00 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
#from scipy.stats import multivariate_normal

# Matriz de transformación T (3x5)
T = np.array([
    [1, 0.8, 0, 0, 0],
    [0, 0, 1, 0.9, 0],
    [0, 0, 0, 0.2, 1]
])

# Rangos de los 5 venenos por reino
rangos_5d = {
    'Devas':    [(2,4), (8,10), (1,3), (7,9), (3,5)],
    'Asuras':   [(4,6), (6,8), (7,9), (8,10), (9,10)],
    'Humanos':  [(5,7), (7,9), (5,7), (6,8), (5,7)],
    'Animales': [(9,10), (4,6), (3,5), (1,3), (1,2)],
    'Pretas':   [(7,8), (9,10), (6,8), (2,4), (4,6)],
    'Infiernos':[(6,7), (1,3), (9,10), (1,2), (2,4)]
}

# Centroides 3D de los reinos
centroides_3d = {
    'Devas':    np.array([8.4, 3.8, 3.5]),
    'Asuras':   np.array([8.0, 15.2, 9.8]),
    'Humanos':  np.array([10.6, 11.1, 6.4]),
    'Animales': np.array([12.8, 4.7, 1.4]),
    'Pretas':   np.array([14.6, 7.6, 4.8]),
    'Infiernos':np.array([3.4, 10.8, 2.8])
}

def encontrar_reino(punto_3d):
    """Identifica el reino más cercano usando distancia de Mahalanobis"""
    distancias = {}
    for reino, centroide in centroides_3d.items():
        # Calcular matriz de covarianza empírica
        cov = np.diag((centroide * 0.3) ** 2)  # 30% de variabilidad
        diff = punto_3d - centroide
        try:
            dist = diff @ np.linalg.inv(cov) @ diff.T
            distancias[reino] = dist
        except:
            distancias[reino] = np.linalg.norm(diff)
    
    return min(distancias, key=distancias.get)

def transformacion_inversa(punto_3d, n_muestras=500, sigma=0.3):
    """
    Transformación inversa difusa con distribución de probabilidad
    basada en restricciones de reino y proximidad a la transformación
    """
    # Paso 1: Identificar reino
    reino = encontrar_reino(punto_3d)
    rangos = rangos_5d[reino]
    
    # Paso 2: Encontrar punto óptimo en 5D usando optimización
    def funcion_perdida(v5):
        # Penalización por salirse de los rangos
        penalizacion = 0
        for i, (min_val, max_val) in enumerate(rangos):
            if v5[i] < min_val:
                penalizacion += (min_val - v5[i]) * 10
            elif v5[i] > max_val:
                penalizacion += (v5[i] - max_val) * 10
        return np.linalg.norm(T @ v5 - punto_3d) + penalizacion
    
    # Punto inicial (centro del reino)
    v5_inicial = np.array([(min_val + max_val)/2 for min_val, max_val in rangos])
    resultado = minimize(funcion_perdida, v5_inicial, method='L-BFGS-B', 
                         bounds=rangos)
    
    if not resultado.success:
        print("Advertencia: Optimización no convergió. Usando solución aproximada.")
    
    v5_optimo = resultado.x
    
    # Paso 3: Crear distribución gaussiana truncada alrededor del óptimo
    covarianza = np.diag([(max_val-min_val)*sigma for min_val, max_val in rangos])
    muestras = []
    
    while len(muestras) < n_muestras:
        candidato = np.random.multivariate_normal(v5_optimo, covarianza)
        
        # Verificar restricciones de rango
        valido = True
        for i, (min_val, max_val) in enumerate(rangos):
            if candidato[i] < min_val or candidato[i] > max_val:
                valido = False
                break
        
        # Verificar proximidad a la transformación
        if valido and np.linalg.norm(T @ candidato - punto_3d) < 1.0:
            muestras.append(candidato)
    
    return np.array(muestras), reino, v5_optimo

def visualizar_nube_5d(muestras_5d, reino, v5_optimo):
    """Visualización avanzada de la nube de puntos en 5D"""
    fig = plt.figure(figsize=(20, 15))
    
    # Configurar proyecciones
    proyecciones = [
        (0, 1, 2, 'Ignorancia', 'Apego', 'Aversión'),
        (1, 3, 4, 'Apego', 'Orgullo', 'Envidia'),
        (0, 3, 4, 'Ignorancia', 'Orgullo', 'Envidia'),
        (2, 3, 4, 'Aversión', 'Orgullo', 'Envidia')
    ]
    
    for i, (x_idx, y_idx, z_idx, x_lab, y_lab, z_lab) in enumerate(proyecciones, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        # Graficar nube de puntos
        ax.scatter(
            muestras_5d[:, x_idx], muestras_5d[:, y_idx], muestras_5d[:, z_idx],
            c=muestras_5d[:, 0], cmap='viridis', alpha=0.6, depthshade=True,
            s=50, edgecolor='k'
        )
        
        # Marcar punto óptimo
        ax.scatter(
            [v5_optimo[x_idx]], [v5_optimo[y_idx]], [v5_optimo[z_idx]],
            s=200, c='red', marker='*', edgecolor='gold'
        )
        
        # Etiquetas y título
        ax.set_xlabel(x_lab, fontsize=12)
        ax.set_ylabel(y_lab, fontsize=12)
        ax.set_zlabel(z_lab, fontsize=12)
        ax.set_title(f'Reino {reino}: {x_lab}-{y_lab}-{z_lab}', fontsize=14)
        
        # Añadir cubo de rango
        min_vals = [rangos_5d[reino][idx][0] for idx in [x_idx, y_idx, z_idx]]
        max_vals = [rangos_5d[reino][idx][1] for idx in [x_idx, y_idx, z_idx]]
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        
        # Añadir grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'nube_5d_{reino}.png', dpi=300)
    plt.show()

# Función para validar la transformación inversa
def validar_transformacion(muestras_5d, punto_3d_original):
    """Calcula el error de reconstrucción"""
    reconstruidos = T @ muestras_5d.T
    errores = np.linalg.norm(reconstruidos - punto_3d_original.reshape(3, 1), axis=0)
    return np.mean(errores), np.std(errores)

# Ejemplo de uso
if __name__ == "__main__":
    # Punto en espacio reducido (Obscuración, Agresión, Defectividad)
    punto_3d = np.array([10.6, 11.1, 6.4])  # Humanos
    
    # Transformación inversa difusa
    muestras_5d, reino, v5_optimo = transformacion_inversa(punto_3d, n_muestras=1000)
    
    print(f"Reino identificado: {reino}")
    print(f"Punto óptimo en 5D: {v5_optimo}")
    
    # Visualización
    visualizar_nube_5d(muestras_5d, reino, v5_optimo)
    
    # Validación
    error_medio, error_std = validar_transformacion(muestras_5d, punto_3d)
    print(f"Error medio de reconstrucción: {error_medio:.4f} ± {error_std:.4f}")
    
    # Análisis estadístico
    print("\nEstadísticas de las muestras generadas:")
    print(f"- Ignorancia: {np.mean(muestras_5d[:,0]):.2f} ± {np.std(muestras_5d[:,0]):.2f}")
    print(f"- Apego: {np.mean(muestras_5d[:,1]):.2f} ± {np.std(muestras_5d[:,1]):.2f}")
    print(f"- Aversión: {np.mean(muestras_5d[:,2]):.2f} ± {np.std(muestras_5d[:,2]):.2f}")
    print(f"- Orgullo: {np.mean(muestras_5d[:,3]):.2f} ± {np.std(muestras_5d[:,3]):.2f}")
    print(f"- Envidia: {np.mean(muestras_5d[:,4]):.2f} ± {np.std(muestras_5d[:,4]):.2f}")