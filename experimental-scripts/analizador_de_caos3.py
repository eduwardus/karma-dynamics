# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 22:20:27 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def visualizar_trayectoria(archivo):
    """Visualización rápida de trayectorias"""
    data = np.load(archivo)
    tiempo = np.arange(0, len(data)) * 0.01  # dt=0.01
    
    plt.figure(figsize=(12, 8))
    
    # Series temporales
    plt.subplot(2, 1, 1)
    for i, nombre in enumerate(['Avidyā', 'Rāga', 'Dveṣa']):
        plt.plot(tiempo, data[:, i], label=nombre)
        
        # Detectar y marcar picos
        peaks, _ = find_peaks(data[:, i], height=0.1, distance=50)
        plt.plot(tiempo[peaks], data[peaks, i], 'ro', markersize=4)
    
    plt.title('Series Temporales')
    plt.xlabel('Tiempo')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Espacio de fases 3D
    ax = plt.subplot(2, 1, 2, projection='3d')
    ax.plot(data[:,0], data[:,1], data[:,2], lw=0.5)
    ax.set_xlabel('Avidyā')
    ax.set_ylabel('Rāga')
    ax.set_zlabel('Dveṣa')
    ax.set_title('Espacio de Fase 3D')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
# visualizar_trayectoria("cyclic_simple_trajectory_sim_123.npy")