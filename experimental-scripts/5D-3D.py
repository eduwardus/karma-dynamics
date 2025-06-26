# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:59:10 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Datos de la tabla (Obscuración, Agresión, Defectividad)
reinos = ['Devas', 'Asuras', 'Humanos', 'Animales', 'Pretas', 'Infiernos']
datos = np.array([
    [8.4, 3.8, 3.5],    # Devas
    [8.0, 15.2, 9.8],   # Asuras
    [10.6, 11.1, 6.4],  # Humanos
    [12.8, 4.7, 1.4],   # Animales
    [14.6, 7.6, 4.8],   # Pretas
    [3.4, 10.8, 2.8]    # Infiernos
])

# Crear figura 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Configurar ejes
ax.set_xlabel('Obscuración', fontsize=12, labelpad=15)
ax.set_ylabel('Agresión', fontsize=12, labelpad=15)
ax.set_zlabel('Defectividad', fontsize=12, labelpad=15)
ax.set_title('Clasificación de los Seis Reinos en el Espacio de los Tres Venenos', fontsize=16, pad=20)

# Ajustar límites de ejes
ax.set_xlim([0, 16])
ax.set_ylim([0, 16])
ax.set_zlim([0, 12])

# Colores y marcadores
colores = ['gold', 'darkred', 'blue', 'green', 'purple', 'black']
marcadores = ['o', '^', 's', 'D', 'P', 'X']

# Graficar puntos
for i, reino in enumerate(reinos):
    ax.scatter(
        datos[i, 0], datos[i, 1], datos[i, 2],
        s=150, c=colores[i], marker=marcadores[i], depthshade=True,
        label=reino, edgecolor='black'
    )
    
    # Añadir etiqueta con offset
    ax.text(
        datos[i, 0] + 0.2, datos[i, 1] + 0.2, datos[i, 2] + 0.2,
        reino, fontsize=11, weight='bold'
    )

# Añadir líneas de referencia
for i in range(len(reinos)):
    ax.plot([datos[i, 0], datos[i, 0]], [datos[i, 1], datos[i, 1]], [0, datos[i, 2]], 
            c=colores[i], alpha=0.3, linestyle='--')
    ax.plot([datos[i, 0], datos[i, 0]], [0, datos[i, 1]], [datos[i, 2], datos[i, 2]], 
            c=colores[i], alpha=0.3, linestyle='--')
    ax.plot([0, datos[i, 0]], [datos[i, 1], datos[i, 1]], [datos[i, 2], datos[i, 2]], 
            c=colores[i], alpha=0.3, linestyle='--')

# Añadir plano de agrupación
x = np.linspace(0, 16, 10)
y = np.linspace(0, 16, 10)
X, Y = np.meshgrid(x, y)
Z_obsc = np.full_like(X, 12)  # Límite obscuración
Z_agr = np.full_like(X, 12)   # Límite agresión

ax.plot_surface(X, Y, Z_obsc, alpha=0.1, color='blue')
ax.plot_surface(X, Y, Z_agr, alpha=0.1, color='red')

# Leyenda y ángulo de vista
ax.legend(loc='upper left', fontsize=10)
ax.view_init(elev=25, azim=-45)

# Añadir cuadrícula
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.7)

# Guardar imagen
plt.tight_layout()
plt.savefig('diagrama-3d.png', dpi=300, bbox_inches='tight')
print("Diagrama 3D guardado como 'diagrama-3d.png'")