# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:35:40 2025

@author: eggra
"""

# Ejemplo de carga y análisis rápido
import numpy as np
import matplotlib.pyplot as plt

data = np.load("karmic_partial_results_472.npz", allow_pickle=True)
results = data['results']
params = data['parameters'].item()

# Gráfico de evolución de Lyapunov
plt.figure(figsize=(10,5))
for i, comp in enumerate(params['components']):
    plt.plot(results[:, i, 4], label=comp)  # Índice 4 = Lyapunov
plt.title("Evolución de Exponentes de Lyapunov")
plt.xlabel("Número de simulación")
plt.ylabel("Lyapunov")
plt.legend()
plt.show()