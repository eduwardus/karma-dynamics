# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 02:43:18 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def karmic_vector(y, t, alpha_I, alpha_A, alpha_V, beta_IA, beta_AV, beta_VI, gamma, w):
    I, A, V = y
    dIdt = alpha_I * I + beta_IA * A * V - gamma * w * I
    dAdt = alpha_A * A + beta_AV * V * I - gamma * w * A
    dVdt = alpha_V * V + beta_VI * I * A - gamma * w * V
    return [dIdt, dAdt, dVdt]


# Parámetros caóticos
params = (0.1, 0.1, 0.1, 0.5, 0.6, 0.7, 0.3, 0.2)  # alpha_I, ..., w
y0= [0.4, 0.3, 0.2]
t= np.linspace(0, 200, 10000)  # Tiempo largo para observar el atractor

# Resolver ecuaciones
sol = odeint(karmic_vector, y0, t, args=params)

# Gráfico 3D del atractor

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.5)
ax.set_xlabel('Ignorancia (I)')
ax.set_ylabel('Apego (A)')
ax.set_zlabel('Aversión (V)')
plt.title('Atractor Caótico del Sistema Kármico')
plt.show()
# Graficar
plt.plot(t, sol[:, 0], label='Ignorancia (I)')
plt.plot(t, sol[:, 1], label='Apego (A)')
plt.plot(t, sol[:, 2], label='Aversión (V)')
plt.xlabel('Tiempo')
plt.ylabel('Intensidad')
plt.legend()
plt.show()