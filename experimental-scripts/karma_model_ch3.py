# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 02:43:18 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def karmic_vector(y, t, alpha_I, alpha_A, alpha_V, beta_IA, beta_AV, beta_VI, gamma, w):
    I, A, V = y
    dIdt = alpha_I * I + beta_IA * A * V - gamma * w * I
    dAdt = alpha_A * A + beta_AV * V * I - gamma * w * A
    dVdt = alpha_V * V + beta_VI * I * A - gamma * w * V
    return [dIdt, dAdt, dVdt]

# Parámetros y condiciones iniciales
params = (0.1, 0.1, 0.1, 0.3, 0.2, 0.4, 0.5, 0.6)  # alpha_I, alpha_A, ..., w
y0 = [0.5, 0.3, 0.2]
t = np.linspace(0, 50, 1000)

# Resolver ecuaciones
sol = odeint(karmic_vector, y0, t, args=params)

# Graficar
plt.plot(t, sol[:, 0], label='Ignorancia (I)')
plt.plot(t, sol[:, 1], label='Apego (A)')
plt.plot(t, sol[:, 2], label='Aversión (V)')
plt.xlabel('Tiempo')
plt.ylabel('Intensidad')
plt.legend()
plt.show()