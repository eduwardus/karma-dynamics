# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:52:14 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:36:12 2025

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

# Parameters and initial conditions
params = (0.1, 0.1, 0.1, 0.3, 0.2, 0.4, 0.5, 0.4)
y0 = [0.5, 0.3, 0.2]
t = np.linspace(0, 50, 1000)

# Resolve equations
sol = odeint(karmic_vector, y0, t, args=params)

# Graficar
plt.plot(t, sol[:, 0], label='Ignorance (I)')
plt.plot(t, sol[:, 1], label='Attachment (A)')
plt.plot(t, sol[:, 2], label='Aversion (V)')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()

# save as vectorial file
plt.savefig("karmic_dynamics.pdf", format="pdf")

plt.show()
