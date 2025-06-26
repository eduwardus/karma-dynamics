# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:52:14 2025

@author: eggra
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def karmic_vector_weak_damp(t, y, alpha_I, alpha_A, alpha_V,
                            beta_IA, beta_AV, beta_VI, gamma, w):
    I, A, V = y
    # saturación más suave
    term_IA_V = np.tanh((A * V) / 2)
    term_AV_I = np.tanh((V * I) / 2)
    term_VI_A = np.tanh((I * A) / 2)
    dIdt = alpha_I * I + beta_IA * term_IA_V - gamma * w * I
    dAdt = alpha_A * A + beta_AV * term_AV_I - gamma * w * A
    dVdt = alpha_V * V + beta_VI * term_VI_A - gamma * w * V
    return [dIdt, dAdt, dVdt]

# Parámetros “caóticos”
params = {
    'alpha_I': 0.02, 'alpha_A': 0.02, 'alpha_V': 0.02,
    'beta_IA': 1.0,  'beta_AV': 0.9,  'beta_VI': 1.1,
    'gamma':    0.01, 'w':       0.05
}

y0 = [0.1, 0.2, 0.1]
t_span = (0, 500)
t_eval = np.linspace(*t_span, 12000)

sol = solve_ivp(
    fun=lambda t, y: karmic_vector_weak_damp(t, y, **params),
    t_span=t_span, y0=y0, t_eval=t_eval,
    method='RK45', rtol=1e-6, atol=1e-9, max_step=0.5
)

if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
    raise RuntimeError("¡Todavía hay valores no finitos!")

# Plot 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
pts   = sol.y.T.reshape(-1,1,3)
segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc    = Line3DCollection(segs, cmap='plasma',
                         norm=plt.Normalize(sol.t.min(), sol.t.max()))
lc.set_array(sol.t); lc.set_linewidth(0.5)
ax.add_collection3d(lc)

ax.set_xlim(sol.y[0].min(), sol.y[0].max())
ax.set_ylim(sol.y[1].min(), sol.y[1].max())
ax.set_zlim(sol.y[2].min(), sol.y[2].max())
ax.set_xlabel('I'); ax.set_ylabel('A'); ax.set_zlabel('V')
plt.title('Atractor Caótico Débilmente Amortiguado')
plt.tight_layout()
plt.show()
