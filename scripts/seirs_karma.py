# scripts/seirs_karma.py
"""
KARMIC DYNAMICS SIMULATION FRAMEWORK - EXPERIMENTAL VERSION

DISCLAIMER: 
This code implements speculative mathematical models bridging 
epidemiology and Buddhist philosophy. It is intended for:
- Conceptual exploration
- Methodological experimentation
- Interdisciplinary dialogue

NOT FOR:
- Doctrinal interpretation
- Clinical application
- Metaphysical claims

All models are provisional abstractions subject to revision.
Parameter values are heuristic estimates without empirical validation.
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Model parameters
params = {
    'alpha': 0.3,    # Latent karma activation rate
    'sigma': 0.5,    # Intention manifestation rate
    'gamma': 0.4,    # Action resolution rate
    'xi': 0.2,       # Waste recycling rate
    'w': 0.3,        # Wisdom factor
    'lambda_val': 0.1 # Maximum feedback rate
}

# Hill function
def hill_function(R):
    return R / (1 + R)

# Model equations
def model(y, t, params):
    S, E, I, R = y
    dSdt = params['xi'] * (1 - params['w']) * R - params['alpha'] * S + params['lambda_val'] * hill_function(R)
    dEdt = params['alpha'] * S - params['sigma'] * E
    dIdt = params['sigma'] * E - params['gamma'] * I
    dRdt = params['gamma'] * I - params['xi'] * (1 - params['w']) * R - params['lambda_val'] * hill_function(R)
    return [dSdt, dEdt, dIdt, dRdt]

# Initial conditions and time
y0 = [0.7, 0.2, 0.05, 0.05]  # S, E, I, R
t = np.linspace(0, 50, 1000)

# Solve equations
solution = odeint(model, y0, t, args=(params,))
S, E, I, R = solution.T

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b-', label='Latent Karma (S)')
plt.plot(t, E, 'y-', label='Activated Karma (E)')
plt.plot(t, I, 'r-', label='Manifested Karma (I)')
plt.plot(t, R, 'g-', label='Resolved Karma (R)')
plt.title('SEIRS-Karma Dynamics')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()
plt.grid(True)
plt.savefig('seirs_karma.png')  # Will save in the same directory
plt.show()
