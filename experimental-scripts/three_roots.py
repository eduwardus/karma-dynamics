# scripts/three_roots.py
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
from scipy.integrate import solve_ivp  # Changed to solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Model parameters (example values based on Table 1 from the paper)
params = {
    # Self-reinforcement rates
    'alpha_I': 0.3,
    'alpha_A': 0.25,
    'alpha_V': 0.35,
    
    # Coupling coefficients
    'beta_IA': 0.6,   # Attachment + Aversion → Ignorance
    'beta_AV': 0.7,   # Aversion + Ignorance → Attachment
    'beta_VI': 0.5,   # Ignorance + Attachment → Aversion
    
    # Wisdom sensitivity
    'gamma_I': 0.4,
    'gamma_A': 0.35,
    'gamma_V': 0.45,
    
    # Wisdom factor (can be constant or time-dependent)
    'w': 0.2
}

# Three Karmic Roots Model Equations with numerical stability improvements
def model(t, y, params):  # Changed signature for solve_ivp
    I, A, V = y
    
    # Apply numerical stability constraints
    I = max(I, 1e-10)
    A = max(A, 1e-10)
    V = max(V, 1e-10)
    
    dIdt = params['alpha_I'] * I + params['beta_IA'] * A * V - params['gamma_I'] * params['w'] * I
    dAdt = params['alpha_A'] * A + params['beta_AV'] * V * I - params['gamma_A'] * params['w'] * A
    dVdt = params['alpha_V'] * V + params['beta_VI'] * I * A - params['gamma_V'] * params['w'] * V
    
    # Scale derivatives to prevent numerical instability
    return [
        dIdt / (1 + abs(dIdt)),
        dAdt / (1 + abs(dAdt)),
        dVdt / (1 + abs(dVdt))
    ]

# Ensure directory exists function
def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# Initial conditions (example: ignorance-predominant state)
y0 = [0.8, 0.1, 0.1]  # I, A, V
t = np.linspace(0, 50, 5000)  # Simulation time

# Solve differential equations with improved solver
solution = solve_ivp(
    fun=model,
    t_span=[t[0], t[-1]],
    y0=y0,
    t_eval=t,
    args=(params,),
    method='BDF',  # Stiff system solver
    rtol=1e-6,
    atol=1e-8
)

# Extract solution
I, A, V = solution.y

# Create plots
plt.figure(figsize=(12, 8))

# Time series plot
plt.subplot(2, 2, 1)
plt.plot(t, I, 'b-', label='Ignorance (I)')
plt.plot(t, A, 'g-', label='Attachment (A)')
plt.plot(t, V, 'r-', label='Aversion (V)')
plt.title('Evolution of the Three Karmic Roots')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)

# 2D Phase Diagram: Ignorance vs Attachment
plt.subplot(2, 2, 2)
plt.plot(I, A, 'm-')
plt.title('Phase Space: Ignorance vs Attachment')
plt.xlabel('Ignorance (I)')
plt.ylabel('Attachment (A)')
plt.grid(True)

# 2D Phase Diagram: Ignorance vs Aversion
plt.subplot(2, 2, 3)
plt.plot(I, V, 'c-')
plt.title('Phase Space: Ignorance vs Aversion')
plt.xlabel('Ignorance (I)')
plt.ylabel('Aversion (V)')
plt.grid(True)

# 2D Phase Diagram: Attachment vs Aversion
plt.subplot(2, 2, 4)
plt.plot(A, V, 'y-')
plt.title('Phase Space: Attachment vs Aversion')
plt.xlabel('Attachment (A)')
plt.ylabel('Aversion (V)')
plt.grid(True)

plt.tight_layout()

# Save 2D plot with directory handling
save_path_2d = 'figs/three_roots_2d.png'
ensure_directory_exists(save_path_2d)
plt.savefig(save_path_2d)

# 3D Phase Diagram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(I, A, V, 'b-', linewidth=0.8)
ax.set_title('Attractor of the Three Karmic Roots')
ax.set_xlabel('Ignorance (I)')
ax.set_ylabel('Attachment (A)') 
ax.set_zlabel('Aversion (V)')

# Save 3D plot with directory handling
save_path_3d = 'figs/three_roots_3d.png'
ensure_directory_exists(save_path_3d)
plt.savefig(save_path_3d)
plt.show()

# Enlightenment Condition Analysis
def enlightenment_condition(params):
    """Calculates the enlightenment condition according to equation (8) from the paper"""
    # Calculate critical value for each root
    w_critical_I = (params['alpha_I'] + 0.5 * abs(params['beta_IA'])) / params['gamma_I']
    w_critical_A = (params['alpha_A'] + 0.5 * abs(params['beta_AV'])) / params['gamma_A']
    w_critical_V = (params['alpha_V'] + 0.5 * abs(params['beta_VI'])) / params['gamma_V']
    
    # The maximum of these values is the minimum w required for enlightenment
    w_min = max(w_critical_I, w_critical_A, w_critical_V)
    
    return w_min

# Calculate and display enlightenment condition
w_min_required = enlightenment_condition(params)
print(f"\nEnlightenment Condition Analysis:")
print(f"  Current wisdom (w): {params['w']}")
print(f"  Minimum wisdom required for enlightenment: {w_min_required:.4f}")

if params['w'] > w_min_required:
    print("  STATE: Conditions for enlightenment SATISFIED")
else:
    print("  STATE: Conditions for enlightenment NOT satisfied (persistent samsara)")
    
# Check if the system converges to zero (enlightenment state)
final_values = solution.y[:, -1]  # Last point in the solution
tolerance = 1e-3
if all(abs(val) < tolerance for val in final_values):
    print("  RESULT: System converges to enlightenment state (0,0,0)")
else:
    print(f"  RESULT: System does NOT converge to zero (final values: I={final_values[0]:.4f}, A={final_values[1]:.4f}, V={final_values[2]:.4f})")