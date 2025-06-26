# scripts/enlightenment.py
"""
ENLIGHTENMENT CONVERGENCE ANALYSIS

DISCLAIMER: 
This simulation explores the mathematical metaphor of enlightenment 
as a dynamical fixed point (K=0). It models the theoretical conditions 
under which karmic patterns dissipate, but makes no claims about:

1. The ontological reality of nirvana
2. The phenomenological experience of awakening
3. The soteriological efficacy of contemplative practices

The model is:
- A formal representation of conceptual relationships
- Subject to parameter sensitivity
- Limited by its own mathematical assumptions

See Section 7 (Discussion) of the accompanying paper for the ethical framework.
"""

import numpy as np
from scipy.integrate import solve_ivp  # Cambiamos a solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import os
import warnings

# Three Karmic Roots Model (con protección numérica)
def three_roots_model(t, y, params):  # Cambiamos el orden de los argumentos
    I, A, V = y
    
    # Protección contra valores negativos o no finitos
    I = max(I, 1e-10)
    A = max(A, 1e-10)
    V = max(V, 1e-10)
    
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    
    # Suavizado para evitar divergencias
    return [
        dIdt / (1 + abs(dIdt)),
        dAdt / (1 + abs(dAdt)),
        dVdt / (1 + abs(dVdt))
    ]

# Enlightenment Condition (Equation 8 from the paper)
def enlightenment_condition(params):
    """Calculates the minimum wisdom threshold for enlightenment"""
    w_critical_I = (params['alpha_I'] + 0.5*(abs(params['beta_IA']))) / params['gamma_I']
    w_critical_A = (params['alpha_A'] + 0.5*(abs(params['beta_AV']))) / params['gamma_A']
    w_critical_V = (params['alpha_V'] + 0.5*(abs(params['beta_VI']))) / params['gamma_V']
    return max(w_critical_I, w_critical_A, w_critical_V)

# Convergence Simulation (mejorada)
def simulate_enlightenment(w_value, params_base, initial_conditions, t):
    """Simulates dynamics for a given wisdom value"""
    params = params_base.copy()
    params['w'] = w_value
    
    # Usamos solve_ivp con método implícito para sistemas rígidos
    solution = solve_ivp(
        fun=three_roots_model,
        t_span=[t[0], t[-1]],
        y0=initial_conditions,
        t_eval=t,
        args=(params,),
        method='BDF',  # Método para sistemas stiff
        rtol=1e-6,    # Tolerancia relativa
        atol=1e-8     # Tolerancia absoluta
    )
    
    return solution.y

# Base parameter configuration
BASE_PARAMS = {
    'alpha_I': 0.3, 'alpha_A': 0.25, 'alpha_V': 0.35,
    'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5,
    'gamma_I': 0.4, 'gamma_A': 0.35, 'gamma_V': 0.45
}

# Función para asegurar que exista el directorio
def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# 1. Convergence analysis for different w levels
def analyze_wisdom_threshold():
    """Shows how w affects convergence to K=0"""
    print("\n" + "="*70)
    print("ENLIGHTENMENT CONVERGENCE ANALYSIS")
    print("="*70)
    
    # Calculate critical threshold
    w_critical = enlightenment_condition(BASE_PARAMS)
    print(f"Critical wisdom threshold (w_c): {w_critical:.4f}")
    
    # Configure simulations
    w_values = [w_critical * 0.7, w_critical * 0.99, w_critical, w_critical * 1.05]
    labels = ['Insufficient Wisdom (w = 0.7·wc)', 
              'Critical Wisdom (w = 0.99·wc)',
              'Threshold Wisdom (w = wc)',
              'Sufficient Wisdom (w = 1.05·wc)']
    colors = ['red', 'orange', 'yellow', 'green']
    t = np.linspace(0, 100, 5000)
    y0 = [0.6, 0.2, 0.2]  # Initial state
    
    plt.figure(figsize=(12, 8))
    
    for i, w in enumerate(w_values):
        # Simulate dynamics
        solution = simulate_enlightenment(w, BASE_PARAMS, y0, t)
        I, A, V = solution
        
        # Calcular intensidad kármica con protección numérica
        intensity = np.sqrt(np.clip(I**2 + A**2 + V**2, 1e-10, None))
        
        # Visualize total intensity
        plt.subplot(2, 2, i+1)
        plt.plot(t, intensity, color=colors[i], linewidth=1.5)
        plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and style
        plt.title(labels[i], fontsize=10)
        plt.xlabel('Time')
        plt.ylabel('Karmic Intensity ||K||')
        plt.yscale('log')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Highlight asymptotic behavior
        final_intensity = intensity[-1]
        plt.annotate(f'Final value: {final_intensity:.2e}', 
                    xy=(0.7, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.suptitle('Convergence to Enlightenment for Different Wisdom Levels', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar con manejo de directorios
    save_path = 'figs/wisdom_convergence.png'  # Cambiamos a directorio local
    ensure_directory_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()

# 2. Attraction basins analysis
def analyze_attraction_basins():
    """Maps attraction basins for different realms"""
    print("\nATTRACTION BASINS ANALYSIS")
    
    # Configure parameters with critical w
    w_critical = enlightenment_condition(BASE_PARAMS)
    params = BASE_PARAMS.copy()
    params['w'] = w_critical * 0.95  # Below threshold
    
    # Configure state space
    I_range = np.linspace(0.01, 0.99, 15)  # Reducimos resolución para estabilidad
    A_range = np.linspace(0.01, 0.99, 15)
    V_fixed = 0.2  # Fixed aversion value
    
    # Matrix to store results
    basin_matrix = np.zeros((len(I_range), len(A_range)))
    t = np.linspace(0, 100, 1000)  # Menos puntos para mayor velocidad
    
    # Simulate for each initial point
    for i, I0 in enumerate(I_range):
        for j, A0 in enumerate(A_range):
            y0 = [I0, A0, V_fixed]
            try:
                solution = simulate_enlightenment(params['w'], params, y0, t)
                final_state = solution[:, -1]  # Tomamos el estado final
                
                # Determinar veneno dominante con protección
                final_state = np.clip(final_state, 1e-10, None)
                dominant_idx = np.argmax(final_state)
                basin_matrix[j, i] = dominant_idx
            except Exception as e:
                print(f"Error at (I={I0:.2f}, A={A0:.2f}): {str(e)}")
                basin_matrix[j, i] = -1  # Valor de error
    
    # Visualize attraction basins
    plt.figure(figsize=(10, 8))
    plt.imshow(basin_matrix, extent=[I_range.min(), I_range.max(), A_range.min(), A_range.max()], 
              origin='lower', cmap='viridis', aspect='auto')
    
    # Configure axes and legend
    plt.xlabel('Initial Ignorance (I0)')
    plt.ylabel('Initial Attachment (A0)')
    plt.title(f'Attraction Basins (V0 fixed = {V_fixed}, w = {params["w"]:.2f})')
    
    # Create custom legend
    realm_colors = {0: 'blue', 1: 'green', 2: 'red'}
    realm_labels = {0: 'Ignorance', 1: 'Attachment', 2: 'Aversion'}
    legend_patches = [patches.Patch(color=color, label=realm_labels[i]) for i, color in realm_colors.items()]
    plt.legend(handles=legend_patches, loc='upper right')
    
    plt.colorbar(label='Final Dominant Poison', ticks=[0, 1, 2])
    plt.grid(False)
    
    # Guardar con manejo de directorios
    save_path = 'figs/attraction_basins.png'
    ensure_directory_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    # Calculate basin sizes
    valid_points = basin_matrix >= 0
    unique, counts = np.unique(basin_matrix[valid_points], return_counts=True)
    total_points = np.sum(valid_points)
    
    print("\nAttraction Basin Distribution:")
    for realm, count in zip(unique, counts):
        print(f"  {realm_labels[int(realm)]}: {count/total_points:.1%} of phase space")

# 3. 3D convergence visualization
def plot_3d_convergence():
    """Shows trajectories in 3D phase space"""
    print("\n3D CONVERGENCE VISUALIZATION")
    
    # Configure w above threshold
    w_critical = enlightenment_condition(BASE_PARAMS)
    params = BASE_PARAMS.copy()
    params['w'] = w_critical * 1.1
    
    # Configure initial conditions
    initial_conditions = [
        [0.8, 0.1, 0.1],  # Ignorance predominance
        [0.1, 0.8, 0.1],  # Attachment predominance
        [0.1, 0.1, 0.8]   # Aversion predominance
    ]
    colors = ['blue', 'green', 'red']
    t = np.linspace(0, 50, 2000)  # Menos puntos para estabilidad
    
    # Configure 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simulate and visualize each trajectory
    for i, y0 in enumerate(initial_conditions):
        try:
            solution = simulate_enlightenment(params['w'], params, y0, t)
            I, A, V = solution
            
            # Trayectoria completa
            ax.plot(I, A, V, color=colors[i], linewidth=0.8, alpha=0.7)
            
            # Punto inicial
            ax.scatter(I[0], A[0], V[0], color=colors[i], s=50, edgecolor='k')
            
            # Punto final
            ax.scatter(I[-1], A[-1], V[-1], color='gold', s=100, marker='*')
        except Exception as e:
            print(f"Error in trajectory {i+1}: {str(e)}")
    
    # Aesthetic configuration
    ax.set_title(f'Trajectories toward Enlightenment (w = {params["w"]:.2f})', fontsize=14)
    ax.set_xlabel('Ignorance (I)')
    ax.set_ylabel('Attachment (A)')
    ax.set_zlabel('Aversion (V)')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_zlim(0, 0.8)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Initial ignorance dominance'),
        Line2D([0], [0], color='green', lw=2, label='Initial attachment dominance'),
        Line2D([0], [0], color='red', lw=2, label='Initial aversion dominance'),
        Line2D([0], [0], marker='*', color='gold', markersize=10, label='Final state', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Guardar con manejo de directorios
    save_path = 'figs/enlightenment_convergence_3d.png'
    ensure_directory_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.show()

# Main function
def main():
    # Suprimimos warnings específicos durante la ejecución
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        
        analyze_wisdom_threshold()
        analyze_attraction_basins()
        plot_3d_convergence()

if __name__ == "__main__":
    main()