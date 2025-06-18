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
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Modelo de las Tres Raíces Kármicas
def three_roots_model(y, t, params):
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# Condición de Iluminación (Ecuación 8 del paper)
def enlightenment_condition(params):
    """Calcula el umbral mínimo de sabiduría para la iluminación"""
    w_critical_I = (params['alpha_I'] + 0.5*(abs(params['beta_IA'])) / params['gamma_I']
    w_critical_A = (params['alpha_A'] + 0.5*(abs(params['beta_AV'])) / params['gamma_A']
    w_critical_V = (params['alpha_V'] + 0.5*(abs(params['beta_VI'])) / params['gamma_V']
    return max(w_critical_I, w_critical_A, w_critical_V)

# Simulación de convergencia
def simulate_enlightenment(w_value, params_base, initial_conditions, t):
    """Simula la dinámica para un valor de sabiduría dado"""
    params = params_base.copy()
    params['w'] = w_value
    solution = odeint(three_roots_model, initial_conditions, t, args=(params,))
    return solution

# Configuración base de parámetros
BASE_PARAMS = {
    'alpha_I': 0.3, 'alpha_A': 0.25, 'alpha_V': 0.35,
    'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5,
    'gamma_I': 0.4, 'gamma_A': 0.35, 'gamma_V': 0.45
}

# 1. Análisis de convergencia para diferentes niveles de w
def analyze_wisdom_threshold():
    """Muestra cómo w afecta la convergencia a K=0"""
    print("\n" + "="*70)
    print("ANÁLISIS DE CONVERGENCIA A LA ILUMINACIÓN")
    print("="*70)
    
    # Calcular umbral crítico
    w_critical = enlightenment_condition(BASE_PARAMS)
    print(f"Umbral crítico de sabiduría (w_c): {w_critical:.4f}")
    
    # Configurar simulaciones
    w_values = [w_critical * 0.7, w_critical * 0.99, w_critical, w_critical * 1.05]
    labels = ['Sabiduría insuficiente (w = 0.7·wc)', 
              'Sabiduría crítica (w = 0.99·wc)',
              'Sabiduría umbral (w = wc)',
              'Sabiduría suficiente (w = 1.05·wc)']
    colors = ['red', 'orange', 'yellow', 'green']
    t = np.linspace(0, 100, 5000)
    y0 = [0.6, 0.2, 0.2]  # Estado inicial
    
    plt.figure(figsize=(12, 8))
    
    for i, w in enumerate(w_values):
        # Simular dinámica
        solution = simulate_enlightenment(w, BASE_PARAMS, y0, t)
        I, A, V = solution.T
        intensity = np.sqrt(I**2 + A**2 + V**2)  # Norma euclidiana
        
        # Visualizar intensidad total
        plt.subplot(2, 2, i+1)
        plt.plot(t, intensity, color=colors[i], linewidth=1.5)
        plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5)
        
        # Etiquetas y estilo
        plt.title(labels[i], fontsize=10)
        plt.xlabel('Tiempo')
        plt.ylabel('Intensidad Kármica ||K||')
        plt.yscale('log')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Resaltar comportamiento asintótico
        final_intensity = intensity[-1]
        plt.annotate(f'Valor final: {final_intensity:.2e}', 
                    xy=(0.7, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.suptitle('Convergencia a la Iluminación para Diferentes Niveles de Sabiduría', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figs/wisdom_convergence.png', dpi=150)
    plt.show()

# 2. Análisis de cuencas de atracción
def analyze_attraction_basins():
    """Mapea las cuencas de atracción para diferentes reinos"""
    print("\nANÁLISIS DE CUENCAS DE ATRACCIÓN")
    
    # Configurar parámetros con w crítico
    w_critical = enlightenment_condition(BASE_PARAMS)
    params = BASE_PARAMS.copy()
    params['w'] = w_critical * 0.95  # Por debajo del umbral
    
    # Configurar espacio de estado
    I_range = np.linspace(0.01, 0.99, 20)
    A_range = np.linspace(0.01, 0.99, 20)
    V_fixed = 0.2  # Valor fijo de aversión
    
    # Matriz para almacenar resultados
    basin_matrix = np.zeros((len(I_range), len(A_range)))
    t = np.linspace(0, 100, 5000)
    
    # Simular para cada punto inicial
    for i, I0 in enumerate(I_range):
        for j, A0 in enumerate(A_range):
            y0 = [I0, A0, V_fixed]
            solution = odeint(three_roots_model, y0, t, args=(params,))
            final_state = solution[-1]
            
            # Determinar veneno dominante
            dominant_idx = np.argmax(final_state)
            basin_matrix[j, i] = dominant_idx  # Transponer para orientación correcta
    
    # Visualizar cuencas de atracción
    plt.figure(figsize=(10, 8))
    plt.imshow(basin_matrix, extent=[I_range.min(), I_range.max(), A_range.min(), A_range.max()], 
              origin='lower', cmap='viridis', aspect='auto')
    
    # Configurar ejes y leyenda
    plt.xlabel('Ignorancia Inicial (I0)')
    plt.ylabel('Apego Inicial (A0)')
    plt.title(f'Cuencas de Atracción (V0 fijo = {V_fixed}, w = {params["w"]:.2f})')
    
    # Crear leyenda personalizada
    realm_colors = {0: 'blue', 1: 'green', 2: 'red'}
    realm_labels = {0: 'Ignorancia', 1: 'Apego', 2: 'Aversión'}
    patches = [patches.Patch(color=color, label=realm_labels[i]) for i, color in realm_colors.items()]
    plt.legend(handles=patches, loc='upper right')
    
    plt.colorbar(label='Veneno Dominante Final', ticks=[0, 1, 2])
    plt.grid(False)
    plt.savefig('../figs/attraction_basins.png', dpi=150)
    plt.show()
    
    # Calcular tamaño de cuencas
    unique, counts = np.unique(basin_matrix, return_counts=True)
    total_points = basin_matrix.size
    print("\nDistribución de Cuencas de Atracción:")
    for realm, count in zip(unique, counts):
        print(f"  {realm_labels[int(realm)]}: {count/total_points:.1%} del espacio de fase")

# 3. Visualización 3D de convergencia
def plot_3d_convergence():
    """Muestra trayectorias en el espacio de fase 3D"""
    print("\nVISUALIZACIÓN 3D DE CONVERGENCIA")
    
    # Configurar w por encima del umbral
    w_critical = enlightenment_condition(BASE_PARAMS)
    params = BASE_PARAMS.copy()
    params['w'] = w_critical * 1.1
    
    # Configurar condiciones iniciales
    initial_conditions = [
        [0.8, 0.1, 0.1],  # Predominio ignorancia
        [0.1, 0.8, 0.1],  # Predominio apego
        [0.1, 0.1, 0.8]   # Predominio aversión
    ]
    colors = ['blue', 'green', 'red']
    t = np.linspace(0, 50, 5000)
    
    # Configurar figura 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Simular y visualizar cada trayectoria
    for i, y0 in enumerate(initial_conditions):
        solution = odeint(three_roots_model, y0, t, args=(params,))
        I, A, V = solution.T
        
        # Trayectoria completa
        ax.plot(I, A, V, color=colors[i], linewidth=0.8, alpha=0.7)
        
        # Punto inicial
        ax.scatter(I[0], A[0], V[0], color=colors[i], s=50, edgecolor='k')
        
        # Punto final
        ax.scatter(I[-1], A[-1], V[-1], color='gold', s=100, marker='*')
    
    # Configuración estética
    ax.set_title(f'Trayectorias hacia la Iluminación (w = {params["w"]:.2f})', fontsize=14)
    ax.set_xlabel('Ignorancia (I)')
    ax.set_ylabel('Apego (A)')
    ax.set_zlabel('Aversión (V)')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_zlim(0, 0.8)
    
    # Crear leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Ignorancia dominante inicial'),
        Line2D([0], [0], color='green', lw=2, label='Apego dominante inicial'),
        Line2D([0], [0], color='red', lw=2, label='Aversión dominante inicial'),
        Line2D([0], [0], marker='*', color='gold', markersize=10, label='Estado final', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig('../figs/enlightenment_convergence_3d.png', dpi=150)
    plt.show()

# Función principal
def main():
    analyze_wisdom_threshold()
    analyze_attraction_basins()
    plot_3d_convergence()

if __name__ == "__main__":
    main()
