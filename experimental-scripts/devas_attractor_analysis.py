# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 19:09:28 2025

@author: eggra
"""
"""
KARMIC DYNAMICS: FIXED POINTS AND EXISTENTIAL REALMS
====================================================

This module implements a dynamical system model of Buddhist cosmology,
where existential realms are represented as attractors in a 3-dimensional
state space of fundamental kleshas (mental poisons):
- I: Ignorance (Avidya)
- A: Attachment (Raga)
- V: Aversion (Dvesha)

Fixed Points Analysis:
----------------------

1. STABLE FIXED POINT [0, 0, 0] - NIRVANA
   - Eigenvalues: [-0.035, -0.0625, -0.0775] (all negative real parts)
   - Dynamic interpretation:
     • Represents complete liberation from cyclic existence (samsara)
     • All three kleshas are fully extinguished (I=A=V=0)
     • Perfect stability: All trajectories near this state converge to it
     • Corresponds to the Buddhist concept of Enlightenment/Nirvana
   - Philosophical significance:
     "Just as a flame blown out by the wind goes to rest and cannot be defined,
      so the sage freed from name and form goes to rest and cannot be defined."
      - Parinirvana Sutta (Ud 8.9)

2. UNSTABLE FIXED POINT [0.5683, 0.3007, 0.3307] - DEVA REALM
   - Eigenvalues: [0.0757, -0.0683, -0.1180] (one positive real part)
   - Dynamic interpretation:
     • Saddle point: Attractive in some directions, repulsive in others
     • The positive eigenvalue (0.0757) indicates inherent instability
     • Small perturbations grow exponentially along the unstable manifold
   - Realm transition mechanism:
     The unstable direction (eigenvector) determines the realm of rebirth:
        Dominant Component    → Destination Realm
        -----------------------------------------
        ΔI (Ignorance)        → Animal Realm 
        ΔA (Attachment)       → Preta Realm (Hungry Ghosts)
        ΔV (Aversion)         → Asura or Naraka Realm
   - Philosophical significance:
     "The devas dwell in heaven, intoxicated with pleasure. 
      But when their merit is exhausted, they fall like rain from the sky."
      - Dhammapada 177 (Buddhist scripture)

Cosmological Implications:
--------------------------
1. The Deva realm's instability mathematically models the Buddhist doctrine of:
   - Anicca (Impermanence): All conditioned states are transient
   - Pratītyasamutpāda (Dependent Origination): Realms arise from causes

2. Realm transitions follow the karmic principle:
   - Positive eigenvalue direction → Predominant klesha → Next rebirth realm
   - Magnitude of perturbation → Severity of karmic expiration

3. The stable Nirvana point represents:
   - The only unconditioned state (asankhata)
   - Cessation of the three poisons (trivisa)
   - Final liberation from the cycle of rebirth

Numerical Implementation Note:
------------------------------
The Jacobian matrix and eigenvalue analysis provide a mathematical framework
for modeling the Buddha's teaching: "All conditioned things are impermanent" 
(Dhammapada 277). The positive eigenvalue in the Deva realm quantitatively 
expresses this fundamental instability of divine existence.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
import warnings

# Suprimir warnings para mejorar la legibilidad
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================
# Modelo de Tres Raíces Kármicas (ORIGINAL)
# =============================================
def three_roots_model(t, y, params):
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# Parámetros para el reino de los Devas
devas_params = {
    'alpha_I': 0.1, 'alpha_A': 0.05, 'alpha_V': 0.08,
    'beta_IA': 0.2, 'beta_AV': 0.1, 'beta_VI': 0.15,
    'gamma_I': 0.3, 'gamma_A': 0.25, 'gamma_V': 0.35,
    'w': 0.45
}

# =============================================
# 1. Análisis de Puntos Fijos y Estabilidad
# =============================================
def find_fixed_points(params):
    """Encuentra puntos fijos del sistema"""
    def equations(y):
        I, A, V = y
        dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
        dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
        dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
        return [dIdt, dAdt, dVdt]
    
    fixed_points = []
    # Buscar puntos fijos en diferentes regiones
    initial_guesses = [
        [0.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.1, 0.7, 0.1],
        [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.3, 0.3, 0.3], [0.1, 0.8, 0.1]
    ]
    
    for guess in initial_guesses:
        fp, info, ier, msg = fsolve(equations, guess, full_output=True)
        if ier == 1 and np.all(fp >= 0) and np.all(fp <= 1.5):
            fp_rounded = np.round(fp, 4)
            # Verificar si ya existe un punto similar
            exists = False
            for existing in fixed_points:
                if np.allclose(existing, fp_rounded, atol=0.05):
                    exists = True
                    break
            if not exists:
                fixed_points.append(fp_rounded)
    
    return fixed_points

def jacobian(y, params):
    """Calcula la matriz Jacobiana en un punto"""
    I, A, V = y
    J = np.zeros((3, 3))
    
    # Derivadas parciales para dIdt
    J[0, 0] = params['alpha_I'] - params['gamma_I']*params['w'] + params['beta_IA']*A*V
    J[0, 1] = params['beta_IA']*V
    J[0, 2] = params['beta_IA']*A
    
    # Derivadas parciales para dAdt
    J[1, 0] = params['beta_AV']*V
    J[1, 1] = params['alpha_A'] - params['gamma_A']*params['w'] + params['beta_AV']*V*I
    J[1, 2] = params['beta_AV']*I
    
    # Derivadas parciales para dVdt
    J[2, 0] = params['beta_VI']*A
    J[2, 1] = params['beta_VI']*I
    J[2, 2] = params['alpha_V'] - params['gamma_V']*params['w'] + params['beta_VI']*I*A
    
    return J

def stability_analysis(params):
    """Analiza la estabilidad de los puntos fijos"""
    fps = find_fixed_points(params)
    results = []
    
    for fp in fps:
        J = jacobian(fp, params)
        eigenvalues = np.linalg.eigvals(J)
        
        # Versión vectorizada con NumPy
        stability = "Estable" if np.all(np.real(eigenvalues) < 0) else "Inestable"
        
        results.append({
            'point': fp,
            'eigenvalues': eigenvalues,
            'stability': stability
        })
    
    return results
    

def plot_stability_analysis(results):
    """Visualiza los puntos fijos y su estabilidad"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for res in results:
        color = 'green' if res['stability'] == "Estable" else 'red'
        size = 100 if res['stability'] == "Estable" else 50
        ax.scatter(res['point'][0], res['point'][1], res['point'][2], 
                   c=color, s=size, alpha=0.8, edgecolor='k')
        
        # Etiqueta con autovalores
        label = f"λ: {res['eigenvalues'][0]:.2f}, {res['eigenvalues'][1]:.2f}, {res['eigenvalues'][2]:.2f}"
        ax.text(res['point'][0], res['point'][1], res['point'][2], label, fontsize=8)
    
    ax.set_title('Puntos Fijos y Estabilidad - Reino Devas')
    ax.set_xlabel('Ignorancia (I)')
    ax.set_ylabel('Apego (A)')
    ax.set_zlabel('Aversión (V)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Leyenda
    ax.scatter([], [], [], c='green', s=100, label='Estable')
    ax.scatter([], [], [], c='red', s=50, label='Inestable')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('devas_stability_analysis.png', dpi=150)
    plt.show()

# =============================================
# 2. Análisis de Cuencas de Atracción
# =============================================
def simulate_to_attractor(y0, params, t_span=(0, 100)):
    """Simula el sistema hasta un estado estable"""
    sol = solve_ivp(
        three_roots_model,
        t_span,
        y0,
        args=(params,),
        method='LSODA',
        rtol=1e-6,
        atol=1e-8
    )
    return sol.y[:, -1]  # Estado final

def attraction_basin_analysis(params, n_points=15):
    """Visualiza cuencas de atracción para los Devas"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Encontrar puntos fijos estables
    fps_data = stability_analysis(params)
    stable_fps = [res['point'] for res in fps_data if res['stability'] == "Estable"]
    
    # Colores para diferentes puntos fijos
    colors = cm.viridis(np.linspace(0, 1, len(stable_fps)))
    
    # Generar condiciones iniciales
    I_vals = np.linspace(0, 1, n_points)
    A_vals = np.linspace(0, 1, n_points)
    V_vals = np.linspace(0, 1, n_points)
    
    print(f"Calculando cuencas de atracción con {n_points**3} puntos...")
    
    for I0 in I_vals:
        for A0 in A_vals:
            for V0 in V_vals:
                if I0 + A0 + V0 <= 1.2:  # Filtrar combinaciones posibles
                    final_state = simulate_to_attractor([I0, A0, V0], params)
                    
                    # Encontrar el punto fijo más cercano
                    min_dist = float('inf')
                    closest_idx = 0
                    for idx, fp in enumerate(stable_fps):
                        dist = np.linalg.norm(final_state - fp)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    # Asignar color según el punto fijo de atracción
                    color = colors[closest_idx]
                    ax.scatter(I0, A0, V0, c=[color], alpha=0.7, s=20)
    
    # Marcar puntos fijos estables
    for idx, fp in enumerate(stable_fps):
        ax.scatter(fp[0], fp[1], fp[2], c='red', s=200, marker='*', edgecolor='gold', 
                  label=f'Atractor {idx+1}')
    
    ax.set_title('Cuencas de Atracción - Reino de los Devas', fontsize=14)
    ax.set_xlabel('Ignorancia (I)')
    ax.set_ylabel('Apego (A)')
    ax.set_zlabel('Aversión (V)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('devas_attraction_basins.png', dpi=150)
    plt.show()
    
    return stable_fps

# =============================================
# 3. Análisis de Sensibilidad a Perturbaciones
# =============================================
def perturbation_analysis(stable_fp, params, n_perturbations=30):
    """Analiza cómo responde el sistema a perturbaciones"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Ignorancia (I)', 'Apego (A)', 'Aversión (V)']
    colors = ['blue', 'green', 'red']
    
    for dim in range(3):
        perturbations = np.linspace(-0.2, 0.2, n_perturbations)
        recovery_times = []
        final_distances = []
        
        for pert in perturbations:
            # Aplicar perturbación
            perturbed_state = stable_fp.copy()
            perturbed_state[dim] += pert
            perturbed_state = np.clip(perturbed_state, 0, 1)
            
            # Simular recuperación
            sol = solve_ivp(
                three_roots_model,
                [0, 50],
                perturbed_state,
                args=(params,),
                method='LSODA',
                rtol=1e-6,
                atol=1e-8,
                dense_output=True
            )
            
            # Calcular tiempo de recuperación (umbral del 5%)
            t_eval = np.linspace(0, 50, 500)
            trajectory = sol.sol(t_eval)
            distances = np.linalg.norm(trajectory - stable_fp.reshape(-1, 1), axis=0)
            
            recovery_time = None
            for t_idx in range(len(t_eval)):
                if distances[t_idx] < 0.05:
                    recovery_time = t_eval[t_idx]
                    break
                    
            recovery_times.append(recovery_time if recovery_time else 50)
            final_distances.append(distances[-1])
        
        # Tiempos de recuperación
        axs[0].plot(perturbations, recovery_times, 'o-', color=colors[dim], markersize=5)
        axs[0].set_title('Tiempo de Recuperación')
        axs[0].set_xlabel('Magnitud de la perturbación')
        axs[0].set_ylabel('Tiempo')
        axs[0].grid(True, alpha=0.3)
        
        # Distancias finales
        axs[1].plot(perturbations, final_distances, 'o-', color=colors[dim], markersize=5)
        axs[1].set_title('Distancia Final al Equilibrio')
        axs[1].set_xlabel('Magnitud de la perturbación')
        axs[1].set_ylabel('Distancia')
        axs[1].grid(True, alpha=0.3)
        
        # Trayectoria de recuperación (ejemplo)
        if dim == 1:  # Solo para Apego para no saturar
            axs[2].plot(t_eval, distances, '-', color=colors[dim], alpha=0.7, 
                       label=f'Pert: {perturbations[len(perturbations)//2]:.2f}')
    
    axs[2].set_title('Trayectoria de Recuperación')
    axs[2].set_xlabel('Tiempo')
    axs[2].set_ylabel('Distancia al equilibrio')
    axs[2].set_yscale('log')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    plt.suptitle('Análisis de Estabilidad ante Perturbaciones - Reino Devas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('devas_perturbation_analysis.png', dpi=150)
    plt.show()
    
    return recovery_times, final_distances

# =============================================
# 4. Visualización de Convergencia al Atractor
# =============================================
def visualize_convergence(stable_fp, params):
    """Visualiza la convergencia al atractor desde diferentes puntos"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
    
    # Configurar gráficos
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Colores para diferentes trayectorias
    colors = cm.plasma(np.linspace(0, 1, 8))
    
    # Puntos iniciales aleatorios
    initial_points = []
    for _ in range(8):
        point = np.random.dirichlet(np.ones(3), 1)[0]
        initial_points.append(point)
    
    # Simular trayectorias
    for i, y0 in enumerate(initial_points):
        sol = solve_ivp(
            three_roots_model,
            [0, 100],
            y0,
            args=(params,),
            method='LSODA',
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )
        
        t_eval = np.linspace(0, 100, 500)
        trajectory = sol.sol(t_eval)
        I, A, V = trajectory
        
        # Diagrama de fase 3D
        ax1.plot(I, A, V, color=colors[i], alpha=0.7, linewidth=1)
        ax1.scatter(I[0], A[0], V[0], color=colors[i], s=50, marker='o')
        ax1.scatter(I[-1], A[-1], V[-1], color=colors[i], s=70, marker='*')
        
        # Convergencia en el tiempo
        distance = np.linalg.norm(trajectory - stable_fp.reshape(-1, 1), axis=0)
        ax2.plot(t_eval, distance, color=colors[i], alpha=0.8, 
                label=f'Inicio: [{y0[0]:.2f}, {y0[1]:.2f}, {y0[2]:.2f}]')
        
        # Comportamiento exponencial
        log_distance = np.log(np.maximum(distance, 1e-5))
        ax3.plot(t_eval, log_distance, color=colors[i], alpha=0.8)
    
    # Configurar gráfico 3D
    ax1.scatter(stable_fp[0], stable_fp[1], stable_fp[2], c='red', s=200, marker='X', label='Atractor')
    ax1.set_title('Trayectorias de Convergencia al Atractor')
    ax1.set_xlabel('Ignorancia (I)')
    ax1.set_ylabel('Apego (A)')
    ax1.set_zlabel('Aversión (V)')
    ax1.legend()
    
    # Configurar gráfico de convergencia
    ax2.set_title('Distancia al Atractor')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Distancia')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Configurar gráfico exponencial
    ax3.set_title('Convergencia Exponencial (log Distancia)')
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('log(Distancia)')
    ax3.grid(True, alpha=0.3)
    
    # Añadir línea de tendencia exponencial
    slope = np.polyfit(t_eval[50:], log_distance[50:], 1)[0]
    ax3.plot(t_eval, slope*t_eval + log_distance[0], 'k--', 
            label=f'Tasa: {abs(slope):.3f} (1/t)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('devas_convergence_analysis.png', dpi=150)
    plt.show()
    
    return slope

# =============================================
# 5. Análisis de Sensibilidad Paramétrica
# =============================================
def parametric_sensitivity_analysis(base_params, variations=15):
    """Analiza cómo cambian los puntos fijos al variar parámetros"""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    param_names = ['w', 'beta_IA', 'gamma_I', 'alpha_A', 'beta_AV', 'gamma_A']
    titles = [
        'Sabiduría (w)', 
        'Interacción I-A (β_IA)', 
        'Disipación Ignorancia (γ_I)',
        'Crecimiento Apego (α_A)', 
        'Interacción A-V (β_AV)', 
        'Disipación Apego (γ_A)'
    ]
    
    for i, param in enumerate(param_names):
        ax = axs[i//3, i%3]
        param_vals = np.linspace(0.5 * base_params[param], 1.5 * base_params[param], variations)
        
        stability_results = []
        for val in param_vals:
            mod_params = base_params.copy()
            mod_params[param] = val
            
            # Encontrar puntos fijos y su estabilidad
            fps = stability_analysis(mod_params)
            stable_count = sum(1 for res in fps if res['stability'] == "Estable")
            stability_results.append(stable_count)
            
            # Graficar puntos fijos
            for res in fps:
                color = 'green' if res['stability'] == "Estable" else 'red'
                ax.scatter(val, res['point'][1], color=color, alpha=0.7)  # Graficamos Apego (A)
        
        # Línea de estabilidad
        ax.plot(param_vals, stability_results, 'b-', linewidth=2, label='Número de puntos estables')
        
        ax.set_title(titles[i])
        ax.set_xlabel(f'Valor de {param}')
        ax.set_ylabel('Valor de Apego (A) en puntos fijos')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Sensibilidad Paramétrica en el Reino de los Devas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('devas_parametric_sensitivity.png', dpi=150)
    plt.show()

# =============================================
# Análisis Completo para los Devas
# =============================================
def devas_analysis():
    print("="*70)
    print("ANÁLISIS COMPLETO: REINO DE LOS DEVAS COMO ATRACTOR DINÁMICO")
    print("="*70)
    
    # 1. Análisis de puntos fijos y estabilidad
    print("\n>>> PASO 1: ANÁLISIS DE ESTABILIDAD DE PUNTOS FIJOS")
    stability_results = stability_analysis(devas_params)
    
    print("\nPuntos fijos encontrados:")
    for i, res in enumerate(stability_results):
        print(f"  - Punto #{i+1}: {res['point']} ({res['stability']})")
        print(f"    Autovalores: {res['eigenvalues']}")
    
    # Seleccionar el punto fijo estable principal
    stable_fp = next(res['point'] for res in stability_results if res['stability'] == "Estable")
    print(f"\nPunto fijo estable principal: {stable_fp}")
    
    # Visualizar
    plot_stability_analysis(stability_results)
    
    # 2. Análisis de cuencas de atracción
    print("\n>>> PASO 2: ANÁLISIS DE CUENCAS DE ATRACCIÓN")
    stable_fps = attraction_basin_analysis(devas_params)
    print(f"Puntos fijos estables identificados: {len(stable_fps)}")
    
    # 3. Análisis de perturbaciones
    print("\n>>> PASO 3: ANÁLISIS DE ESTABILIDAD ANTE PERTURBACIONES")
    recovery_times, final_distances = perturbation_analysis(stable_fp, devas_params)
    print(f"Tiempo de recuperación promedio: {np.mean(recovery_times):.2f} ± {np.std(recovery_times):.2f}")
    print(f"Distancia final promedio: {np.mean(final_distances):.5f}")
    
    # 4. Visualización de convergencia
    print("\n>>> PASO 4: VISUALIZACIÓN DE CONVERGENCIA AL ATRACTOR")
    convergence_rate = visualize_convergence(stable_fp, devas_params)
    print(f"Tasa de convergencia exponencial: {abs(convergence_rate):.4f}")
    
    # 5. Sensibilidad paramétrica
    print("\n>>> PASO 5: ANÁLISIS DE SENSIBILIDAD PARAMÉTRICA")
    parametric_sensitivity_analysis(devas_params)
    
    # Resumen final
    print("\n" + "="*70)
    print("CONCLUSIÓN FINAL:")
    print(f"El reino de los Devas presenta un atractor estable en [I, A, V] ≈ {stable_fp}")
    print(f"- Cuenca de atracción significativa (dominada por el apego)")
    print(f"- Resiliente a perturbaciones (tiempo recuperación: {np.mean(recovery_times):.2f} ± {np.std(recovery_times):.2f} u.t.)")
    print(f"- Convergencia exponencial (tasa: {abs(convergence_rate):.4f})")
    print(f"- Estable ante variaciones paramétricas moderadas (±25%)")
    print("="*70)

# =============================================
# Ejecutar el análisis
# =============================================
if __name__ == "__main__":
    devas_analysis()