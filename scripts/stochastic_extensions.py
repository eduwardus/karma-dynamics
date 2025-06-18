# scripts/stochastic_extensions.py
"""
STOCHASTIC KARMIC DYNAMICS SIMULATION

DISCLAIMER: 
This model explores the effects of randomness in karmic processes through:
- Stochastic Differential Equations (SDEs)
- Time-Delayed Karmic Maturation (DDEs)
- Perturbation Analysis

These implementations are mathematical thought experiments representing:
1. The inherent uncertainty in karmic outcomes
2. The delayed effects of actions
3. Resilience to external shocks

The models do NOT imply:
- Randomness in ultimate karmic principles
- Reduction of ethics to probabilities
- Negation of volitional responsibility

See Section 8 (Future Work) of the companion paper for context.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sdeint import itoint  # Requiere instalar sdeint: pip install sdeint
import ddeint  # Requiere instalar ddeint: pip install ddeint

# =============================================
# 1. Modelo Estocástico (SDE) para Karma Individual
# =============================================
def stochastic_seirs(y, t, params):
    """Drift function (deterministic part)"""
    S, E, I, R = y
    dS = params['xi']*(1-params['w'])*R - params['alpha']*S + params['lambda_val']*(R/(1+R))
    dE = params['alpha']*S - params['sigma']*E
    dI = params['sigma']*E - params['gamma']*I
    dR = params['gamma']*I - params['xi']*(1-params['w'])*R - params['lambda_val']*(R/(1+R))
    return np.array([dS, dE, dI, dR])

def noise_function(y, t, params):
    """Diffusion function (stochastic part)"""
    S, E, I, R = y
    # Intensidad de ruido proporcional al estado actual
    noise_S = params['sigma_S'] * S
    noise_E = params['sigma_E'] * E
    noise_I = params['sigma_I'] * I
    noise_R = params['sigma_R'] * R
    return np.diag([noise_S, noise_E, noise_I, noise_R])

def simulate_stochastic_seirs(params, y0, t):
    """Simula el modelo SEIRS-Karma estocástico"""
    # Convertir tiempo a array
    t_array = np.array(t)
    
    # Simulación SDE usando integración Ito
    result = itoint(stochastic_seirs, noise_function, y0, t_array, args=(params,))
    return result.T

# =============================================
# 2. Modelo con Retraso Temporal (DDE)
# =============================================
def delayed_three_roots(Y, t, params):
    """Modelo de Tres Raíces con retraso en la maduración kármica"""
    # Valores actuales
    I, A, V = Y(t)
    
    # Valores retardados (t - tau)
    tau = params['tau']
    I_tau, A_tau, V_tau = Y(t - tau)
    
    # Ecuaciones con efectos retardados
    dIdt = (params['alpha_I'] * I + 
            params['beta_IA'] * A_tau * V_tau - 
            params['gamma_I'] * params['w'] * I)
    
    dAdt = (params['alpha_A'] * A + 
            params['beta_AV'] * V_tau * I_tau - 
            params['gamma_A'] * params['w'] * A)
    
    dVdt = (params['alpha_V'] * V + 
            params['beta_VI'] * I_tau * A_tau - 
            params['gamma_V'] * params['w'] * V)
    
    return [dIdt, dAdt, dVdt]

# =============================================
# 3. Análisis de Perturbaciones
# =============================================
def apply_perturbation(solution, t, perturbation):
    """Aplica una perturbación al sistema"""
    perturbed = solution.copy()
    event_time, intensity, affected_var = perturbation
    
    # Encontrar índice más cercano al tiempo del evento
    idx = np.abs(t - event_time).argmin()
    
    # Aplicar perturbación
    if affected_var < perturbed.shape[0]:
        perturbed[affected_var, idx:] += intensity
    
    return perturbed

# =============================================
# Simulaciones Principales
# =============================================
def main():
    print("\n" + "="*70)
    print("STOCHASTIC EXTENSIONS OF KARMIC MODELS")
    print("="*70)
    print("Running: 1. Stochastic SEIRS  2. Delayed Three Roots  3. Perturbation Analysis")
    
    # Parámetros comunes
    t = np.linspace(0, 50, 5000)
    t_dde = np.linspace(0, 100, 5000)
    
    # 1. Simulación SEIRS Estocástica
    seirs_params = {
        'alpha': 0.3, 'sigma': 0.5, 'gamma': 0.4, 'xi': 0.2,
        'w': 0.3, 'lambda_val': 0.1,
        'sigma_S': 0.05, 'sigma_E': 0.07, 'sigma_I': 0.1, 'sigma_R': 0.08
    }
    y0_seirs = [0.7, 0.2, 0.05, 0.05]
    
    # Ejecutar múltiples trayectorias estocásticas
    num_trajectories = 5
    plt.figure(figsize=(12, 8))
    
    for i in range(num_trajectories):
        S, E, I, R = simulate_stochastic_seirs(seirs_params, y0_seirs, t)
        plt.plot(t, I, alpha=0.6, label=f'Trayectoria {i+1}' if i < 3 else "")
    
    # Comparar con solución determinista
    det_solution = odeint(
        lambda y, t: stochastic_seirs(y, t, seirs_params), 
        y0_seirs, t
    )
    plt.plot(t, det_solution[:, 2], 'k--', linewidth=2, label='Determinista')
    
    plt.title('Manifestación Kármica (I) con Ruido Estocástico')
    plt.xlabel('Tiempo')
    plt.ylabel('Karma Manifestado (I)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/stochastic_seirs.png', dpi=150)
    
    # 2. Simulación con Retraso Temporal
    delay_params = {
        'alpha_I': 0.3, 'alpha_A': 0.25, 'alpha_V': 0.35,
        'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5,
        'gamma_I': 0.4, 'gamma_A': 0.35, 'gamma_V': 0.45,
        'w': 0.2,
        'tau': 15.0  # Retraso de 15 unidades de tiempo
    }
    
    # Historia inicial (constante para t < 0)
    def history(t):
        return [0.5, 0.3, 0.2]
    
    # Resolver DDE
    solution_dde = ddeint.ddeint(delayed_three_roots, history, t_dde, args=(delay_params,))
    I_dde, A_dde, V_dde = solution_dde.T
    
    plt.figure(figsize=(12, 8))
    plt.plot(t_dde, I_dde, 'b-', label='Ignorancia (I)')
    plt.plot(t_dde, A_dde, 'g-', label='Apego (A)')
    plt.plot(t_dde, V_dde, 'r-', label='Aversión (V)')
    
    # Marcar efecto del retraso
    plt.axvline(x=delay_params['tau'], color='gray', linestyle='--', alpha=0.7)
    plt.text(delay_params['tau']+1, 0.9*max(V_dde), 'Primeros efectos\ndel retraso', fontsize=10)
    
    plt.title('Tres Raíces Kármicas con Retraso en la Maduración (τ=15)')
    plt.xlabel('Tiempo')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/delayed_karma.png', dpi=150)
    
    # 3. Análisis de Perturbaciones
    # Simular sistema determinista
    three_roots_params = {
        'alpha_I': 0.3, 'alpha_A': 0.25, 'alpha_V': 0.35,
        'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5,
        'gamma_I': 0.4, 'gamma_A': 0.35, 'gamma_V': 0.45,
        'w': 0.3
    }
    y0_roots = [0.4, 0.3, 0.3]
    solution_det = odeint(
        lambda y, t: three_roots_model(y, t, three_roots_params), 
        y0_roots, t
    ).T
    
    # Aplicar perturbaciones
    perturbations = [
        (20, 0.4, 0),   # Perturbación a Ignorancia en t=20
        (35, -0.3, 1),  # Perturbación negativa a Apego en t=35
        (45, 0.5, 2)    # Perturbación a Aversión en t=45
    ]
    
    plt.figure(figsize=(14, 10))
    variables = ['Ignorancia (I)', 'Apego (A)', 'Aversión (V)']
    colors = ['blue', 'green', 'red']
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        
        # Trayectoria original
        plt.plot(t, solution_det[i], color=colors[i], linestyle='-', label='Original')
        
        # Aplicar cada perturbación
        for j, pert in enumerate(perturbations):
            perturbed = apply_perturbation(solution_det, t, pert)
            plt.plot(t, perturbed[i], 
                    linestyle='--', 
                    alpha=0.8,
                    label=f'Perturbación {j+1}' if i == 0 else "")
            
            # Marcar evento perturbador
            if i == pert[2]:
                plt.axvline(x=pert[0], color='gray', linestyle=':', alpha=0.5)
        
        plt.title(f'Resiliencia a Perturbaciones: {variables[i]}')
        plt.xlabel('Tiempo')
        plt.ylabel('Intensidad')
        plt.grid(True, alpha=0.3)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('../figs/karma_perturbations.png', dpi=150)
    
    # 4. Análisis Cuantitativo de Resiliencia
    analyze_resilience(solution_det, perturbations, t)
    
    plt.show()

def three_roots_model(y, t, params):
    """Modelo determinista de tres raíces (para análisis de perturbaciones)"""
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

def analyze_resilience(original, perturbations, t):
    """Calcula métricas de resiliencia después de perturbaciones"""
    print("\nANÁLISIS DE RESILIENCIA")
    recovery_times = []
    overshoots = []
    
    for i, pert in enumerate(perturbations):
        # Aplicar perturbación
        perturbed = apply_perturbation(original, t, pert)
        var_idx = pert[2]
        
        # Encontrar tiempo de recuperación
        recovery_threshold = 0.1 * abs(pert[1])  # 10% de la perturbación
        diff = np.abs(perturbed[var_idx] - original[var_idx])
        recovery_idx = np.where(diff < recovery_threshold)[0]
        recovery_idx = recovery_idx[recovery_idx > np.where(t >= pert[0])[0][0]]
        
        if len(recovery_idx) > 0:
            recovery_time = t[recovery_idx[0]] - pert[0]
            recovery_times.append(recovery_time)
        else:
            recovery_time = np.inf
        
        # Calcular sobreimpulso máximo
        max_deviation = np.max(np.abs(perturbed[var_idx] - original[var_idx]))
        overshoot = max_deviation / abs(pert[1]) - 1  # Porcentaje sobre la perturbación
        overshoots.append(max(0, overshoot))
        
        print(f"Perturbación {i+1} ({'+' if pert[1] > 0 else ''}{pert[1]} a {['I','A','V'][var_idx]} en t={pert[0]}):")
        print(f"  Tiempo de recuperación: {recovery_time:.2f} unidades")
        print(f"  Sobreimpulso máximo: {overshoot*100:.1f}%")
    
    # Visualización de resultados
    plt.figure(figsize=(10, 6))
    
    # Tiempos de recuperación
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(recovery_times)+1), recovery_times, color='teal')
    plt.title('Tiempo de Recuperación')
    plt.xlabel('Perturbación')
    plt.ylabel('Unidades de tiempo')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Sobreimpulsos
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(overshoots)+1), overshoots, color='purple')
    plt.title('Sobreimpulso Máximo')
    plt.xlabel('Perturbación')
    plt.ylabel('Porcentaje sobre perturbación')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Métricas de Resiliencia Kármica')
    plt.tight_layout()
    plt.savefig('../figs/karma_resilience_metrics.png', dpi=150)

if __name__ == "__main__":
    main()
