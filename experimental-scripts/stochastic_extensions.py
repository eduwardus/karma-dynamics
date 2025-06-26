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
from scipy.integrate import solve_ivp
from sdeint import itoint
import ddeint
import os

# Create directory for figures if it doesn't exist
if not os.path.exists('../figs'):
    os.makedirs('../figs')

# =============================================
# 1. Stochastic Model (SDE) for Individual Karma
# =============================================
def stochastic_seirs(y, t, params):
    S, E, I, R = y
    # Using abs() for stability in denominators
    denom = 1 + np.abs(R)
    dS = params['xi']*(1-params['w'])*R - params['alpha']*S + params['lambda_val']*(R/denom)
    dE = params['alpha']*S - params['sigma']*E
    dI = params['sigma']*E - params['gamma']*I
    dR = params['gamma']*I - params['xi']*(1-params['w'])*R - params['lambda_val']*(R/denom)
    return np.array([dS, dE, dI, dR])

def noise_function(y, t, params):
    S, E, I, R = y
    noise_S = params['sigma_S'] * S
    noise_E = params['sigma_E'] * E
    noise_I = params['sigma_I'] * I
    noise_R = params['sigma_R'] * R
    return np.diag([noise_S, noise_E, noise_I, noise_R])

def simulate_stochastic_seirs(params, y0, t):
    t_array = np.array(t)
    f = lambda y, t: stochastic_seirs(y, t, params)
    G = lambda y, t: noise_function(y, t, params)
    result = itoint(f, G, y0, t_array)
    return result.T

# =============================================
# 2. Time-Delayed Model (DDE) with Clipping
# =============================================
def delayed_three_roots(Y, t, params):
    I, A, V = Y(t)
    tau = params['tau']
    I_tau, A_tau, V_tau = Y(t - tau)
    
    # Clip values to prevent overflow
    I = np.clip(I, 0, 10)
    A = np.clip(A, 0, 10)
    V = np.clip(V, 0, 10)
    I_tau = np.clip(I_tau, 0, 10)
    A_tau = np.clip(A_tau, 0, 10)
    V_tau = np.clip(V_tau, 0, 10)
    
    dIdt = (params['alpha_I'] * I + 
            params['beta_IA'] * A_tau * V_tau - 
            params['gamma_I'] * params['w'] * I)
    
    dAdt = (params['alpha_A'] * A + 
            params['beta_AV'] * V_tau * I_tau - 
            params['gamma_A'] * params['w'] * A)
    
    dVdt = (params['alpha_V'] * V + 
            params['beta_VI'] * I_tau * A_tau - 
            params['gamma_V'] * params['w'] * V)
    
    # Clip derivatives to prevent instability
    dIdt = np.clip(dIdt, -10, 10)
    dAdt = np.clip(dAdt, -10, 10)
    dVdt = np.clip(dVdt, -10, 10)
    
    return [dIdt, dAdt, dVdt]

# =============================================
# 3. Deterministic Three Roots Model with Clipping
# =============================================
def three_roots_model(t, y, params):
    I, A, V = y
    # Clip values to prevent overflow
    I = np.clip(I, 0, 10)
    A = np.clip(A, 0, 10)
    V = np.clip(V, 0, 10)
    
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    
    # Clip derivatives to prevent instability
    dIdt = np.clip(dIdt, -10, 10)
    dAdt = np.clip(dAdt, -10, 10)
    dVdt = np.clip(dVdt, -10, 10)
    
    return [dIdt, dAdt, dVdt]

# =============================================
# 4. Perturbation Analysis
# =============================================
def apply_perturbation(solution, t, perturbation):
    perturbed = solution.copy()
    event_time, intensity, affected_var = perturbation
    idx = np.abs(t - event_time).argmin()
    
    if affected_var < perturbed.shape[0]:
        perturbed[affected_var, idx:] += intensity
    
    return perturbed

# =============================================
# 5. Resilience Analysis
# =============================================
def analyze_resilience(original, perturbations, t):
    print("\nRESILIENCE ANALYSIS")
    recovery_times = []
    overshoots = []
    
    for i, pert in enumerate(perturbations):
        perturbed = apply_perturbation(original, t, pert)
        var_idx = pert[2]
        recovery_threshold = 0.1 * abs(pert[1])
        diff = np.abs(perturbed[var_idx] - original[var_idx])
        
        # Find indices after perturbation event
        post_event_idx = np.where(t >= pert[0])[0]
        if len(post_event_idx) == 0:
            recovery_time = np.inf
            max_deviation = 0
        else:
            # First index where difference falls below threshold
            recovery_candidates = np.where(diff[post_event_idx] < recovery_threshold)[0]
            if len(recovery_candidates) > 0:
                recovery_time = t[post_event_idx[recovery_candidates[0]]] - pert[0]
            else:
                recovery_time = np.inf
            
            # Calculate maximum deviation
            max_deviation = np.max(np.abs(perturbed[var_idx, post_event_idx] - original[var_idx, post_event_idx]))
        
        overshoot = max_deviation / abs(pert[1]) - 1
        overshoots.append(max(0, overshoot))
        recovery_times.append(recovery_time)
        
        print(f"Perturbation {i+1} ({'+' if pert[1] > 0 else ''}{pert[1]} to {['I','A','V'][var_idx]} at t={pert[0]}):")
        print(f"  Recovery time: {recovery_time:.2f} units")
        print(f"  Maximum overshoot: {overshoot*100:.1f}%")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(recovery_times)+1), recovery_times, color='teal')
    plt.title('Recovery Time')
    plt.xlabel('Perturbation')
    plt.ylabel('Time units')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(overshoots)+1), overshoots, color='purple')
    plt.title('Maximum Overshoot')
    plt.xlabel('Perturbation')
    plt.ylabel('Percentage over perturbation')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Karmic Resilience Metrics')
    plt.tight_layout()
    plt.savefig('../figs/karma_resilience_metrics.png', dpi=150)

# =============================================
# Main Function
# =============================================
def main():
    print("\n" + "="*70)
    print("STOCHASTIC EXTENSIONS OF KARMIC MODELS")
    print("="*70)
    
    # Common parameters with reduced time span
    t = np.linspace(0, 100, 3000)  # Reduced from 50 to 30
    t_dde = np.linspace(0, 30, 2000)  # Reduced from 100 to 30
    
    # 1. Stochastic SEIRS Simulation
    print("Running: 1. Stochastic SEIRS")
    seirs_params = {
        'alpha': 0.3, 'sigma': 0.5, 'gamma': 0.4, 'xi': 0.2,
        'w': 0.3, 'lambda_val': 0.1,
        'sigma_S': 0.05, 'sigma_E': 0.07, 'sigma_I': 0.1, 'sigma_R': 0.08
    }
    y0_seirs = [0.7, 0.2, 0.05, 0.05]
    
    num_trajectories = 5
    plt.figure(figsize=(12, 8))
    
    for i in range(num_trajectories):
        S, E, I, R = simulate_stochastic_seirs(seirs_params, y0_seirs, t)
        plt.plot(t, I, alpha=0.6, label=f'Trajectory {i+1}' if i < 3 else "")
    
    # Deterministic solution with relaxed tolerances
    sol = solve_ivp(
        lambda t, y: stochastic_seirs(y, t, seirs_params),
        [t[0], t[-1]],
        y0_seirs,
        method='BDF',  # Better for stiff systems
        t_eval=t,
        atol=1e-4,
        rtol=1e-3
    )
    
    # Handle solution
    if not sol.success:
        print(f"Integration warning: {sol.message}")
        # Use successful portion
        valid_idx = np.where(sol.t <= t[-1])[0]
        det_solution = sol.y[:, valid_idx]
        t_valid = sol.t[valid_idx]
    else:
        det_solution = sol.y
        t_valid = t
    
    plt.plot(t_valid, det_solution[2], 'k--', linewidth=2, label='Deterministic')
    
    plt.title('Karmic Manifestation (I) with Stochastic Noise')
    plt.xlabel('Time')
    plt.ylabel('Manifested Karma (I)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/stochastic_seirs.png', dpi=150)
    
    # 2. Time-Delayed Simulation with reduced parameters
    print("Running: 2. Delayed Three Roots")
    delay_params = {
        'alpha_I': 0.1, 'alpha_A': 0.08, 'alpha_V': 0.12,  # Reduced growth rates
        'beta_IA': 0.06, 'beta_AV': 0.07, 'beta_VI': 0.05,  # Reduced interaction terms
        'gamma_I': 0.15, 'gamma_A': 0.12, 'gamma_V': 0.18,  # Increased decay rates
        'w': 0.3,  # Increased weight
        'tau': 5.0  # Reduced delay
    }
    
    # Initial history
    def history(t):
        return [0.5, 0.3, 0.2]
    
    # Wrapper function for DDE
    def delayed_wrapper(Y, t):
        return delayed_three_roots(Y, t, delay_params)
    
    # Solve DDE
    solution_dde = ddeint.ddeint(delayed_wrapper, history, t_dde)
    I_dde, A_dde, V_dde = solution_dde.T
    
    # Filter out NaN values
    valid_mask = ~np.isnan(I_dde) & ~np.isnan(A_dde) & ~np.isnan(V_dde)
    I_dde = I_dde[valid_mask]
    A_dde = A_dde[valid_mask]
    V_dde = V_dde[valid_mask]
    t_dde_valid = t_dde[valid_mask]
    
    plt.figure(figsize=(12, 8))
    plt.plot(t_dde_valid, I_dde, 'b-', label='Ignorance (I)')
    plt.plot(t_dde_valid, A_dde, 'g-', label='Attachment (A)')
    plt.plot(t_dde_valid, V_dde, 'r-', label='Aversion (V)')
    
    plt.axvline(x=delay_params['tau'], color='gray', linestyle='--', alpha=0.7)
    plt.text(delay_params['tau']+0.5, 0.9*max(V_dde), 'First effects\nof delay', fontsize=10)
    
    plt.title('Three Karmic Roots with Maturation Delay (τ=5)')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/delayed_karma.png', dpi=150)
    
    # 3. Perturbation Analysis with reduced parameters
    print("Running: 3. Perturbation Analysis")
    three_roots_params = {
        'alpha_I': 0.25, 'alpha_A': 0.22, 'alpha_V': 0.28,  # Reduced growth rates
        'beta_IA': 0.45, 'beta_AV': 0.5, 'beta_VI': 0.4,  # Reduced interaction terms
        'gamma_I': 0.15, 'gamma_A': 0.12, 'gamma_V': 0.18,  # Increased decay rates
        'w': 0.35  # Increased weight
    }
    

    y0_roots = [0.4, 0.3, 0.3]
    
    # Deterministic solution with BDF method
    sol_roots = solve_ivp(
        lambda t, y: three_roots_model(t, y, three_roots_params),
        [t[0], t[-1]],
        y0_roots,
        method='BDF',
        t_eval=t,
        atol=1e-4,
        rtol=1e-3
    )
    
    # Handle solution
    if not sol_roots.success:
        print(f"Integration warning: {sol_roots.message}")
        # Use successful portion
        valid_idx = np.where(sol_roots.t <= t[-1])[0]
        solution_det = sol_roots.y[:, valid_idx]
        t_valid_roots = sol_roots.t[valid_idx]
    else:
        solution_det = sol_roots.y
        t_valid_roots = t
    
    # Filter out NaN values
    valid_mask = ~np.isnan(solution_det[0]) & ~np.isnan(solution_det[1]) & ~np.isnan(solution_det[2])
    solution_det = solution_det[:, valid_mask]
    t_valid_roots = t_valid_roots[valid_mask]
    
    # Adjusted perturbation times
    perturbations = [
        (20, 0.3, 0),    # Perturbation to Ignorance
        (40, -0.25, 1),  # Negative perturbation to Attachment
        (60, 0.35, 2)    # Perturbation to Aversion
    ]
    
    plt.figure(figsize=(14, 10))
    variables = ['Ignorance (I)', 'Attachment (A)', 'Aversion (V)']
    colors = ['blue', 'green', 'red']
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_valid_roots, solution_det[i], color=colors[i], linestyle='-', label='Original')
        
        for j, pert in enumerate(perturbations):
            perturbed = apply_perturbation(solution_det, t_valid_roots, pert)
            plt.plot(t_valid_roots, perturbed[i], linestyle='--', alpha=0.8,
                     label=f'Perturbation {j+1}' if i == 0 else "")
            
            if i == pert[2]:
                plt.axvline(x=pert[0], color='gray', linestyle=':', alpha=0.5)
        
        plt.title(f'Resilience to Perturbations: {variables[i]}')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('../figs/karma_perturbations.png', dpi=150)
    
    # 4. Quantitative Resilience Analysis
    analyze_resilience(solution_det, perturbations, t_valid_roots)
    
    plt.show()
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
    
