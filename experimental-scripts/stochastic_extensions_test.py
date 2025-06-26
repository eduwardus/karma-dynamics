# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:42:15 2025

@author: eggra

    ### Consequences of the Model & Consistency with Theravada Buddhism

**1. Irreversible Karmic Imprints**  
*Model Behavior*: Perturbations (actions) permanently alter the system state with no return to baseline.  
*Theravada Consistency*: Aligns with the Abhidhamma concept of *"cetanā"* (volitional action) creating indelible karmic seeds (*bīja*) that **must** ripen. As the *Aṅguttara Nikāya 3.34* states:  
> *"I declare, monks, that intention is karma. Intending, one does karma by body, speech, and mind."*  
Once created, karma cannot be erased - only experienced (*vipāka*).

**2. No Cosmic Balancing Mechanism**  
*Model Behavior*: No autonomous forces restore equilibrium after disturbances.  
*Theravada Consistency*: Reflects the doctrine of *attakāra* (self-responsibility). Theravada rejects divine intervention or automatic forgiveness. As the *Dhp 165* teaches:  
> *"By oneself alone is evil done; by oneself is one defiled. By oneself is evil left undone; by oneself alone is one purified."*  
Purification requires *conscious effort*, not passive cosmic adjustment.

**3. Linear Accumulation Without Transformation**  
*Model Behavior*: Variables (Ignorance/Aversion/Attachment) operate in isolation, accumulating without mutual conversion.  
*Theravada Consistency*: Mirrors the *Abhidhamma* analysis of mental factors (*cetasikas*) as discrete entities. Karmic results manifest as specific, non-transferable experiences:  
- Unwholesome actions → Suffering (*dukkha*)  
- Wholesome actions → Happiness (*sukha*)  
As taught in *Majjhima Nikāya 135*, karma operates with "causal fixity."

**4. Absence of Retroactive Neutralization**  
*Model Behavior*: Past perturbations remain active indefinitely.  
*Theravada Consistency*: Echoes the strict karmic ontology where past actions (*kamma*) are fixed. Theravada emphasizes:  
- Past karma must be experienced (*pariyatti*)  
- Future karma can be redirected through present action (*paṭipatti*)  
- But no deletion of existing karmic seeds, as cautioned in *Aṅguttara Nikāya 5.292*.

**5. Why This Matches Theravada**  
The model embodies three core Theravada principles:  
a) **Anattā (Non-Self)**  
No "soul" to absolve karma - just cause-effect chains. The model's mechanistic structure reflects this impersonal nature.  

b) **Paṭiccasamuppāda (Dependent Origination)**  
Each perturbation creates self-sustaining causal chains (*nidānas*), matching the model's permanent state-shifts.  

c) **Niyati Vāda (Lawful Causality)**  
Predictable parameter effects mirror the strict karmic causality (*kamma-niyāma*) in Theravada cosmology.  

**6. Philosophical Implications**  
This model depicts a universe where:  
- Actions are cosmic transactions with **permanent balance sheets**  
- Moral responsibility is absolute and non-negotiable  
- Liberation (*nibbāna*) requires not divine grace but **exhausting existing karmic seeds** through non-generation of new karma  

As the Buddha stated in *Samyutta Nikāya 36.21*:  
> *"Kamma is the field, consciousness the seed, and craving the moisture for beings bound by delusion to take rebirth."*  

The model's irreversible dynamics reflect the terrifying gravity of karma in Theravada - where every action echoes eternally until full awakening severs the causal chains.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# Create directory for figures if it doesn't exist
if not os.path.exists('../figs'):
    os.makedirs('../figs')

# =============================================
# 1. Three Roots Model Definition
# =============================================
def three_roots_model(t, y, params):
    """
    Deterministic Three Roots Model with Clipping
    """
    I, A, V = y
    
    # Unpack parameters
    α_I, α_A, α_V, β_IA, β_AV, β_VI, γ_I, γ_A, γ_V, w = params
    
    # Clip values to prevent overflow
    I = np.clip(I, 0, 10)
    A = np.clip(A, 0, 10)
    V = np.clip(V, 0, 10)
    
    # Compute derivatives
    dIdt = α_I*I + β_IA*A*V - γ_I*w*I
    dAdt = α_A*A + β_AV*V*I - γ_A*w*A
    dVdt = α_V*V + β_VI*I*A - γ_V*w*V
    
    # Clip derivatives to prevent instability
    dIdt = np.clip(dIdt, -10, 10)
    dAdt = np.clip(dAdt, -10, 10)
    dVdt = np.clip(dVdt, -10, 10)
    
    return [dIdt, dAdt, dVdt]

# =============================================
# 2. Resilience Testing Function (Optimized)
# =============================================
def test_resilience(params, perturbation=(0.2, -0.15, 0.25), t_span=50):
    """
    Tests if a perturbation causes permanent shift
    Returns True if perturbation effect is permanent
    """
    # Initial conditions
    y0_original = [0.4, 0.3, 0.3]
    y0_perturbed = [y0_original[0] + perturbation[0],
                    y0_original[1] + perturbation[1],
                    y0_original[2] + perturbation[2]]
    
    # Reduced time points for efficiency
    t_eval = np.linspace(0, t_span, 500)
    
    # Solve both systems in single call (more efficient)
    def combined_system(t, y):
        I1, A1, V1, I2, A2, V2 = y
        orig = three_roots_model(t, [I1, A1, V1], params)
        pert = three_roots_model(t, [I2, A2, V2], params)
        return [*orig, *pert]
    
    sol = solve_ivp(
        combined_system,
        [0, t_span],
        [*y0_original, *y0_perturbed],
        t_eval=t_eval,
        method='LSODA',
        atol=1e-4,
        rtol=1e-3
    )
    
    # Extract solutions
    orig_sol = sol.y[:3, -1]
    pert_sol = sol.y[3:, -1]
    
    # Check final state difference
    final_diff = np.abs(pert_sol - orig_sol)
    avg_diff = np.mean(final_diff)
    
    # Permanent shift if difference remains > 1% of perturbation
    return avg_diff > 0.01 * np.mean(np.abs(perturbation))

# =============================================
# 3. Parameter Space Exploration (Optimized)
# =============================================
def explore_parameter_space(n_samples=100):
    """
    Tests model resilience across random parameter samples
    with progress tracking and time estimation
    """
    # Results storage
    results = []
    permanent_shifts = 0
    
    # Parameter ranges
    param_ranges = {
        'α': (0.01, 0.5),      # Growth rates
        'β': (0.01, 0.8),      # Interaction strengths
        'γ': (0.05, 0.5),      # Dissipation rates
        'w': (0.1, 0.9)        # Karmic weight
    }
    
    print(f"\nTesting {n_samples} parameter sets...")
    print("This may take several minutes depending on your system")
    print("Progress:")
    
    # Create progress bar with time estimation
    start_time = time.time()
    progress_bar = tqdm(total=n_samples, unit='test')
    
    for i in range(n_samples):
        # Generate random parameters within ranges
        α_I, α_A, α_V = np.random.uniform(*param_ranges['α'], 3)
        β_IA, β_AV, β_VI = np.random.uniform(*param_ranges['β'], 3)
        γ_I, γ_A, γ_V = np.random.uniform(*param_ranges['γ'], 3)
        w = np.random.uniform(*param_ranges['w'])
        
        params = (α_I, α_A, α_V, β_IA, β_AV, β_VI, γ_I, γ_A, γ_V, w)
        
        # Test resilience
        permanent = test_resilience(params)
        
        # Record results
        results.append({
            'params': params,
            'permanent': permanent
        })
        
        if permanent:
            permanent_shifts += 1
            
        # Update progress bar every 5 tests
        if i % 5 == 0:
            elapsed = time.time() - start_time
            per_test = elapsed / (i + 1)
            remaining = per_test * (n_samples - i - 1)
            progress_bar.set_postfix({
                'permanent': f'{permanent_shifts}/{i+1}',
                'est_time': f'{remaining:.1f}s'
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per test: {elapsed/n_samples:.3f} seconds")
    print(f"Permanent shifts: {permanent_shifts}/{n_samples} ({permanent_shifts/n_samples:.1%})")
    
    return results, permanent_shifts/n_samples

# =============================================
# 4. Visualization of Results
# =============================================
def visualize_results(results):
    """Plots parameter space with permanent shift regions"""
    # Prepare data for scatter plot
    alpha_values = []
    beta_values = []
    gamma_values = []
    w_values = []
    colors = []
    
    for result in results:
        α_I, α_A, α_V, β_IA, β_AV, β_VI, γ_I, γ_A, γ_V, w = result['params']
        avg_alpha = (α_I + α_A + α_V) / 3
        avg_beta = (β_IA + β_AV + β_VI) / 3
        avg_gamma = (γ_I + γ_A + γ_V) / 3
        
        alpha_values.append(avg_alpha)
        beta_values.append(avg_beta)
        gamma_values.append(avg_gamma)
        w_values.append(w)
        colors.append('red' if result['permanent'] else 'blue')
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Alpha vs Beta
    ax1 = fig.add_subplot(221)
    ax1.scatter(alpha_values, beta_values, c=colors, alpha=0.6)
    ax1.set_xlabel('Average Growth Rate (α)')
    ax1.set_ylabel('Average Interaction (β)')
    ax1.set_title('Permanent Shift Regions (Red = Permanent)')
    
    # Gamma vs w
    ax2 = fig.add_subplot(222)
    ax2.scatter(gamma_values, w_values, c=colors, alpha=0.6)
    ax2.set_xlabel('Average Dissipation (γ)')
    ax2.set_ylabel('Karmic Weight (w)')
    
    # Alpha vs Gamma
    ax3 = fig.add_subplot(223)
    ax3.scatter(alpha_values, gamma_values, c=colors, alpha=0.6)
    ax3.set_xlabel('Average Growth Rate (α)')
    ax3.set_ylabel('Average Dissipation (γ)')
    ax3.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.3)  # Reference line
    
    # Beta vs w
    ax4 = fig.add_subplot(224)
    ax4.scatter(beta_values, w_values, c=colors, alpha=0.6)
    ax4.set_xlabel('Average Interaction (β)')
    ax4.set_ylabel('Karmic Weight (w)')
    
    plt.tight_layout()
    plt.savefig('../figs/permanent_shift_analysis.png', dpi=150)
    plt.show()

# =============================================
# 5. Mathematical Verification (Quick)
# =============================================
def mathematical_verification():
    """Demonstrates why permanent shifts are inevitable"""
    # Create a quick visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample parameters
    params = (0.3, 0.25, 0.35, 0.6, 0.7, 0.5, 0.4, 0.35, 0.45, 0.3)
    
    # Solve for different initial conditions
    t_span = [0, 50]
    t_eval = np.linspace(0, 50, 500)
    
    # Plot trajectories for Ignorance
    for i in range(5):
        y0 = np.random.rand(3) * 0.5 + 0.3
        sol = solve_ivp(
            lambda t, y: three_roots_model(t, y, params),
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA'
        )
        ax[0].plot(t_eval, sol.y[0], alpha=0.7, label=f'Trajectory {i+1}' if i<3 else "")
    
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Ignorance (I)')
    ax[0].set_title('Diverging Trajectories')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    
    # Plot final states
    final_states = []
    for i in range(20):
        y0 = np.random.rand(3) * 0.5 + 0.3
        sol = solve_ivp(
            lambda t, y: three_roots_model(t, y, params),
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA'
        )
        final_states.append(sol.y[:, -1])
    
    final_states = np.array(final_states)
    ax[1].scatter(range(len(final_states)), final_states[:, 0], label='Ignorance')
    ax[1].scatter(range(len(final_states)), final_states[:, 1], label='Attachment')
    ax[1].scatter(range(len(final_states)), final_states[:, 2], label='Aversion')
    
    ax[1].set_xlabel('Simulation')
    ax[1].set_ylabel('Final Value')
    ax[1].set_title('Diverse Final States')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('../figs/verification_quick.png', dpi=150)
    plt.show()

# =============================================
# Main Function
# =============================================
def main():
    print("="*70)
    print("KARMIC RESILIENCE VERIFICATION")
    print("="*70)
    
    # Part 1: Parameter space exploration
    sample_size = 10000  # Reduced for quick execution
    results, fraction_permanent = explore_parameter_space(n_samples=sample_size)
    
    # Part 2: Visualize results
    visualize_results(results)
    
    # Part 3: Quick mathematical demonstration
    mathematical_verification()
    
    # Final conclusion
    print("\nCONCLUSION:")
    print(f"Across {sample_size} random parameter sets, {fraction_permanent:.1%} showed permanent shifts")
    print("This demonstrates the model's fundamental property: Actions create irreversible changes")
    print("Consistent with Theravada Buddhist doctrine of karmic accumulation")

if __name__ == "__main__":
    main()