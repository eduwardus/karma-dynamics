# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 19:59:37 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eigvals

# ===== CONFIGURATION =====
lambda_w_vals = np.linspace(0, 50, 80)
betaVE_vals   = np.linspace(0, 10, 80)


# Frozen constants
V0, I0, P0, E0 = 0.05, 0.10, 0.05, 0.05

# Model constants
kappa, gamma_m, eps = 0.10, 0.03, 0.01
eta_M               = 0.20
alpha_A, beta_AP, gamma_A = 0.20, 0.10, 0.05
alpha_V, gamma_V, theta   = 0.15, 0.03, 0.02
alpha_I, beta_IP, gamma_I = 0.15, 0.10, 0.05
lambda_Psi                = 0.05

JAC_EPS    = 1e-6
FSOLVE_TOL = 1e-12
FSOLVE_MAXF= 2000
# =========================

def F6(x, lambda_w, beta_VE):
    m, w, A, V, I, Psi = x
    dm   = kappa*A - gamma_m*m*(1+np.tanh(w)) - eps*V*m
    dw   = eta_M - 0.01*w - lambda_w*I*w
    dA   = alpha_A*A + beta_AP*A*np.tanh(m) - gamma_A*w*A
    S    = 1/(1+np.exp(-0.5*(m-5)))
    dV   = alpha_V*V + beta_VE*V*E0 - gamma_V*w*V + theta*S
    dI   = alpha_I*I + beta_IP*np.tanh(I*P0) - gamma_I*w*I
    dPsi = -lambda_Psi*Psi + V
    return np.array([dm, dw, dA, dV, dI, dPsi])

def jac6(x_star, lambda_w, beta_VE):
    n  = len(x_star)
    J  = np.zeros((n,n))
    f0 = F6(x_star, lambda_w, beta_VE)
    for i in range(n):
        xp     = x_star.copy()
        xp[i] += JAC_EPS
        f1     = F6(xp, lambda_w, beta_VE)
        J[:,i] = (f1 - f0)/JAC_EPS
    return J

# allocate
max_re = np.full((len(lambda_w_vals),len(betaVE_vals)), np.nan)
x_prev = np.array([0.1,0.2,0.1,0.05,0.1,0.0])

print("2D scan starting...")
for i, lw in enumerate(lambda_w_vals):
    for j, bVE in enumerate(betaVE_vals):
        # find fixed point
        def F_wrap(x): return F6(x, lw, bVE)
        sol = None
        for guess in (x_prev, np.ones(6)*0.1):
            try:
                s, info, ier, _ = fsolve(
                    F_wrap, guess,
                    full_output=True, xtol=FSOLVE_TOL, maxfev=FSOLVE_MAXF
                )
                if ier==1:
                    sol    = s
                    x_prev = s
                    break
            except:
                pass
        if sol is None:
            continue

        ev            = eigvals(jac6(sol, lw, bVE))
        max_re[i,j]   = np.max(np.real(ev))
    if i%10==0:
        print(f"Completed row {i+1}/{len(lambda_w_vals)} (λ_w={lw:.1f})")
print("Scan finished.")

# Build sign matrix and detect sign changes
S = np.sign(max_re)
candidates = []
rows, cols = S.shape
for i in range(rows):
    for j in range(cols):
        if j+1 < cols and not np.isnan(S[i,j])*np.isnan(S[i,j+1]):
            if S[i,j]*S[i,j+1] < 0:
                candidates.append((lambda_w_vals[i], betaVE_vals[j]))
        if i+1 < rows and not np.isnan(S[i,j])*np.isnan(S[i+1,j]):
            if S[i,j]*S[i+1,j] < 0:
                candidates.append((lambda_w_vals[i], betaVE_vals[j]))

# Unique
candidates = sorted(set(candidates))

# Plot heatmap
plt.figure(figsize=(6,5))
plt.imshow(max_re, origin='lower',
           extent=(betaVE_vals[0],betaVE_vals[-1],
                   lambda_w_vals[0],lambda_w_vals[-1]),
           aspect='auto', cmap='RdBu_r')
plt.colorbar(label='max Re(λ)')

# Overlay candidates
if candidates:
    lw_c, bve_c = zip(*candidates)
    plt.scatter(bve_c, lw_c, c='yellow', edgecolors='black',
                s=50, label='Zero‐cross candidates')
    plt.legend()

plt.xlabel(r'$\beta_{VE}$')
plt.ylabel(r'$\lambda_w$')
plt.title('Stability zero‐crossings')
plt.tight_layout()
plt.show()

# Print them
print("Candidates (λ_w, β_VE) where max_re changes sign:")
for lw, bVE in candidates:
    print(f"  λ_w = {lw:.2f}, β_VE = {bVE:.2f}")
