# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:18:15 2025

@author: eggra
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eigvals

# ----------------- CONFIGURATION -----------------
# Parameter grid for (eta_M, beta_AP)
#eta_vals  = np.linspace(0.0, 5.0, 60)
beta_vals = np.linspace(0.0, 2.0, 60)
# Solo eta_M, con beta_AP=1.0
eta_vals = np.linspace(0, 10, 100)
beta_AP  = 1.0




# Model constants (frozen for this subsystem)
V0, I0        = 0.05, 0.10
kappa         = 0.10
gamma_m, eps  = 0.1, 0.1
alpha_A       = 0.20
gamma_A       = 0.05
mu, lambda_w  = 0.01, 0.01

# Numerical Jacobian increment
JAC_EPS = 1e-6

# Tolerance and max iters for fsolve
FSOLVE_TOL  = 1e-12
FSOLVE_MAXF = 2000
# --------------------------------------------------

def F3(x, eta_M, beta_AP):
    """Right–hand side of the (m, w, A) subsystem."""
    m, w, A = x
    dm = kappa*A - gamma_m*m*(1 + np.tanh(w)) - eps*V0*m
    dw = eta_M - mu*w - lambda_w*I0*w
    dA = alpha_A*A + beta_AP*A*np.tanh(m) - gamma_A*w*A
    return np.array([dm, dw, dA])

def numeric_jacobian(x_star, eta_M, beta_AP):
    """Finite‐difference Jacobian of F3 at x_star."""
    n = len(x_star)
    J = np.zeros((n,n))
    f0 = F3(x_star, eta_M, beta_AP)
    for i in range(n):
        xp = x_star.copy()
        xp[i] += JAC_EPS
        f1 = F3(xp, eta_M, beta_AP)
        J[:, i] = (f1 - f0) / JAC_EPS
    return J

# Storage for detected bifurcations
hopf_points        = []
saddlenode_points  = []

# Warm‐start initial guess
x_prev = np.array([0.1, 0.2, 0.1])

total = len(eta_vals)*len(beta_vals)
count = 0

print("Starting bifurcation detection sweep over {} points...".format(total))

for i, eta in enumerate(eta_vals):
    for j, beta in enumerate(beta_vals):
        count += 1
        # wrapper for fsolve
        def F_wrap(x): return F3(x, eta, beta)

        # try to find fixed point via warm‐start → default → fallback
        sol = None
        for guess in (x_prev, [0.1,0.2,0.1], [0.5,0.5,0.5]):
            try:
                sol_cand, info, ier, mesg = fsolve(
                    F_wrap, guess, full_output=True,
                    xtol=FSOLVE_TOL, maxfev=FSOLVE_MAXF
                )
                if ier == 1:  # converged
                    sol = sol_cand
                    x_prev = sol_cand
                    break
            except Exception:
                continue

        # if we failed to converge, skip
        if sol is None:
            print(f"[{count}/{total}] η={eta:.3f}, β={beta:.3f}: no converg.")
            continue

        # compute Jacobian & eigenvalues
        J = numeric_jacobian(sol, eta, beta)
        ev = eigvals(J)
        re, im = np.real(ev), np.imag(ev)

        # detect saddle-node: any real eigenvalue near zero crossing
        for rv in re:
            if abs(rv) < 1e-3:
                saddlenode_points.append((eta, beta))
                break

        # detect Hopf: complex‐conjugate pair with small real part change sign
        # criterion: two eigenvalues with |Re|<small and |Im|>small
        complex_pairs = [(re[k], im[k]) for k in range(len(ev)) if abs(im[k])>1e-3]
        if len(complex_pairs) >= 2:
            # check if any pair has small real parts
            if any(abs(rv)<1e-2 for rv, _ in complex_pairs):
                hopf_points.append((eta, beta))

        if count % 100 == 0 or sol is None:
            print(f"[{count}/{total}] η={eta:.3f}, β={beta:.3f} → Re(eigen)={re}")

print("Sweep complete.")
print(f"Detected {len(saddlenode_points)} saddle‐node candidates.")
print(f"Detected {len(hopf_points)} Hopf candidates.")

# ------------------- PLOTTING -------------------
plt.figure(figsize=(6,4))
if saddlenode_points:
    es, bs = zip(*saddlenode_points)
    plt.scatter(es, bs, c='blue', s=10, label='Saddle‐node')
if hopf_points:
    es, bs = zip(*hopf_points)
    plt.scatter(es, bs, c='red',  s=10, label='Hopf')
plt.xlabel(r'$\eta_M$')
plt.ylabel(r'$\beta_{AP}$')
plt.title('Bifurcation loci in parameter space')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
