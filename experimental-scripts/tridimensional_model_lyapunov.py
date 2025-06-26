# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:09:33 2025

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Exploración Monte Carlo ampliada para detectar regiones caóticas
en el modelo karmic_vector original, con aleatorización también
de los parámetros α y condiciones iniciales, usando exponente de Lyapunov calculado por trozos,
y guardando todos los parámetros en el CSV de salida.

Created on Thu Jun 12 2025
@author: eggra
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

def karmic_vector(t, y,
                  alpha_I, alpha_A, alpha_V,
                  beta_IA, beta_AV, beta_VI,
                  gamma, w):
    """
    Ecuaciones originales:
      dI/dt = α_I·I + β_IA·(A·V) − (γ·w)·I
      dA/dt = α_A·A + β_AV·(V·I) − (γ·w)·A
      dV/dt = α_V·V + β_VI·(I·A) − (γ·w)·V
    """
    I, A, V = y
    mu = gamma * w
    dIdt = alpha_I * I + beta_IA * (A * V) - mu * I
    dAdt = alpha_A * A + beta_AV * (V * I) - mu * A
    dVdt = alpha_V * V + beta_VI * (I * A) - mu * V
    return np.array([dIdt, dAdt, dVdt])

def extended_system(t, Y, fun, args, n):
    """
    Sistema extendido para la parte variacional:
    Y = [y (n,), V (n×n).ravel()]
    """
    y = Y[:n]
    V = Y[n:].reshape((n, n))
    f0 = fun(t, y, *args)
    eps = 1e-6
    J = np.zeros((n, n))
    for j in range(n):
        dy = np.zeros(n); dy[j] = eps
        f1 = fun(t, y + dy, *args)
        J[:, j] = (f1 - f0) / eps
    dV = J.dot(V)
    return np.concatenate([f0, dV.ravel()])

def lyapunov_max_piecewise(fun, y0, t0, t1, dt, args, delta0=1e-8):
    """
    Cálculo del exponente de Lyapunov mayor L_max
    por integración en trozos de longitud dt y renormalización.
    """
    n = len(y0)
    V0 = np.eye(n) * delta0
    Y = np.concatenate([y0, V0.ravel()])
    t = t0
    le_sum = 0.0
    N = int(np.ceil((t1 - t0) / dt))

    for _ in range(N):
        t_span = (t, min(t + dt, t1))
        sol = solve_ivp(lambda tt, YY: extended_system(tt, YY, fun, args, n),
                        t_span, Y, method='RK45',
                        rtol=1e-6, atol=1e-9)
        Y_end = sol.y[:, -1]
        y_end = Y_end[:n]
        V_end = Y_end[n:].reshape((n, n))

        Q, R = np.linalg.qr(V_end)
        le_sum += np.log(abs(R[0, 0]))

        Y = np.concatenate([y_end, (Q * delta0).ravel()])
        t = sol.t[-1]

    return le_sum / (t1 - t0)

if __name__ == "__main__":
    # Rangos de muestreo
    N = 500
    beta_min, beta_max = 0.1, 50.0
    mu_min, mu_max     = 1e-6, 1.0   # μ = γ·w
    w_min, w_max       = 0.0, 1.0
    alpha_min, alpha_max = -0.5, 1.0  # rango para α_I, α_A, α_V
    y0_min, y0_max = 0.01, 5.0        # Rango para condiciones iniciales

    # Configuración Lyapunov
    t0, t1, dt = 0.0, 500.0, 2.0

    rng = np.random.default_rng(12345)
    # Generar parámetros aleatorios
    alphas_I = rng.uniform(alpha_min, alpha_max, N)
    alphas_A = rng.uniform(alpha_min, alpha_max, N)
    alphas_V = rng.uniform(alpha_min, alpha_max, N)
    betas_IA = rng.uniform(beta_min, beta_max, N)
    betas_AV = rng.uniform(beta_min, beta_max, N)
    betas_VI = rng.uniform(beta_min, beta_max, N)
    mus      = rng.uniform(mu_min, mu_max, N)
    ws       = rng.uniform(w_min, w_max, N)
    # Generar condiciones iniciales aleatorias
    I0s = rng.uniform(y0_min, y0_max, N)
    A0s = rng.uniform(y0_min, y0_max, N)
    V0s = rng.uniform(y0_min, y0_max, N)

    resultados = []
    total_runs = N
    start_time = time.time()

    print("Iniciando Monte Carlo extendido (incluye α y condiciones iniciales)...")
    print(f"Total runs: {total_runs}\n")

    for idx, (αI, αA, αV, ia, av, vi, mu, w, I0, A0, V0) in enumerate(
            zip(alphas_I, alphas_A, alphas_V,
                betas_IA, betas_AV, betas_VI,
                mus, ws, I0s, A0s, V0s), start=1):

        # Construir condición inicial para esta iteración
        y0 = np.array([I0, A0, V0])
        
        gamma = mu / max(w, 1e-6)
        params = (αI, αA, αV, ia, av, vi, gamma, w)

        L = lyapunov_max_piecewise(
            karmic_vector, y0, t0, t1, dt, params)

        # Guardar resultados incluyendo condiciones iniciales
        resultados.append((αI, αA, αV, ia, av, vi, gamma, w, I0, A0, V0, L))
        if idx % 50 == 0:  # Guardar cada 50 iteraciones
            np.savetxt(f"lyapunov_partial_{idx}.csv", np.column_stack(resultados), header="alpha_I,alpha_A,alpha_V,beta_IA,beta_AV,beta_VI,gamma,w,I0,A0,V0,L_max",delimiter=",")
        # Mostrar progreso y ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_runs - idx)
        eta_min = remaining / 60
        print(f"[{idx}/{total_runs}] L_max = {L: .4f} | ETA ≃ {eta_min:.1f} min")

    total_time = (time.time() - start_time) / 60
    print(f"\nMuestreo completo en {total_time:.1f} minutos.\n")

    # Ordenar y mostrar top 10
    resultados.sort(key=lambda x: x[-1], reverse=True)
    print("Top 10 combinaciones con mayor L_max:")
    print("α_I   α_A   α_V   β_IA  β_AV  β_VI    γ      w     I0     A0     V0    L_max")
    for res in resultados[:10]:
        print(f"{res[0]:5.3f} {res[1]:5.3f} {res[2]:5.3f} {res[3]:5.2f} {res[4]:5.2f} {res[5]:5.2f} "
              f"{res[6]:8.5e} {res[7]:4.2f} {res[8]:5.3f} {res[9]:5.3f} {res[10]:5.3f} {res[11]: .4f}")

    # Guardar todos los datos (ahora con 12 columnas)
    np.savetxt(
        "lyapunov_montecarlo_extended.csv",
        np.column_stack(resultados),
        header="alpha_I,alpha_A,alpha_V,beta_IA,beta_AV,beta_VI,gamma,w,I0,A0,V0,L_max",
        delimiter=","
    )
    print("\nDatos guardados en lyapunov_montecarlo_extended.csv")