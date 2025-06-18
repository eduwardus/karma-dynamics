# scripts/seirs_karma.py
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parámetros del modelo
params = {
    'alpha': 0.3,    # Tasa de activación karma latente
    'sigma': 0.5,    # Tasa de manifestación de intenciones
    'gamma': 0.4,    # Tasa de resolución de acciones
    'xi': 0.2,       # Tasa de reciclaje de residuos
    'w': 0.3,        # Factor de sabiduría
    'lambda_val': 0.1 # Tasa máxima de retroalimentación
}

# Función de Hill
def hill_function(R):
    return R / (1 + R)

# Ecuaciones del modelo
def model(y, t, params):
    S, E, I, R = y
    dSdt = params['xi'] * (1 - params['w']) * R - params['alpha'] * S + params['lambda_val'] * hill_function(R)
    dEdt = params['alpha'] * S - params['sigma'] * E
    dIdt = params['sigma'] * E - params['gamma'] * I
    dRdt = params['gamma'] * I - params['xi'] * (1 - params['w']) * R - params['lambda_val'] * hill_function(R)
    return [dSdt, dEdt, dIdt, dRdt]

# Condiciones iniciales y tiempo
y0 = [0.7, 0.2, 0.05, 0.05]  # S, E, I, R
t = np.linspace(0, 50, 1000)

# Resolver ecuaciones
solution = odeint(model, y0, t, args=(params,))
S, E, I, R = solution.T

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b-', label='Karma Latente (S)')
plt.plot(t, E, 'y-', label='Karma Activado (E)')
plt.plot(t, I, 'r-', label='Karma Manifestado (I)')
plt.plot(t, R, 'g-', label='Karma Resuelto (R)')
plt.title('Dinámica SEIRS-Karma')
plt.xlabel('Tiempo')
plt.ylabel('Proporción')
plt.legend()
plt.grid(True)
plt.savefig('seirs_karma.png')  # Guardará en el mismo directorio
plt.show()
