# scripts/three_roots.py
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del modelo (valores de ejemplo basados en la Tabla 1 del paper)
params = {
    # Tasas de autorefuerzo
    'alpha_I': 0.3,
    'alpha_A': 0.25,
    'alpha_V': 0.35,
    
    # Coeficientes de acoplamiento
    'beta_IA': 0.6,   # Attachment + Aversion → Ignorance
    'beta_AV': 0.7,   # Aversion + Ignorance → Attachment
    'beta_VI': 0.5,   # Ignorance + Attachment → Aversion
    
    # Sensibilidad a la sabiduría
    'gamma_I': 0.4,
    'gamma_A': 0.35,
    'gamma_V': 0.45,
    
    # Factor de sabiduría (puede ser constante o función del tiempo)
    'w': 0.2
}

# Ecuaciones del modelo de las Tres Raíces Kármicas
def model(y, t, params):
    I, A, V = y
    
    dIdt = params['alpha_I'] * I + params['beta_IA'] * A * V - params['gamma_I'] * params['w'] * I
    dAdt = params['alpha_A'] * A + params['beta_AV'] * V * I - params['gamma_A'] * params['w'] * A
    dVdt = params['alpha_V'] * V + params['beta_VI'] * I * A - params['gamma_V'] * params['w'] * V
    
    return [dIdt, dAdt, dVdt]

# Condiciones iniciales (ejemplo: estado con predominio de ignorancia)
y0 = [0.8, 0.1, 0.1]  # I, A, V
t = np.linspace(0, 50, 5000)  # Tiempo de simulación

# Resolver las ecuaciones diferenciales
solution = odeint(model, y0, t, args=(params,))
I, A, V = solution.T

# Crear gráficos
plt.figure(figsize=(12, 8))

# Gráfico de series temporales
plt.subplot(2, 2, 1)
plt.plot(t, I, 'b-', label='Ignorancia (I)')
plt.plot(t, A, 'g-', label='Apego (A)')
plt.plot(t, V, 'r-', label='Aversión (V)')
plt.title('Evolución de las Tres Raíces Kármicas')
plt.xlabel('Tiempo')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True)

# Diagrama de fase 2D: Ignorancia vs Apego
plt.subplot(2, 2, 2)
plt.plot(I, A, 'm-')
plt.title('Espacio de Fase: Ignorancia vs Apego')
plt.xlabel('Ignorancia (I)')
plt.ylabel('Apego (A)')
plt.grid(True)

# Diagrama de fase 2D: Ignorancia vs Aversión
plt.subplot(2, 2, 3)
plt.plot(I, V, 'c-')
plt.title('Espacio de Fase: Ignorancia vs Aversión')
plt.xlabel('Ignorancia (I)')
plt.ylabel('Aversión (V)')
plt.grid(True)

# Diagrama de fase 2D: Apego vs Aversión
plt.subplot(2, 2, 4)
plt.plot(A, V, 'y-')
plt.title('Espacio de Fase: Apego vs Aversión')
plt.xlabel('Apego (A)')
plt.ylabel('Aversión (V)')
plt.grid(True)

plt.tight_layout()
plt.savefig('../figs/three_roots_2d.png')

# Diagrama de fase 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(I, A, V, 'b-', linewidth=0.8)
ax.set_title('Atractor de las Tres Raíces Kármicas')
ax.set_xlabel('Ignorancia (I)')
ax.set_ylabel('Apego (A)')
ax.set_zlabel('Aversión (V)')
plt.savefig('../figs/three_roots_3d.png')
plt.show()

# Análisis de la condición de iluminación
def enlightenment_condition(params):
    """Calcula la condición de iluminación según la ecuación (8) del paper"""
    # Calcular el valor crítico para cada raíz
    w_critical_I = (params['alpha_I'] + 0.5*(abs(params['beta_IA'])) / params['gamma_I']
    w_critical_A = (params['alpha_A'] + 0.5*(abs(params['beta_AV'])) / params['gamma_A']
    w_critical_V = (params['alpha_V'] + 0.5*(abs(params['beta_VI'])) / params['gamma_V']
    
    # El máximo de estos valores es el w mínimo requerido para la iluminación
    w_min = max(w_critical_I, w_critical_A, w_critical_V)
    
    return w_min

# Calcular y mostrar la condición de iluminación
w_min_required = enlightenment_condition(params)
print(f"\nAnálisis de Condición de Iluminación:")
print(f"  Sabiduría actual (w): {params['w']}")
print(f"  Sabiduría mínima requerida para iluminación: {w_min_required:.4f}")

if params['w'] > w_min_required:
    print("  ESTADO: Condiciones para iluminación SATISFECHAS")
else:
    print("  ESTADO: Condiciones para iluminación NO satisfechas (samsara persistente)")
    
# Verificar si el sistema converge a cero (estado de iluminación)
final_values = solution[-1]
tolerance = 1e-3
if all(abs(val) < tolerance for val in final_values):
    print("  RESULTADO: El sistema converge al estado de iluminación (0,0,0)")
else:
    print(f"  RESULTADO: El sistema NO converge a cero (valores finales: I={final_values[0]:.4f}, A={final_values[1]:.4f}, V={final_values[2]:.4f})")
