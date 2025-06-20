# scripts/five_poisons.py
"""
KARMIC DYNAMICS SIMULATION FRAMEWORK - EXPERIMENTAL VERSION

DISCLAIMER: 
This code implements speculative mathematical models bridging 
epidemiology and Buddhist philosophy. It is intended for:
- Conceptual exploration
- Methodological experimentation
- Interdisciplinary dialogue

NOT FOR:
- Doctrinal interpretation
- Clinical application
- Metaphysical claims

All models are provisional abstractions subject to revision.
Parameter values are heuristic estimates without empirical validation.
"""
import numpy as np
from scipy.integrate import solve_ivp  # Usamos solve_ivp en lugar de odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Función para asegurar que exista el directorio
def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# Parámetros con acoplamientos asimétricos (basados en Tabla A.1 del paper)
params = {
    # Tasas de autorefuerzo
    'alpha_I': 0.25,  # Ignorance
    'alpha_A': 0.20,  # Attachment
    'alpha_V': 0.30,  # Aversion
    'alpha_P': 0.35,  # Pride-Greed
    'alpha_E': 0.28,  # Envy-Fear
    
    # Coeficientes de acoplamiento ASIMÉTRICOS (β_ij ≠ β_ji)
    # Ignorance
    'beta_IP': 0.8,   # Pride → Ignorance (fuerte)
    'beta_IE': 0.3,   # Envy → Ignorance
    'beta_IV': 0.2,   # Aversion → Ignorance
    
    # Attachment
    'beta_AP': 0.6,   # Pride → Attachment
    'beta_AE': 0.4,   # Envy → Attachment
    
    # Aversion
    'beta_VE': 0.5,   # Envy → Aversion
    'beta_VI': 0.4,   # Ignorance → Aversion
    
    # Pride-Greed
    'beta_PA': 0.1,   # Attachment → Pride
    'beta_PI': 0.05,  # Ignorance → Pride
    'beta_PE': 0.1,   # Envy → Pride
    
    # Envy-Fear
    'beta_EV': 0.3,   # Aversion → Envy
    'beta_EP': 0.2,   # Pride → Envy
    'beta_EA': 0.15,  # Attachment → Envy
    
    # Sensibilidad a la sabiduría
    'gamma_I': 0.4,
    'gamma_A': 0.35,
    'gamma_V': 0.45,
    'gamma_P': 0.5,
    'gamma_E': 0.38,
    
    # Factor de sabiduría
    'w': 0.25
}

# Ecuaciones de los Cinco Venenos Mentales con protección numérica
def model(t, y, params):  # Cambiamos la firma para solve_ivp
    I, A, V, P, E = y
    
    # Aplicar protección numérica - evitar valores negativos o cero
    I = max(I, 1e-10)
    A = max(A, 1e-10)
    V = max(V, 1e-10)
    P = max(P, 1e-10)
    E = max(E, 1e-10)
    
    # Términos de crecimiento
    growth_I = params['alpha_I'] * I
    growth_A = params['alpha_A'] * A
    growth_V = params['alpha_V'] * V
    growth_P = params['alpha_P'] * P
    growth_E = params['alpha_E'] * E
    
    # Términos de interacción - con escalado para estabilidad
    interaction_I = params['beta_IP'] * I * P + params['beta_IE'] * I * E
    interaction_A = params['beta_AP'] * A * P + params['beta_AE'] * A * E
    interaction_V = params['beta_VE'] * V * E + params['beta_VI'] * V * I
    interaction_P = params['beta_PA'] * P * A + params['beta_PI'] * P * I
    interaction_E = params['beta_EV'] * E * V + params['beta_EP'] * E * P
    
    # Términos de sabiduría (reducción)
    wisdom_I = params['gamma_I'] * params['w'] * I
    wisdom_A = params['gamma_A'] * params['w'] * A
    wisdom_V = params['gamma_V'] * params['w'] * V
    wisdom_P = params['gamma_P'] * params['w'] * P
    wisdom_E = params['gamma_E'] * params['w'] * E
    
    # Ecuaciones diferenciales con escalado de derivadas
    dIdt = (growth_I + interaction_I - wisdom_I) / (1 + abs(growth_I + interaction_I - wisdom_I))
    dAdt = (growth_A + interaction_A - wisdom_A) / (1 + abs(growth_A + interaction_A - wisdom_A))
    dVdt = (growth_V + interaction_V - wisdom_V) / (1 + abs(growth_V + interaction_V - wisdom_V))
    dPdt = (growth_P + interaction_P - wisdom_P) / (1 + abs(growth_P + interaction_P - wisdom_P))
    dEdt = (growth_E + interaction_E - wisdom_E) / (1 + abs(growth_E + interaction_E - wisdom_E))
    
    return [dIdt, dAdt, dVdt, dPdt, dEdt]

# Condiciones iniciales (ejemplo: estado con predominio de orgullo y envidia)
y0 = [0.1, 0.2, 0.1, 0.5, 0.4]  # I, A, V, P, E
t = np.linspace(0, 50, 5000)  # Reducimos el tiempo de simulación

# Resolver las ecuaciones diferenciales con solve_ivp
solution = solve_ivp(
    fun=model,
    t_span=[t[0], t[-1]],
    y0=y0,
    t_eval=t,
    args=(params,),
    method='BDF',  # Método para sistemas rígidos
    rtol=1e-6,
    atol=1e-8
)

# Extraer solución
I, A, V, P, E = solution.y

# Crear gráficos
plt.figure(figsize=(14, 10))

# Gráfico de series temporales
plt.subplot(3, 2, 1)
plt.plot(t, I, 'b-', label='Ignorance (I)')
plt.plot(t, A, 'g-', label='Attachment (A)')
plt.plot(t, V, 'r-', label='Aversion (V)')
plt.plot(t, P, 'm-', label='Pride-Greed (P)')
plt.plot(t, E, 'c-', label='Envy-Fear (E)')
plt.title('Evolution of the Five Mental Poisons')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)

# Diagramas de fase 2D
combinations = [
    ('I', 'P', 'b', 'm', 'Ignorance vs Pride-Greed'),
    ('A', 'E', 'g', 'c', 'Attachment vs Envy-Fear'),
    ('V', 'I', 'r', 'b', 'Aversion vs Ignorance'),
    ('P', 'E', 'm', 'c', 'Pride-Greed vs Envy-Fear')
]

# Diccionario para acceso a variables
var_dict = {
    'I': I, 'A': A, 'V': V, 'P': P, 'E': E
}

for i, (x_var, y_var, x_color, y_color, title) in enumerate(combinations, 2):
    plt.subplot(3, 2, i)
    plt.plot(var_dict[x_var], var_dict[y_var], color=x_color)
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True)

plt.tight_layout()

# Guardar gráficos 2D
save_path_2d = 'figs/five_poisons_2d.png'
ensure_directory_exists(save_path_2d)
plt.savefig(save_path_2d)

# Diagrama de fase 3D para tres venenos principales
fig = plt.figure(figsize=(14, 10))

# Combinación 1: Ignorance, Pride, Envy
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(I, P, E, 'b-', linewidth=0.7)
ax1.set_title('Attractor: Ignorance-Pride-Envy')
ax1.set_xlabel('Ignorance (I)')
ax1.set_ylabel('Pride-Greed (P)')
ax1.set_zlabel('Envy-Fear (E)')

# Combinación 2: Attachment, Aversion, Pride
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(A, V, P, 'g-', linewidth=0.7)
ax2.set_title('Attractor: Attachment-Aversion-Pride')
ax2.set_xlabel('Attachment (A)')
ax2.set_ylabel('Aversion (V)')
ax2.set_zlabel('Pride-Greed (P)')

# Combinación 3: Ignorance, Aversion, Envy
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(I, V, E, 'r-', linewidth=0.7)
ax3.set_title('Attractor: Ignorance-Aversion-Envy')
ax3.set_xlabel('Ignorance (I)')
ax3.set_ylabel('Aversion (V)')
ax3.set_zlabel('Envy-Fear (E)')

# Combinación 4: Todos los venenos (proyección)
ax4 = fig.add_subplot(224, projection='3d')
sc = ax4.scatter(I, A, V, c=P, s=np.clip(E*50, 0.1, 100), cmap='viridis', alpha=0.7)
ax4.set_title('Multidimensional Projection (Size: Envy, Color: Pride)')
ax4.set_xlabel('Ignorance (I)')
ax4.set_ylabel('Attachment (A)')
ax4.set_zlabel('Aversion (V)')
fig.colorbar(sc, label='Pride-Greed (P)')

plt.tight_layout()

# Guardar gráficos 3D
save_path_3d = 'figs/five_poisons_3d.png'
ensure_directory_exists(save_path_3d)
plt.savefig(save_path_3d)
plt.show()

# Análisis de estabilidad
final_values = solution.y[:, -1]  # Último punto en la solución
print("\nFinal Analysis of the Five Poisons:")
print(f"  Ignorance: {final_values[0]:.4f}")
print(f"  Attachment: {final_values[1]:.4f}")
print(f"  Aversion: {final_values[2]:.4f}")
print(f"  Pride-Greed: {final_values[3]:.4f}")
print(f"  Envy-Fear: {final_values[4]:.4f}")

# Identificar veneno dominante
dominant_idx = np.argmax(final_values)
dominant_names = ['Ignorance', 'Attachment', 'Aversion', 'Pride-Greed', 'Envy-Fear']
print(f"\nDominant poison: {dominant_names[dominant_idx]}")

# Relación con reinos samsáricos (Tabla 1 del paper)
realm_mapping = {
    'Pride-Greed': 'Devas (Gods)',
    'Envy-Fear': 'Asuras (Demigods)',
    'Attachment': 'Humans',
    'Ignorance': 'Animals',
    'Aversion': 'Naraka (Hells)'
}

print(f"  Corresponding realm: {realm_mapping[dominant_names[dominant_idx]]}")

# Generalización de la condición de iluminación
def generalized_enlightenment_condition(params):
    """Calculates a generalized enlightenment condition for 5 variables"""
    # Coeficientes de acoplamiento total por variable
    coupling_strengths = {
        'I': params['beta_IP'] + params['beta_IE'],
        'A': params['beta_AP'] + params['beta_AE'],
        'V': params['beta_VE'] + params['beta_VI'],
        'P': params['beta_PA'] + params['beta_PI'] + params['beta_PE'],
        'E': params['beta_EV'] + params['beta_EP'] + params['beta_EA']
    }
    
    # Calcular w crítico para cada veneno
    w_critical = {}
    w_critical['I'] = (params['alpha_I'] + 0.5 * coupling_strengths['I']) / params['gamma_I']
    w_critical['A'] = (params['alpha_A'] + 0.5 * coupling_strengths['A']) / params['gamma_A']
    w_critical['V'] = (params['alpha_V'] + 0.5 * coupling_strengths['V']) / params['gamma_V']
    w_critical['P'] = (params['alpha_P'] + 0.5 * coupling_strengths['P']) / params['gamma_P']
    w_critical['E'] = (params['alpha_E'] + 0.5 * coupling_strengths['E']) / params['gamma_E']
    
    # El máximo de estos valores es el w mínimo requerido
    w_min = max(w_critical.values())
    
    return w_min, w_critical

# Calcular y mostrar la condición de iluminación
w_min_required, w_critical_values = generalized_enlightenment_condition(params)
print("\nGeneralized Enlightenment Condition:")
for poison, w_crit in w_critical_values.items():
    print(f"  {poison}: w > {w_crit:.4f}")

print(f"\n  Current wisdom (w): {params['w']}")
print(f"  Minimum wisdom required: {w_min_required:.4f}")

if params['w'] > w_min_required:
    print("  STATE: Conditions for enlightenment SATISFIED")
else:
    print("  STATE: Conditions for enlightenment NOT satisfied")
    
# Verificar convergencia a cero
tolerance = 1e-3
if all(abs(val) < tolerance for val in final_values):
    print("  RESULT: System converges to enlightenment state (0,0,0,0,0)")
else:
    print("  RESULT: System does NOT converge to zero (persistent samsara)")
