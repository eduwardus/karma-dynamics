# scripts/five_poisons.py
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    'beta_PI': 0.05,  # Ignorance → Pride (débil)
    'beta_IE': 0.3,   # Envy → Ignorance
    'beta_EI': 0.1,   # Ignorance → Envy
    
    # Attachment
    'beta_AP': 0.6,   # Pride → Attachment
    'beta_PA': 0.1,   # Attachment → Pride
    'beta_AE': 0.4,   # Envy → Attachment
    'beta_EA': 0.15,  # Attachment → Envy
    
    # Aversion
    'beta_VE': 0.5,   # Envy → Aversion
    'beta_EV': 0.3,   # Aversion → Envy
    'beta_VI': 0.4,   # Ignorance → Aversion
    'beta_IV': 0.2,   # Aversion → Ignorance
    
    # Pride-Greed
    'beta_PA': 0.1,   # Attachment → Pride (ya definido)
    'beta_PI': 0.05,  # Ignorance → Pride (ya definido)
    
    # Envy-Fear
    'beta_EV': 0.3,   # Aversion → Envy (ya definido)
    'beta_EP': 0.2,   # Pride → Envy
    'beta_PE': 0.1,   # Envy → Pride
    
    # Sensibilidad a la sabiduría
    'gamma_I': 0.4,
    'gamma_A': 0.35,
    'gamma_V': 0.45,
    'gamma_P': 0.5,
    'gamma_E': 0.38,
    
    # Factor de sabiduría
    'w': 0.25
}

# Ecuaciones de los Cinco Venenos Mentales
def model(y, t, params):
    I, A, V, P, E = y
    
    # dI/dt = α_I I + β_IP I P + β_IE I E - γ_I w I
    dIdt = (params['alpha_I'] * I + 
            params['beta_IP'] * I * P + 
            params['beta_IE'] * I * E - 
            params['gamma_I'] * params['w'] * I)
    
    # dA/dt = α_A A + β_AP A P + β_AE A E - γ_A w A
    dAdt = (params['alpha_A'] * A + 
            params['beta_AP'] * A * P + 
            params['beta_AE'] * A * E - 
            params['gamma_A'] * params['w'] * A)
    
    # dV/dt = α_V V + β_VE V E + β_VI V I - γ_V w V
    dVdt = (params['alpha_V'] * V + 
            params['beta_VE'] * V * E + 
            params['beta_VI'] * V * I - 
            params['gamma_V'] * params['w'] * V)
    
    # dP/dt = α_P P + β_PA P A + β_PI P I - γ_P w P
    dPdt = (params['alpha_P'] * P + 
            params['beta_PA'] * P * A + 
            params['beta_PI'] * P * I - 
            params['gamma_P'] * params['w'] * P)
    
    # dE/dt = α_E E + β_EV E V + β_EP E P - γ_E w E
    dEdt = (params['alpha_E'] * E + 
            params['beta_EV'] * E * V + 
            params['beta_EP'] * E * P - 
            params['gamma_E'] * params['w'] * E)
    
    return [dIdt, dAdt, dVdt, dPdt, dEdt]

# Condiciones iniciales (ejemplo: estado con predominio de orgullo y envidia)
y0 = [0.1, 0.2, 0.1, 0.5, 0.4]  # I, A, V, P, E
t = np.linspace(0, 100, 5000)  # Tiempo de simulación más largo

# Resolver las ecuaciones diferenciales
solution = odeint(model, y0, t, args=(params,))
I, A, V, P, E = solution.T

# Crear gráficos
plt.figure(figsize=(14, 10))

# Gráfico de series temporales
plt.subplot(3, 2, 1)
plt.plot(t, I, 'b-', label='Ignorance (I)')
plt.plot(t, A, 'g-', label='Attachment (A)')
plt.plot(t, V, 'r-', label='Aversion (V)')
plt.plot(t, P, 'm-', label='Pride-Greed (P)')
plt.plot(t, E, 'c-', label='Envy-Fear (E)')
plt.title('Evolución de los Cinco Venenos Mentales')
plt.xlabel('Tiempo')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True)

# Diagramas de fase 2D
combinations = [
    ('I', 'P', 'b', 'm', 'Ignorance vs Pride-Greed'),
    ('A', 'E', 'g', 'c', 'Attachment vs Envy-Fear'),
    ('V', 'I', 'r', 'b', 'Aversion vs Ignorance'),
    ('P', 'E', 'm', 'c', 'Pride-Greed vs Envy-Fear')
]

for i, (x_var, y_var, x_color, y_color, title) in enumerate(combinations, 2):
    plt.subplot(3, 2, i)
    plt.plot(eval(x_var), eval(y_var), color=x_color)
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True)

plt.tight_layout()
plt.savefig('../figs/five_poisons_2d.png')

# Diagrama de fase 3D para tres venenos principales
fig = plt.figure(figsize=(14, 10))

# Combinación 1: Ignorance, Pride, Envy
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(I, P, E, 'b-', linewidth=0.7)
ax1.set_title('Atractor: Ignorance-Pride-Envy')
ax1.set_xlabel('Ignorance (I)')
ax1.set_ylabel('Pride-Greed (P)')
ax1.set_zlabel('Envy-Fear (E)')

# Combinación 2: Attachment, Aversion, Pride
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(A, V, P, 'g-', linewidth=0.7)
ax2.set_title('Atractor: Attachment-Aversion-Pride')
ax2.set_xlabel('Attachment (A)')
ax2.set_ylabel('Aversion (V)')
ax2.set_zlabel('Pride-Greed (P)')

# Combinación 3: Ignorance, Aversion, Envy
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(I, V, E, 'r-', linewidth=0.7)
ax3.set_title('Atractor: Ignorance-Aversion-Envy')
ax3.set_xlabel('Ignorance (I)')
ax3.set_ylabel('Aversion (V)')
ax3.set_zlabel('Envy-Fear (E)')

# Combinación 4: Todos los venenos (proyección)
ax4 = fig.add_subplot(224, projection='3d')
sc = ax4.scatter(I, A, V, c=P, s=E*50, cmap='viridis', alpha=0.7)
ax4.set_title('Proyección Multidimensional (Tamaño: Envy, Color: Pride)')
ax4.set_xlabel('Ignorance (I)')
ax4.set_ylabel('Attachment (A)')
ax4.set_zlabel('Aversion (V)')
fig.colorbar(sc, label='Pride-Greed (P)')

plt.tight_layout()
plt.savefig('../figs/five_poisons_3d.png')
plt.show()

# Análisis de estabilidad
final_values = solution[-1]
print("\nAnálisis Final de los Cinco Venenos:")
print(f"  Ignorance: {final_values[0]:.4f}")
print(f"  Attachment: {final_values[1]:.4f}")
print(f"  Aversion: {final_values[2]:.4f}")
print(f"  Pride-Greed: {final_values[3]:.4f}")
print(f"  Envy-Fear: {final_values[4]:.4f}")

# Identificar veneno dominante
dominant_idx = np.argmax(final_values)
dominant_names = ['Ignorance', 'Attachment', 'Aversion', 'Pride-Greed', 'Envy-Fear']
print(f"\nVeneno dominante: {dominant_names[dominant_idx]}")

# Relación con reinos samsáricos (Tabla 1 del paper)
realm_mapping = {
    'Pride-Greed': 'Devas (Gods)',
    'Envy-Fear': 'Asuras (Demigods)',
    'Attachment': 'Humans',
    'Ignorance': 'Animals',
    'Aversion': 'Naraka (Hells)'
}

print(f"  Realm correspondiente: {realm_mapping[dominant_names[dominant_idx]]}")

# Generalización de la condición de iluminación
def generalized_enlightenment_condition(params):
    """Calcula una condición de iluminación generalizada para 5 variables"""
    # Coeficientes de acoplamiento total por variable
    coupling_strengths = {
        'I': params['beta_IP'] + params['beta_IE'],
        'A': params['beta_AP'] + params['beta_AE'],
        'V': params['beta_VE'] + params['beta_VI'],
        'P': params['beta_PA'] + params['beta_PI'],
        'E': params['beta_EV'] + params['beta_EP']
    }
    
    # Calcular w crítico para cada veneno
    w_critical = {}
    w_critical['I'] = (params['alpha_I'] + 0.5*coupling_strengths['I']) / params['gamma_I']
    w_critical['A'] = (params['alpha_A'] + 0.5*coupling_strengths['A']) / params['gamma_A']
    w_critical['V'] = (params['alpha_V'] + 0.5*coupling_strengths['V']) / params['gamma_V']
    w_critical['P'] = (params['alpha_P'] + 0.5*coupling_strengths['P']) / params['gamma_P']
    w_critical['E'] = (params['alpha_E'] + 0.5*coupling_strengths['E']) / params['gamma_E']
    
    # El máximo de estos valores es el w mínimo requerido
    w_min = max(w_critical.values())
    
    return w_min, w_critical

# Calcular y mostrar la condición de iluminación
w_min_required, w_critical_values = generalized_enlightenment_condition(params)
print("\nCondición Generalizada de Iluminación:")
for poison, w_crit in w_critical_values.items():
    print(f"  {poison}: w > {w_crit:.4f}")

print(f"\n  Sabiduría actual (w): {params['w']}")
print(f"  Sabiduría mínima requerida: {w_min_required:.4f}")

if params['w'] > w_min_required:
    print("  ESTADO: Condiciones para iluminación SATISFECHAS")
else:
    print("  ESTADO: Condiciones para iluminación NO satisfechas")
    
# Verificar convergencia a cero
tolerance = 1e-3
if all(abs(val) < tolerance for val in final_values):
    print("  RESULTADO: El sistema converge al estado de iluminación (0,0,0,0,0)")
else:
    print("  RESULTADO: El sistema NO converge a cero (samsara persistente)")
