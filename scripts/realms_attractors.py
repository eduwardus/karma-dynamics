# scripts/realms_attractors.py
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
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# =============================================
# Modelo de Tres Raíces Kármicas (base)
# =============================================
def three_roots_model(y, t, params):
    I, A, V = y
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# =============================================
# Configuraciones de Parámetros para cada Reino
# =============================================
def get_realm_config(realm_name):
    """Devuelve parámetros y condiciones iniciales para cada reino samsárico"""
    configs = {
        # Devas (Gods) - Punto fijo estable
        'devas': {
            'params': {
                'alpha_I': 0.1, 'alpha_A': 0.05, 'alpha_V': 0.08,
                'beta_IA': 0.2, 'beta_AV': 0.1, 'beta_VI': 0.15,
                'gamma_I': 0.3, 'gamma_A': 0.25, 'gamma_V': 0.35,
                'w': 0.45  # Alta sabiduría estabiliza
            },
            'y0': [0.1, 0.7, 0.1],  # Predominio de apego (A)
            'color': 'gold',
            't': np.linspace(0, 100, 5000)
        },
        
        # Asuras (Demigods) - Ciclo límite
        'asuras': {
            'params': {
                'alpha_I': 0.15, 'alpha_A': 0.1, 'alpha_V': 0.25,
                'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5,
                'gamma_I': 0.2, 'gamma_A': 0.15, 'gamma_V': 0.25,
                'w': 0.25  # Sabiduría moderada
            },
            'y0': [0.1, 0.2, 0.6],  # Predominio de aversión (V)
            'color': 'darkorange',
            't': np.linspace(0, 200, 10000)
        },
        
        # Humans - Atractor caótico débil
        'humans': {
            'params': {
                'alpha_I': 0.25, 'alpha_A': 0.3, 'alpha_V': 0.2,
                'beta_IA': 0.8, 'beta_AV': 0.9, 'beta_VI': 0.85,
                'gamma_I': 0.3, 'gamma_A': 0.35, 'gamma_V': 0.25,
                'w': 0.15  # Baja sabiduría
            },
            'y0': [0.2, 0.5, 0.2],  # Predominio de apego (A)
            'color': 'green',
            't': np.linspace(0, 300, 15000)
        },
        
        # Animals - Toro cuasiperiódico
        'animals': {
            'params': {
                'alpha_I': 0.4, 'alpha_A': 0.1, 'alpha_V': 0.15,
                'beta_IA': 0.3, 'beta_AV': 0.25, 'beta_VI': 0.2,
                'gamma_I': 0.25, 'gamma_A': 0.35, 'gamma_V': 0.3,
                'w': 0.1  # Muy baja sabiduría
            },
            'y0': [0.7, 0.15, 0.1],  # Predominio de ignorancia (I)
            'color': 'brown',
            't': np.linspace(0, 500, 20000)
        },
        
        # Pretas (Hungry Ghosts) - Órbita periódica inestable
        'pretas': {
            'params': {
                'alpha_I': 0.2, 'alpha_A': 0.25, 'alpha_V': 0.1,
                'beta_IA': 1.2, 'beta_AV': 0.8, 'beta_VI': 1.0,
                'gamma_I': 0.2, 'gamma_A': 0.15, 'gamma_V': 0.25,
                'w': 0.05  # Sabiduría casi nula
            },
            'y0': [0.3, 0.4, 0.1],  # Predominio de apego (A)
            'color': 'purple',
            't': np.linspace(0, 400, 20000)
        },
        
        # Naraka (Hells) - Atractor caótico fuerte
        'naraka': {
            'params': {
                'alpha_I': 0.35, 'alpha_A': 0.1, 'alpha_V': 0.4,
                'beta_IA': 1.5, 'beta_AV': 1.2, 'beta_VI': 1.8,
                'gamma_I': 0.15, 'gamma_A': 0.2, 'gamma_V': 0.1,
                'w': 0.01  # Sabiduría mínima
            },
            'y0': [0.2, 0.1, 0.7],  # Predominio de aversión (V)
            'color': 'red',
            't': np.linspace(0, 500, 25000)
        }
    }
    return configs[realm_name]

# =============================================
# Visualización de Atractores
# =============================================
def plot_attractor_3d(ax, solution, title, color):
    """Visualiza un atractor en 3D"""
    I, A, V = solution.T
    ax.plot(I, A, V, color=color, linewidth=0.7, alpha=0.8)
    
    # Configuración estética
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Ignorancia (I)')
    ax.set_ylabel('Apego (A)')
    ax.set_zlabel('Aversión (V)')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True, linestyle=':', alpha=0.6)

def plot_time_series(ax, t, solution, title, color):
    """Visualiza series temporales"""
    I, A, V = solution.T
    ax.plot(t, I, 'b-', alpha=0.7, label='Ignorancia')
    ax.plot(t, A, 'g-', alpha=0.7, label='Apego')
    ax.plot(t, V, 'r-', alpha=0.7, label='Aversión')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Intensidad')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

# =============================================
# Simulación Principal
# =============================================
def main():
    # Configurar figura maestro
    plt.figure(figsize=(16, 12))
    plt.suptitle('Reinos Samsáricos como Atractores Dinámicos', fontsize=16, y=0.98)
    
    # Crear rejilla para visualizaciones
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1])
    
    realms = ['devas', 'asuras', 'humans', 'animals', 'pretas', 'naraka']
    realm_names = ['Devas (Dioses)', 'Asuras (Semidioses)', 'Humanos', 'Animales', 'Pretas (Esp. Hamb.)', 'Naraka (Infiernos)']
    attractor_types = ['Punto Fijo Estable', 'Ciclo Límite', 'Atractor Caótico Débil', 
                      'Toro Cuasiperiódico', 'Órbita Periódica Inestable', 'Atractor Caótico Fuerte']
    
    for i, realm in enumerate(realms):
        # Obtener configuración del reino
        config = get_realm_config(realm)
        
        # Simular dinámica
        solution = odeint(three_roots_model, config['y0'], config['t'], args=(config['params'],))
        
        # Diagrama de fase 3D
        ax1 = plt.subplot(gs[i, 0], projection='3d')
        plot_attractor_3d(ax1, solution, 
                         f"{realm_names[i]} - {attractor_types[i]}", 
                         config['color'])
        
        # Series temporales (solo primeros 100 unidades de tiempo)
        ax2 = plt.subplot(gs[i, 1])
        t_short = config['t'][config['t'] <= 100]
        sol_short = solution[:len(t_short)]
        plot_time_series(ax2, t_short, sol_short, 
                        f"Evolución Temporal: {realm_names[i]}", 
                        config['color'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figs/realms_attractors.png', dpi=150)
    plt.show()
    
    # Análisis de sensibilidad a la sabiduría
    analyze_wisdom_sensitivity()

# =============================================
# Análisis de Sensibilidad a la Sabiduría
# =============================================
def analyze_wisdom_sensitivity():
    """Muestra cómo cambian los reinos al variar el factor de sabiduría"""
    realm = 'humans'  # Podemos usar cualquier reino como base
    config = get_realm_config(realm)
    w_values = [0.01, 0.15, 0.3, 0.5, 0.7, 0.9]
    t = np.linspace(0, 100, 5000)
    
    plt.figure(figsize=(14, 8))
    
    for i, w in enumerate(w_values):
        # Actualizar parámetro de sabiduría
        params = config['params'].copy()
        params['w'] = w
        
        # Simular
        solution = odeint(three_roots_model, config['y0'], t, args=(params,))
        I, A, V = solution.T
        
        # Calcular estado final
        final_state = solution[-1]
        avg_intensity = np.mean(final_state)
        
        # Determinar tipo de dinámica
        if avg_intensity < 0.05:
            realm_type = "Cercano a Iluminación"
        elif np.std([I[-100:], ddof=1) < 0.01:
            realm_type = "Punto Fijo"
        elif 0.01 < np.std([I[-100:]]) < 0.1:
            realm_type = "Ciclo Límite"
        else:
            realm_type = "Caótico"
        
        # Gráfico de evolución
        plt.subplot(2, 3, i+1)
        plt.plot(t, I, 'b-', alpha=0.6, label='Ignorancia')
        plt.plot(t, A, 'g-', alpha=0.6, label='Apego')
        plt.plot(t, V, 'r-', alpha=0.6, label='Aversión')
        plt.title(f'w = {w:.2f} - {realm_type}')
        plt.xlabel('Tiempo')
        plt.ylabel('Intensidad')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            plt.legend()
    
    plt.suptitle('Sensibilidad a la Sabiduría (w) en el Reino Humano', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figs/wisdom_sensitivity.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
