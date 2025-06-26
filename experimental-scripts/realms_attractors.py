# scripts/realms_attractors.py
"""
KARMIC DYNAMICS SIMULATION FRAMEWORK - EXPERIMENTAL VERSION
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import warnings

# Suprimir warnings específicos para mejorar la legibilidad
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================
# Modelo de Tres Raíces Kármicas (ORIGINAL)
# =============================================
def three_roots_model(t, y, params):
    I, A, V = y
    # Manejo de valores extremos para evitar overflow
    I = np.clip(I, 1e-10, 1e10)
    A = np.clip(A, 1e-10, 1e10)
    V = np.clip(V, 1e-10, 1e10)
    
    dIdt = params['alpha_I']*I + params['beta_IA']*A*V - params['gamma_I']*params['w']*I
    dAdt = params['alpha_A']*A + params['beta_AV']*V*I - params['gamma_A']*params['w']*A
    dVdt = params['alpha_V']*V + params['beta_VI']*I*A - params['gamma_V']*params['w']*V
    return [dIdt, dAdt, dVdt]

# Evento para detener la integración cuando el sistema explota
def explosion_event(t, y, params):
    return np.max(y) - 1e5  # Detener cuando cualquier variable excede 100,000
explosion_event.terminal = True
explosion_event.direction = 1

# =============================================
# Configuraciones de Parámetros para cada Reino
# =============================================
def get_realm_config(realm_name):
    configs = {
        'devas': {
            'params': {'alpha_I': 0.1, 'alpha_A': 0.05, 'alpha_V': 0.08, 'beta_IA': 0.2, 'beta_AV': 0.1, 'beta_VI': 0.15, 'gamma_I': 0.3, 'gamma_A': 0.25, 'gamma_V': 0.35, 'w': 0.45},
            'y0': [0.1, 0.7, 0.1],
            'color': 'gold',
            't_span': (0, 100),
            't_eval': np.linspace(0, 100, 5000)
        },
        'asuras': {
            'params': {'alpha_I': 0.15, 'alpha_A': 0.1, 'alpha_V': 0.25, 'beta_IA': 0.6, 'beta_AV': 0.7, 'beta_VI': 0.5, 'gamma_I': 0.2, 'gamma_A': 0.15, 'gamma_V': 0.25, 'w': 0.25},
            'y0': [0.1, 0.2, 0.6],
            'color': 'darkorange',
            't_span': (0, 200),
            't_eval': np.linspace(0, 200, 10000)
        },
        'humans': {
            'params': {'alpha_I': 0.25, 'alpha_A': 0.3, 'alpha_V': 0.2, 'beta_IA': 0.8, 'beta_AV': 0.9, 'beta_VI': 0.85, 'gamma_I': 0.3, 'gamma_A': 0.35, 'gamma_V': 0.25, 'w': 0.15},
            'y0': [0.2, 0.5, 0.2],
            'color': 'green',
            't_span': (0, 300),
            't_eval': np.linspace(0, 300, 15000)
        },
        'animals': {
            'params': {'alpha_I': 0.4, 'alpha_A': 0.1, 'alpha_V': 0.15, 'beta_IA': 0.3, 'beta_AV': 0.25, 'beta_VI': 0.2, 'gamma_I': 0.25, 'gamma_A': 0.35, 'gamma_V': 0.3, 'w': 0.1},
            'y0': [0.7, 0.15, 0.1],
            'color': 'brown',
            't_span': (0, 500),
            't_eval': np.linspace(0, 500, 20000)
        },
        'pretas': {
            'params': {'alpha_I': 0.2, 'alpha_A': 0.25, 'alpha_V': 0.1, 'beta_IA': 1.2, 'beta_AV': 0.8, 'beta_VI': 1.0, 'gamma_I': 0.2, 'gamma_A': 0.15, 'gamma_V': 0.25, 'w': 0.05},
            'y0': [0.3, 0.4, 0.1],
            'color': 'purple',
            't_span': (0, 20),
            't_eval': np.linspace(0, 20, 10000)
        },
        'naraka': {
            'params': {'alpha_I': 0.35, 'alpha_A': 0.1, 'alpha_V': 0.4, 'beta_IA': 1.5, 'beta_AV': 1.2, 'beta_VI': 1.8, 'gamma_I': 0.15, 'gamma_A': 0.2, 'gamma_V': 0.1, 'w': 0.01},
            'y0': [0.2, 0.1, 0.7],
            'color': 'red',
            't_span': (0, 10),
            't_eval': np.linspace(0, 10, 5000)
        }
    }
    return configs[realm_name]

# =============================================
# Visualización de Atractores
# =============================================
def plot_attractor_3d(ax, solution, title, color):
    """Visualiza un atractor en 3D"""
    I, A, V = solution
    ax.plot(I, A, V, color=color, linewidth=0.7, alpha=0.8)
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
    I, A, V = solution
    ax.plot(t, I, 'b-', alpha=0.7, label='Ignorancia')
    ax.plot(t, A, 'g-', alpha=0.7, label='Apego')
    ax.plot(t, V, 'r-', alpha=0.7, label='Aversión')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Intensidad')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

# =============================================
# Simulación Principal con manejo robusto
# =============================================
def main():
    plt.figure(figsize=(16, 12))
    plt.suptitle('Reinos Samsáricos como Atractores Dinámicos', fontsize=16, y=0.98)
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1])
    
    realms = ['devas', 'asuras', 'humans', 'animals', 'pretas', 'naraka']
    realm_names = ['Devas (Dioses)', 'Asuras (Semidioses)', 'Humanos', 'Animales', 'Pretas (Esp. Hamb.)', 'Naraka (Infiernos)']
    attractor_types = ['Punto Fijo Estable', 'Ciclo Límite', 'Atractor Caótico Débil', 
                      'Toro Cuasiperiódico', 'Órbita Periódica Inestable', 'Atractor Caótico Fuerte']
    
    for i, realm in enumerate(realms):
        config = get_realm_config(realm)
        
        try:
            # Configuración para reinos problemáticos
            if realm in ['pretas', 'naraka']:
                sol = solve_ivp(
                    three_roots_model,
                    config['t_span'],
                    config['y0'],
                    args=(config['params'],),
                    method='LSODA',
                    dense_output=True,
                    events=[explosion_event],
                    rtol=1e-3,
                    atol=1e-4,
                    max_step=0.01
                )
            else:
                sol = solve_ivp(
                    three_roots_model,
                    config['t_span'],
                    config['y0'],
                    args=(config['params'],),
                    method='LSODA',
                    dense_output=True,
                    events=[explosion_event],
                    rtol=1e-6,
                    atol=1e-8
                )
            
            # Verificar si la integración fue exitosa
            if sol.status == -1 and sol.t_events[0].size > 0:
                print(f"Advertencia: Integración interrumpida para {realm_names[i]} en t={sol.t_events[0][0]:.2f}")
                # Evaluar hasta el punto de interrupción
                t_eval = np.linspace(config['t_span'][0], min(sol.t_events[0][0], config['t_span'][1]), 1000)
                solution_values = sol.sol(t_eval)
            else:
                t_eval = config['t_eval']
                solution_values = sol.sol(t_eval)
            
            # Diagrama de fase 3D
            ax1 = plt.subplot(gs[i, 0], projection='3d')
            plot_attractor_3d(ax1, solution_values, 
                             f"{realm_names[i]} - {attractor_types[i]}", 
                             config['color'])
            
            # Series temporales (primeros 20 unidades o menos)
            ax2 = plt.subplot(gs[i, 1])
            max_time = min(20, config['t_span'][1])
            mask = t_eval <= max_time
            plot_time_series(ax2, t_eval[mask], 
                            solution_values[:, mask],
                            f"Evolución Temporal: {realm_names[i]}", 
                            config['color'])
            
        except Exception as e:
            print(f"Error grave en {realm_names[i]}: {str(e)}")
            # Crear gráficos vacíos para mantener el layout
            ax1 = plt.subplot(gs[i, 0], projection='3d')
            ax1.set_title(f"Error en {realm_names[i]}", color='red')
            ax1.text(0.5, 0.5, 0.5, "Simulación fallida", transform=ax1.transAxes)
            
            ax2 = plt.subplot(gs[i, 1])
            ax2.set_title(f"Error en {realm_names[i]}", color='red')
            ax2.text(0.5, 0.5, "Simulación fallida", transform=ax2.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figs/realms_attractors.png', dpi=150)
    plt.show()
    
    # Análisis de sensibilidad a la sabiduría
    analyze_wisdom_sensitivity()

# =============================================
# Análisis de Sensibilidad a la Sabiduría
# =============================================
def analyze_wisdom_sensitivity():
    realm = 'humans'
    config = get_realm_config(realm)
    w_values = [0.01, 0.15, 0.3, 0.5, 0.7, 0.9]
    t_span = (0, 100)
    t_eval = np.linspace(0, 100, 5000)
    
    plt.figure(figsize=(14, 8))
    
    for i, w in enumerate(w_values):
        params = config['params'].copy()
        params['w'] = w
        
        try:
            # Configuración especial para valores bajos de w
            if w < 0.1:
                sol = solve_ivp(
                    three_roots_model,
                    t_span,
                    config['y0'],
                    args=(params,),
                    method='LSODA',
                    dense_output=True,
                    events=[explosion_event],
                    rtol=1e-3,
                    atol=1e-4,
                    max_step=0.01
                )
            else:
                sol = solve_ivp(
                    three_roots_model,
                    t_span,
                    config['y0'],
                    args=(params,),
                    method='LSODA',
                    dense_output=True,
                    events=[explosion_event],
                    rtol=1e-6,
                    atol=1e-8
                )
            
            if sol.status == -1 and sol.t_events[0].size > 0:
                print(f"Advertencia: Integración interrumpida para w={w} en t={sol.t_events[0][0]:.2f}")
                t_eval_short = np.linspace(0, min(sol.t_events[0][0], t_span[1]), 1000)
                solution_values = sol.sol(t_eval_short)
            else:
                solution_values = sol.sol(t_eval)
            
            I, A, V = solution_values
            
            # Determinar tipo de dinámica
            if solution_values.size == 0 or np.isnan(I).any():
                realm_type = "Error"
            else:
                # Usar solo los últimos 100 puntos si hay suficientes
                last_points = min(100, len(I))
                std_dev = np.std(I[-last_points:])
                final_values = solution_values[:, -1]
                
                if np.mean(final_values) < 0.05:
                    realm_type = "Cercano a Iluminación"
                elif std_dev < 0.01:
                    realm_type = "Punto Fijo"
                elif 0.01 <= std_dev < 0.1:
                    realm_type = "Ciclo Límite"
                else:
                    realm_type = "Caótico"
            
            # Gráfico de evolución
            plt.subplot(2, 3, i+1)
            plt.plot(t_eval[:len(I)] if len(I) < len(t_eval) else t_eval, I, 'b-', alpha=0.6, label='Ignorancia')
            plt.plot(t_eval[:len(A)] if len(A) < len(t_eval) else t_eval, A, 'g-', alpha=0.6, label='Apego')
            plt.plot(t_eval[:len(V)] if len(V) < len(t_eval) else t_eval, V, 'r-', alpha=0.6, label='Aversión')
            plt.title(f'w = {w:.2f} - {realm_type}')
            plt.xlabel('Tiempo')
            plt.ylabel('Intensidad')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            if i == 0:
                plt.legend()
                
        except Exception as e:
            print(f"Error en w={w}: {str(e)}")
            plt.subplot(2, 3, i+1)
            plt.text(0.5, 0.5, f"Error en simulación", transform=plt.gca().transAxes, ha='center')
            plt.title(f'w = {w:.2f} - Error')
            plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Sensibilidad a la Sabiduría (w) en el Reino Humano', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figs/wisdom_sensitivity.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()                                                                                                                              