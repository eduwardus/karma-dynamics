# scripts/network_karma.py
"""
NETWORK KARMA DYNAMICS SIMULATION

DISCLAIMER: 
This model explores collective karma formation through social networks as:
- A formal metaphor for interpersonal influence
- A mathematical representation of social habit transmission
- An exploration of wisdom diffusion thresholds

The simulation does NOT imply:
- That social relationships determine individual karma
- That spiritual development is reducible to network topology
- A mechanistic view of ethical responsibility

See Section 8 (Social Physics Implications) of the companion paper for context.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint
from tqdm import tqdm

# =============================================
# 1. Generación de Red Social
# =============================================
def generate_social_network(n_nodes, network_type='scale-free'):
    """Genera diferentes tipos de redes sociales"""
    if network_type == 'scale-free':
        # Red libre de escala (Barabási-Albert)
        G = nx.barabasi_albert_graph(n_nodes, 2)
    elif network_type == 'small-world':
        # Mundo pequeño (Watts-Strogatz)
        G = nx.watts_strogatz_graph(n_nodes, 4, 0.3)
    elif network_type == 'community':
        # Red comunitaria (girvan_newman)
        G = nx.connected_caveman_graph(5, n_nodes//5)
    else:  # Aleatoria
        G = nx.erdos_renyi_graph(n_nodes, 0.1)
    
    return G

# =============================================
# 2. Modelo de Karma Colectivo
# =============================================
def collective_karma_model(y, t, G, params):
    """Ecuaciones diferenciales acopladas para karma en red"""
    n = len(G.nodes)
    I, A, V, P, E, w = y[:n], y[n:2*n], y[2*n:3*n], y[3*n:4*n], y[4*n:5*n], y[5*n:]
    
    # Derivadas temporales
    dIdt = np.zeros(n)
    dAdt = np.zeros(n)
    dVdt = np.zeros(n)
    dPdt = np.zeros(n)
    dEdt = np.zeros(n)
    dwdt = np.zeros(n)
    
    for i in G.nodes:
        # Términos locales (dinámica individual)
        dIdt[i] = (params['alpha_I'] * I[i] + 
                  params['beta_IP'] * I[i] * P[i] + 
                  params['beta_IE'] * I[i] * E[i] - 
                  params['gamma_I'] * w[i] * I[i])
        
        dAdt[i] = (params['alpha_A'] * A[i] + 
                  params['beta_AP'] * A[i] * P[i] + 
                  params['beta_AE'] * A[i] * E[i] - 
                  params['gamma_A'] * w[i] * A[i])
        
        dVdt[i] = (params['alpha_V'] * V[i] + 
                  params['beta_VE'] * V[i] * E[i] + 
                  params['beta_VI'] * V[i] * I[i] - 
                  params['gamma_V'] * w[i] * V[i])
        
        dPdt[i] = (params['alpha_P'] * P[i] + 
                  params['beta_PA'] * P[i] * A[i] + 
                  params['beta_PI'] * P[i] * I[i] - 
                  params['gamma_P'] * w[i] * P[i])
        
        dEdt[i] = (params['alpha_E'] * E[i] + 
                  params['beta_EV'] * E[i] * V[i] + 
                  params['beta_EP'] * E[i] * P[i] - 
                  params['gamma_E'] * w[i] * E[i])
        
        # Términos de red (influencia social)
        neighbors = list(G.neighbors(i))
        if neighbors:
            # Influencia en venenos mentales
            influence_I = params['kappa_I'] * (np.mean([I[j] for j in neighbors]) - I[i])
            influence_A = params['kappa_A'] * (np.mean([A[j] for j in neighbors]) - A[i])
            influence_V = params['kappa_V'] * (np.mean([V[j] for j in neighbors]) - V[i])
            influence_P = params['kappa_P'] * (np.mean([P[j] for j in neighbors]) - P[i])
            influence_E = params['kappa_E'] * (np.mean([E[j] for j in neighbors]) - E[i])
            
            dIdt[i] += influence_I
            dAdt[i] += influence_A
            dVdt[i] += influence_V
            dPdt[i] += influence_P
            dEdt[i] += influence_E
            
            # Difusión de sabiduría
            wisdom_diff = np.mean([w[j] for j in neighbors]) - w[i]
            dwdt[i] = (params['epsilon'] * (1 - w[i]) -  # Crecimiento intrínseco
                      params['mu'] * w[i] +             # Degradación
                      params['delta'] * wisdom_diff)    # Influencia social
    
    return np.concatenate([dIdt, dAdt, dVdt, dPdt, dEdt, dwdt])

# =============================================
# 3. Visualización de la Red
# =============================================
def visualize_network(G, states, title, step, max_steps):
    """Visualiza la red con estados kármicos codificados por colores"""
    plt.figure(figsize=(12, 8))
    
    # Extraer estados
    n = len(G.nodes)
    I, A, V, P, E, w = states[:n], states[n:2*n], states[2*n:3*n], states[3*n:4*n], states[4*n:5*n], states[5*n:]
    
    # Calcular veneno dominante para cada nodo
    dominant = np.argmax(np.array([I, A, V, P, E]), axis=0)
    colors = ['blue', 'green', 'red', 'purple', 'cyan']
    node_colors = [colors[d] for d in dominant]
    
    # Tamaño proporcional a la sabiduría
    node_sizes = 100 + 500 * np.array(w)
    
    # Posiciones de los nodos
    pos = nx.spring_layout(G, seed=42)
    
    # Dibujar red
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Ignorance'),
        Patch(facecolor='green', label='Attachment'),
        Patch(facecolor='red', label='Aversion'),
        Patch(facecolor='purple', label='Pride'),
        Patch(facecolor='cyan', label='Envy')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"{title} (Paso {step}/{max_steps})")
    plt.axis('off')
    plt.savefig(f"../figs/network_step_{step:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()

# =============================================
# 4. Simulación Principal
# =============================================
def main():
    print("\n" + "="*70)
    print("COLLECTIVE KARMA NETWORK SIMULATION")
    print("="*70)
    
    # Parámetros de simulación
    n_nodes = 50
    simulation_time = 20
    time_steps = 100
    t = np.linspace(0, simulation_time, time_steps)
    
    # Generar red social
    G = generate_social_network(n_nodes, 'scale-free')
    
    # Parámetros del modelo
    params = {
        # Parámetros individuales
        'alpha_I': 0.25, 'alpha_A': 0.20, 'alpha_V': 0.30, 
        'alpha_P': 0.35, 'alpha_E': 0.28,
        'beta_IP': 0.8, 'beta_IE': 0.3,
        'beta_AP': 0.6, 'beta_AE': 0.4,
        'beta_VE': 0.5, 'beta_VI': 0.4,
        'beta_PA': 0.1, 'beta_PI': 0.05,
        'beta_EV': 0.3, 'beta_EP': 0.2,
        'gamma_I': 0.4, 'gamma_A': 0.35, 'gamma_V': 0.45,
        'gamma_P': 0.5, 'gamma_E': 0.38,
        
        # Parámetros de red
        'kappa_I': 0.05, 'kappa_A': 0.07, 'kappa_V': 0.06,
        'kappa_P': 0.08, 'kappa_E': 0.09,
        'epsilon': 0.1,  # Tasa de crecimiento de sabiduría intrínseca
        'mu': 0.05,     # Tasa de degradación de sabiduría
        'delta': 0.3    # Fuerza de difusión de sabiduría
    }
    
    # Condiciones iniciales aleatorias
    np.random.seed(42)
    y0 = np.zeros(6 * n_nodes)
    
    # Venenos mentales iniciales (distribución desigual)
    y0[:n_nodes] = np.random.uniform(0.1, 0.8, n_nodes)  # I
    y0[n_nodes:2*n_nodes] = np.random.uniform(0.1, 0.6, n_nodes)  # A
    y0[2*n_nodes:3*n_nodes] = np.random.uniform(0.1, 0.7, n_nodes)  # V
    y0[3*n_nodes:4*n_nodes] = np.random.uniform(0.1, 0.9, n_nodes)  # P
    y0[4*n_nodes:5*n_nodes] = np.random.uniform(0.1, 0.75, n_nodes)  # E
    
    # Sabiduría inicial (algunos nodos "iluminados")
    wisdom_init = np.random.uniform(0.1, 0.3, n_nodes)
    wisdom_init[np.random.choice(n_nodes, 3, replace=False)] = 0.8  # Semillas de sabiduría
    y0[5*n_nodes:] = wisdom_init
    
    # Resolver el sistema de ecuaciones
    print("Simulando dinámica de red...")
    solution = odeint(collective_karma_model, y0, t, args=(G, params))
    
    # Visualizar estados iniciales y finales
    print("Generando visualizaciones...")
    visualize_network(G, solution[0], "Estado Inicial de la Red", 0, len(t))
    visualize_network(G, solution[-1], "Estado Final de la Red", len(t)-1, len(t))
    
    # Crear animación de la evolución
    print("Creando animación...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Configurar colores
    colors = ['blue', 'green', 'red', 'purple', 'cyan']
    
    def update(frame):
        ax.clear()
        n = n_nodes
        states = solution[frame]
        I, A, V, P, E, w = states[:n], states[n:2*n], states[2*n:3*n], states[3*n:4*n], states[4*n:5*n], states[5*n:]
        
        # Veneno dominante
        dominant = np.argmax(np.array([I, A, V, P, E]), axis=0)
        node_colors = [colors[d] for d in dominant]
        
        # Tamaño proporcional a la sabiduría
        node_sizes = 100 + 500 * np.array(w)
        
        # Posiciones
        pos = nx.spring_layout(G, seed=42)
        
        # Dibujar red
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)
        
        # Título
        ax.set_title(f"Evolución del Karma Colectivo (t = {t[frame]:.1f})")
        ax.axis('off')
        
        # Barra de progreso
        progress = (frame + 1) / len(t)
        ax.annotate(f"Progreso: {progress*100:.0f}%", 
                   xy=(0.02, 0.95), xycoords='figure fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Crear animación
    ani = FuncAnimation(fig, update, frames=len(t), interval=50, repeat=False)
    ani.save('../figs/network_evolution.mp4', writer='ffmpeg', fps=10, dpi=150)
    
    # Análisis de la difusión de sabiduría
    analyze_wisdom_diffusion(G, solution, t)
    
    # Análisis de la estructura de la red
    analyze_network_structure(G, solution)
    
    print("Simulación completada!")

# =============================================
# 5. Análisis de Difusión de Sabiduría
# =============================================
def analyze_wisdom_diffusion(G, solution, t):
    """Analiza cómo se difunde la sabiduría en la red"""
    n = len(G.nodes)
    wisdom = solution[:, 5*n:6*n]
    
    # Sabiduría promedio a lo largo del tiempo
    avg_wisdom = np.mean(wisdom, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, avg_wisdom, 'g-', linewidth=2)
    plt.title('Evolución de la Sabiduría Promedio en la Red')
    plt.xlabel('Tiempo')
    plt.ylabel('Sabiduría Promedio')
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/avg_wisdom_evolution.png', dpi=150)
    
    # Relación entre sabiduría y grado de conexión
    degrees = [d for n, d in G.degree()]
    final_wisdom = wisdom[-1]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(degrees, final_wisdom, alpha=0.6, c='teal')
    plt.title('Relación entre Grado de Conexión y Sabiduría Final')
    plt.xlabel('Grado del Nodo')
    plt.ylabel('Sabiduría Final')
    
    # Ajuste lineal
    coef = np.polyfit(degrees, final_wisdom, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(degrees, poly1d_fn(degrees), 'r--')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/degree_wisdom_correlation.png', dpi=150)
    
    # Calcular umbral crítico para sabiduría colectiva
    w_critical = 0.55  # Valor hipotético basado en el modelo
    enlightened_nodes = final_wisdom > w_critical
    print(f"\nANÁLISIS DE SABIDURÍA COLECTIVA:")
    print(f"- Sabiduría promedio inicial: {np.mean(wisdom[0]):.3f}")
    print(f"- Sabiduría promedio final: {np.mean(final_wisdom):.3f}")
    print(f"- Nodos con w > {w_critical}: {np.sum(enlightened_nodes)}/{n} ({np.mean(enlightened_nodes)*100:.1f}%)")
    
    # Visualizar distribución de sabiduría final
    plt.figure(figsize=(12, 6))
    plt.hist(final_wisdom, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=w_critical, color='red', linestyle='--', label=f'Umbral crítico ({w_critical})')
    plt.title('Distribución de Sabiduría Final')
    plt.xlabel('Nivel de Sabiduría')
    plt.ylabel('Número de Nodos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/wisdom_distribution.png', dpi=150)

# =============================================
# 6. Análisis de Estructura de Red
# =============================================
def analyze_network_structure(G, solution):
    """Analiza la relación entre estructura de red y estados kármicos"""
    n = len(G.nodes)
    final_states = solution[-1]
    I, A, V, P, E = final_states[:n], final_states[n:2*n], final_states[2*n:3*n], final_states[3*n:4*n], final_states[4*n:5*n]
    
    # Calcular centralidades
    degree_centrality = np.array(list(nx.degree_centrality(G).values()))
    betweenness = np.array(list(nx.betweenness_centrality(G).values()))
    closeness = np.array(list(nx.closeness_centrality(G).values()))
    
    # Dominant poison
    dominant = np.argmax(np.array([I, A, V, P, E]), axis=0)
    
    # Visualizar correlaciones
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    centralities = [degree_centrality, betweenness, closeness]
    centrality_names = ['Grado', 'Intermediación', 'Cercanía']
    poisons = [I, A, V, P, E]
    poison_names = ['Ignorancia', 'Apego', 'Aversión', 'Orgullo', 'Envidia']
    
    for i, poison in enumerate(poisons):
        for j, centrality in enumerate(centralities):
            ax = axs[i, j]
            ax.scatter(centrality, poison, alpha=0.6)
            ax.set_title(f'{poison_names[i]} vs {centrality_names[j]}')
            ax.set_xlabel(centrality_names[j])
            ax.set_ylabel(poison_names[i])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figs/centrality_poison_correlations.png', dpi=150)
    
    # Distribución de venenos dominantes por centralidad
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(poison_names):
        mask = (dominant == i)
        plt.scatter(degree_centrality[mask], betweenness[mask], 
                   label=name, alpha=0.7, s=100)
    
    plt.title('Veneno Dominante por Centralidad de Red')
    plt.xlabel('Centralidad de Grado')
    plt.ylabel('Centralidad de Intermediación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figs/dominant_poison_by_centrality.png', dpi=150)

if __name__ == "__main__":
    main()
