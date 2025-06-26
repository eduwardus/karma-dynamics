# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 21:25:59 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

class KarmaModel:
    def __init__(self, params):
        # Parámetros del modelo
        self.params = params
        self.N = params['N']  # Número de agentes
        self.dt = params['dt']  # Paso temporal
        self.total_time = params['total_time']  # Tiempo total de simulación
        self.n_steps = int(self.total_time / self.dt)  # Número de pasos
        
        # Inicialización de variables de estado
        self.initialize_variables()
        
        # Configuración de retardos
        self.setup_delays()
        
        # Configuración de red
        self.setup_network()
    
    def initialize_variables(self):
        # Parámetros
        p = self.params
        
        # Variables principales
        self.I = np.full(self.N, p['I0'])
        self.A = np.full(self.N, p['A0'])
        self.V = np.full(self.N, p['V0'])
        self.P = np.full(self.N, p['P0'])
        self.E = np.full(self.N, p['E0'])
        
        # Sabiduría global y componentes
        self.w = np.full(self.N, p['w0'])
        self.w_conciencia = np.full(self.N, p['w_conciencia0'])
        self.w_compasion = np.full(self.N, p['w_compasion0'])
        self.w_ego = np.full(self.N, p['w_ego0'])
        
        # Historial de trayectorias
        self.history = {
            'I': np.zeros((self.n_steps + 1, self.N)),
            'A': np.zeros((self.n_steps + 1, self.N)),
            'V': np.zeros((self.n_steps + 1, self.N)),
            'P': np.zeros((self.n_steps + 1, self.N)),
            'E': np.zeros((self.n_steps + 1, self.N)),
            'w': np.zeros((self.n_steps + 1, self.N)),
            'w_conciencia': np.zeros((self.n_steps + 1, self.N)),
            'w_compasion': np.zeros((self.n_steps + 1, self.N)),
            'w_ego': np.zeros((self.n_steps + 1, self.N))
        }
        
        # Guardar estado inicial
        self.save_state(0)
    
    def setup_delays(self):
        """Configura buffers para manejar retardos temporales"""
        p = self.params
        self.delay_steps = int(p['tau'] / self.dt) if p['tau'] > 0 else 0
        
        # Buffers circulares para retardos
        self.P_delay_buffer = np.tile(self.P, (self.delay_steps + 1, 1))
        self.E_delay_buffer = np.tile(self.E, (self.delay_steps + 1, 1))
        self.buffer_index = 0
    
    def setup_network(self):
        """Configura la matriz de conectividad entre agentes"""
        p = self.params
        if p['network_type'] == 'complete':
            # Red completa (todos conectados con todos)
            C = np.ones((self.N, self.N)) / (self.N - 1)
            np.fill_diagonal(C, 0)
            self.C = csr_matrix(C)  # Usamos matrices dispersas para eficiencia
        elif p['network_type'] == 'random':
            # Red aleatoria
            C = np.random.rand(self.N, self.N)
            C = C / C.sum(axis=1, keepdims=True)
            np.fill_diagonal(C, 0)
            self.C = csr_matrix(C)
        elif p['network_type'] == 'ring':
            # Anillo unidireccional
            C = np.zeros((self.N, self.N))
            for i in range(self.N):
                C[i, (i + 1) % self.N] = 1.0
            self.C = csr_matrix(C)
        else:
            raise ValueError("Tipo de red no válido")
    
    def get_delayed_values(self):
        """Obtiene valores retardados de P y E"""
        if self.delay_steps == 0:
            return self.P, self.E
        
        # Índice para valores retardados (buffer circular)
        delay_index = (self.buffer_index - self.delay_steps) % (self.delay_steps + 1)
        return self.P_delay_buffer[delay_index], self.E_delay_buffer[delay_index]
    
    def update_delay_buffers(self):
        """Actualiza los buffers de valores retardados"""
        self.P_delay_buffer[self.buffer_index] = self.P
        self.E_delay_buffer[self.buffer_index] = self.E
        self.buffer_index = (self.buffer_index + 1) % (self.delay_steps + 1)
    
    def save_state(self, step):
        """Guarda el estado actual en el historial"""
        self.history['I'][step] = self.I
        self.history['A'][step] = self.A
        self.history['V'][step] = self.V
        self.history['P'][step] = self.P
        self.history['E'][step] = self.E
        self.history['w'][step] = self.w
        self.history['w_conciencia'][step] = self.w_conciencia
        self.history['w_compasion'][step] = self.w_compasion
        self.history['w_ego'][step] = self.w_ego
    
    def step(self, step):
        """Realiza un paso de integración"""
        p = self.params
        dt = self.dt
        
        # Obtener valores retardados
        P_delayed, E_delayed = self.get_delayed_values()
        
        # Calcular sabiduría actual
        if p['use_multidimensional_wisdom']:
            self.w = self.w_conciencia + self.w_compasion - self.w_ego
        
        # Generar ruidos estocásticos
        dW_I = np.random.normal(0, np.sqrt(dt), self.N)
        dW_A = np.random.normal(0, np.sqrt(dt), self.N)
        dW_V = np.random.normal(0, np.sqrt(dt), self.N)
        
        # --- Ecuaciones estocásticas (SDEs) ---
        # 1. Estado interno (I_i)
        dI = (p['alpha_I'] * self.I + 
              p['beta_IP'] * np.tanh(self.I / (self.P + 1e-8)) - 
              p['gamma_I'] * self.w * self.I) * dt + p['sigma_I'] * self.I * dW_I
        
        # 2. Acciones altruistas (A_i)
        dA = (p['alpha_A'] * self.A + 
              p['beta_AP'] * self.A * P_delayed - 
              p['gamma_A'] * self.w * self.A) * dt + p['sigma_A'] * self.A * dW_A
        
        # 3. Estado de virtud (V_i)
        dV = (p['alpha_V'] * self.V + 
              p['beta_VE'] * self.V * E_delayed - 
              p['gamma_V'] * self.w * self.V) * dt + p['sigma_V'] * self.V * dW_V
        
        # --- Ecuaciones deterministas (ODEs) ---
        # 4. Recursos personales (P_i)
        coupling_P = self.C.dot(self.P) - self.P * np.array(self.C.sum(axis=1)).flatten()
        dP = (p['alpha_P'] * self.P + coupling_P - p['gamma_P'] * self.w * self.P) * dt
        
        # 5. Recursos ambientales (E_i)
        dE = (p['alpha_E'] * self.E + 
              p['beta_EV'] * self.V * self.E / (1 + self.V) - 
              p['gamma_E'] * self.w * self.E) * dt
        
        # 6. Componentes de sabiduría
        # Conciencia
        dw_conciencia = (p['eta_M'] * p['meditacion'](step * dt) - 
                         p['mu_C'] * self.w_conciencia) * dt
        
        # Compasión
        dw_compasion = (p['eta_A'] * p['altruismo'](step * dt) - 
                        p['mu_K'] * self.w_compasion) * dt
        
        # Ego
        dw_ego = (p['alpha_ego'] * self.P - 
                  p['gamma_E_wisdom'] * self.w_conciencia) * dt
        
        # Sabiduría global (si no se usa la descomposición multidimensional)
        if not p['use_multidimensional_wisdom']:
            coupling_w = self.C.dot(self.w) - self.w * np.array(self.C.sum(axis=1)).flatten()
            dw = (p['eta1'] * p['meditacion'](step * dt) + 
                  p['eta2'] * coupling_w - 
                  p['mu'] * self.w) * dt
        else:
            dw = np.zeros(self.N)
        
        # Actualizar variables
        self.I += dI
        self.A += dA
        self.V += dV
        self.P += dP
        self.E += dE
        
        # Actualizar componentes de sabiduría
        self.w_conciencia += dw_conciencia
        self.w_compasion += dw_compasion
        self.w_ego += dw_ego
        
        # Actualizar sabiduría global (si no se usa descomposición)
        if not p['use_multidimensional_wisdom']:
            self.w += dw
        
        # Actualizar buffers de retardos
        self.update_delay_buffers()
    
    def simulate(self):
        """Ejecuta la simulación completa"""
        for step in range(1, self.n_steps + 1):
            self.step(step)
            self.save_state(step)
    
    def plot_results(self, agent_idx=0):
        """Visualiza resultados para un agente específico"""
        time = np.linspace(0, self.total_time, self.n_steps + 1)
        
        plt.figure(figsize=(15, 12))
        
        # Variables principales
        plt.subplot(3, 2, 1)
        plt.plot(time, self.history['I'][:, agent_idx], label='Estado Interno (I)')
        plt.plot(time, self.history['A'][:, agent_idx], label='Acciones Altruistas (A)')
        plt.plot(time, self.history['V'][:, agent_idx], label='Virtud (V)')
        plt.title('Variables Principales')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Recursos
        plt.subplot(3, 2, 2)
        plt.plot(time, self.history['P'][:, agent_idx], label='Recursos Personales (P)')
        plt.plot(time, self.history['E'][:, agent_idx], label='Recursos Ambientales (E)')
        plt.title('Recursos')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Sabiduría y componentes
        plt.subplot(3, 2, 3)
        plt.plot(time, self.history['w'][:, agent_idx], label='Sabiduría Global (w)', linewidth=2)
        plt.title('Sabiduría Global')
        plt.xlabel('Tiempo')
        plt.ylabel('w')
        plt.grid(True)
        
        plt.subplot(3, 2, 4)
        plt.plot(time, self.history['w_conciencia'][:, agent_idx], label='Conciencia')
        plt.plot(time, self.history['w_compasion'][:, agent_idx], label='Compasión')
        plt.plot(time, self.history['w_ego'][:, agent_idx], label='Ego')
        plt.title('Componentes de Sabiduría')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Evolución del karma
        plt.subplot(3, 2, 5)
        karma = (self.history['I'][:, agent_idx] + 
                 self.history['A'][:, agent_idx] + 
                 self.history['V'][:, agent_idx])
        plt.plot(time, karma, 'k-', linewidth=2)
        plt.title('Evolución del Karma')
        plt.xlabel('Tiempo')
        plt.ylabel('Karma Integral')
        plt.grid(True)
        
        # Espacio de fases
        plt.subplot(3, 2, 6)
        plt.plot(self.history['I'][:, agent_idx], 
                 self.history['A'][:, agent_idx], 
                 'b-', alpha=0.7)
        plt.scatter(self.history['I'][0, agent_idx], 
                    self.history['A'][0, agent_idx], 
                    c='green', s=100, label='Inicio')
        plt.scatter(self.history['I'][-1, agent_idx], 
                    self.history['A'][-1, agent_idx], 
                    c='red', s=100, label='Final')
        plt.title('Espacio de Fases: I vs A')
        plt.xlabel('Estado Interno (I)')
        plt.ylabel('Acciones Altruistas (A)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Configuración de parámetros (¡personalizable!)
params = {
    # Parámetros generales
    'N': 10,                   # Número de agentes
    'dt': 0.01,                # Paso temporal
    'total_time': 50,          # Tiempo total de simulación
    'tau': 0.5,                # Retardo temporal
    'network_type': 'complete', # Tipo de red ('complete', 'random', 'ring')
    
    # Parámetros de las ecuaciones
    'alpha_I': 0.05, 'beta_IP': 0.1, 'gamma_I': 0.02, 'sigma_I': 0.05,
    'alpha_A': 0.04, 'beta_AP': 0.08, 'gamma_A': 0.01, 'sigma_A': 0.04,
    'alpha_V': 0.03, 'beta_VE': 0.07, 'gamma_V': 0.015, 'sigma_V': 0.03,
    'alpha_P': 0.02, 'gamma_P': 0.01,
    'alpha_E': 0.01, 'beta_EV': 0.06, 'gamma_E': 0.005,
    
    # Parámetros de sabiduría
    'eta1': 0.05, 'eta2': 0.03, 'mu': 0.01,
    'eta_M': 0.1, 'mu_C': 0.02,
    'eta_A': 0.08, 'mu_K': 0.015,
    'alpha_ego': 0.02, 'gamma_E_wisdom': 0.01,
    
    # Funciones de entrada (pueden personalizarse)
    'meditacion': lambda t: 0.5 + 0.3 * np.sin(t),
    'altruismo': lambda t: 0.6 + 0.2 * np.cos(0.5 * t),
    
    # Bandera para sabiduría multidimensional
    'use_multidimensional_wisdom': True,
    
    # Valores iniciales (pueden ser arrays de tamaño N o escalares)
    'I0': 0.5, 'A0': 0.4, 'V0': 0.6, 
    'P0': 0.8, 'E0': 1.0, 'w0': 0.1,
    'w_conciencia0': 0.2, 'w_compasion0': 0.3, 'w_ego0': 0.4
}

# Crear y ejecutar simulación
model = KarmaModel(params)
model.simulate()

# Visualizar resultados para el primer agente
model.plot_results(agent_idx=0)