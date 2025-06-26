# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 01:03:32 2025

@author: eggra
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys
import threading

# =============================================
# 0. Configuración inicial
# =============================================
print("="*70)
print("MODELO DE TRANSFORMACIÓN KÁRMICA VAJRAYANA")
print("="*70)

# Crear directorio para figuras
if not os.path.exists('../figs'):
    os.makedirs('../figs')

# =============================================
# 1. Funciones de práctica
# =============================================
def constant_practice(t):
    """Práctica constante"""
    return 0.8

def retreat_practice(t):
    """Práctica intensiva con decaimiento"""
    return 1.0 if t < 20 else 0.6 * np.exp(-0.05*(t-20))

def periodic_practice(t):
    """Práctica periódica (diaria/semanal)"""
    return 0.7 + 0.3 * np.sin(0.5 * t)

# =============================================
# 2. Modelo Vajrayana de Transformación Kármica
# =============================================
def vajrayana_model(t, y, params):
    """
    I: Ignorancia
    A: Apego
    V: Aversión
    W: Sabiduría
    D: Sabiduría Dakini
    T: Energía Tummo
    """
    I, A, V, W, D, T = y
    P = params['practice'](t)  # Intensidad de práctica en tiempo t
    
    # Desempaquetar parámetros
    α_I, α_A, α_V, β_IA, β_AV, β_VI, γ_I, γ_A, γ_V, w = params['base_params']
    δ = params['nonduality']  # Conciencia no dual
    γ_trans = params['transformation_rate']  # Tasa de transformación
    κ = params['dakini_response']  # Respuesta dakini
    τ_max = params['tummo_capacity']  # Capacidad máxima de tummo
    
    # Dinámica de la Sabiduría Dakini (responde a la ignorancia)
    dDdt = κ * I - 0.1 * D
    
    # Energía Tummo (alimentada por la práctica)
    dTdt = P * (τ_max - T) - 0.05 * T
    
    # Transformación kármica (innovación Vajrayana)
    trans_I = γ_trans * D * T * I**2 / (1 + I)
    trans_A = γ_trans * D * T * A**2 / (1 + A)
    trans_V = γ_trans * D * T * V**2 / (1 + V)
    
    # Conciencia no dual reduce distinciones
    nondual = 1 - np.exp(-δ * W)
    
    # Ecuaciones principales con transformación
    dIdt = (α_I*I + β_IA*A*V - γ_I*w*I) * nondual - trans_I
    dAdt = (α_A*A + β_AV*V*I - γ_A*w*A) * nondual - trans_A
    dVdt = (α_V*V + β_VI*I*A - γ_V*w*V) * nondual - trans_V
    
    # La sabiduría se acumula a través de la transformación
    dWdt = 0.7*trans_I + 0.5*trans_A + 0.6*trans_V - 0.1*W
    
    return [dIdt, dAdt, dVdt, dWdt, dDdt, dTdt]

# =============================================
# 3. Clase para mostrar progreso en tiempo real
# =============================================
class ProgressIndicator:
    def __init__(self, message, max_time=100):
        self.message = message
        self.max_time = max_time
        self.start_time = time.perf_counter()
        self.running = True
        self.progress_thread = threading.Thread(target=self.show_progress)
        self.progress_thread.daemon = True
        self.progress_thread.start()
    
    def show_progress(self):
        """Muestra una animación de progreso con tiempo transcurrido"""
        animation = ["◐", "◓", "◑", "◒"]
        i = 0
        while self.running:
            elapsed = time.perf_counter() - self.start_time
            sys.stdout.write(f"\r{animation[i % 4]} {self.message} | Tiempo: {elapsed:.1f}s")
            sys.stdout.flush()
            time.sleep(0.2)
            i += 1
        # Limpiar la línea al finalizar
        sys.stdout.write("\r" + " " * 80 + "\r")
    
    def stop(self):
        """Detiene el indicador de progreso"""
        self.running = False
        self.progress_thread.join()

# =============================================
# 4. Simulación de Transformación Kármica
# =============================================
def simulate_vajrayana():
    """Simula el proceso completo de transformación kármica"""
    print("\nSimulando transformación kármica...")
    progress = ProgressIndicator("Calculando transformación")
    start_time = time.perf_counter()
    
    try:
        # Parámetros base (similares al modelo Theravada)
        base_params = (0.3, 0.25, 0.35, 0.6, 0.7, 0.5, 0.4, 0.35, 0.45, 0.3)
        
        # Parámetros específicos de Vajrayana
        params = {
            'base_params': base_params,
            'nonduality': 0.5,          # Fuerza de conciencia no dual
            'transformation_rate': 0.8,  # Tasa de purificación kármica
            'dakini_response': 0.4,      # Capacidad de respuesta dakini
            'tummo_capacity': 1.0,       # Capacidad máxima de energía tummo
            'practice': periodic_practice  # Régimen de práctica
        }
        
        # Estado inicial
        y0 = [0.6, 0.5, 0.4, 0.1, 0.3, 0.2]  # [I, A, V, W, D, T]
        
        # Intervalo de tiempo
        t_span = [0, 100]
        t_eval = np.linspace(0, 100, 2000)
        
        # Resolver el sistema
        sol = solve_ivp(
            lambda t, y: vajrayana_model(t, y, params),
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA'
        )
        
        # Extraer soluciones
        I, A, V, W, D, T = sol.y
        
        # Visualización
        plt.figure(figsize=(15, 12))
        
        # Raíces negativas y sabiduría
        plt.subplot(3, 1, 1)
        plt.plot(t_eval, I, 'r-', label='Ignorancia')
        plt.plot(t_eval, A, 'g-', label='Apego')
        plt.plot(t_eval, V, 'b-', label='Aversión')
        plt.plot(t_eval, W, 'm-', linewidth=2, label='Sabiduría')
        plt.title('Transformación de Raíces Kármicas en Sabiduría')
        plt.ylabel('Intensidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Práctica y energía
        plt.subplot(3, 1, 2)
        practice = params['practice'](t_eval)
        plt.plot(t_eval, practice, 'k-', label='Intensidad de Práctica')
        plt.plot(t_eval, T, 'c-', label='Energía Tummo')
        plt.plot(t_eval, D, 'y-', label='Sabiduría Dakini')
        plt.title('Práctica y Energía de Transformación')
        plt.ylabel('Nivel')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tasas de transformación
        plt.subplot(3, 1, 3)
        trans_I = params['transformation_rate'] * D * T * I**2 / (1 + I)
        trans_A = params['transformation_rate'] * D * T * A**2 / (1 + A)
        trans_V = params['transformation_rate'] * D * T * V**2 / (1 + V)
        plt.plot(t_eval, trans_I, 'r--', label='Transformación Ignorancia')
        plt.plot(t_eval, trans_A, 'g--', label='Transformación Apego')
        plt.plot(t_eval, trans_V, 'b--', label='Transformación Aversión')
        plt.title('Tasas de Transformación Kármica')
        plt.xlabel('Tiempo')
        plt.ylabel('Tasa de Transformación')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figs/vajrayana_transformation.png', dpi=150)
        plt.show()
        
        # Análisis del estado final
        final_negative = I[-1] + A[-1] + V[-1]
        final_wisdom = W[-1]
        print(f"\nKarma Negativo Final: {final_negative:.3f}")
        print(f"Sabiduría Final: {final_wisdom:.3f}")
        print(f"Ratio de Transformación: {final_wisdom/(final_negative+1e-6):.1%}")
        
        elapsed = time.perf_counter() - start_time
        print(f"Simulación de transformación completada en {elapsed:.2f} segundos")
        return elapsed
        
    finally:
        progress.stop()

# =============================================
# 5. Prueba de Resiliencia con Indicador de Progreso
# =============================================
def test_vajrayana_resilience():
    """Prueba la capacidad de recuperación ante perturbaciones"""
    print("\nProbando resiliencia a perturbaciones...")
    progress = ProgressIndicator("Calculando resiliencia")
    start_time = time.perf_counter()
    
    try:
        # Parámetros base
        base_params = (0.3, 0.25, 0.35, 0.6, 0.7, 0.5, 0.4, 0.35, 0.45, 0.3)
        
        # Parámetros específicos de Vajrayana
        params = {
            'base_params': base_params,
            'nonduality': 0.6,
            'transformation_rate': 0.9,
            'dakini_response': 0.5,
            'tummo_capacity': 1.2,
            'practice': periodic_practice
        }
        
        # Estado inicial
        y0 = [0.4, 0.3, 0.3, 0.2, 0.4, 0.3]  # [I, A, V, W, D, T]
        
        # Intervalo de tiempo
        t_span = [0, 100]
        t_eval = np.linspace(0, 100, 2000)
        
        # Simulación original
        sol_orig = solve_ivp(
            lambda t, y: vajrayana_model(t, y, params),
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA'
        )
        
        # Aplicar perturbación en t=40
        perturb_index = 800  # 800/2000*100 = 40
        perturbed_y0 = sol_orig.y[:, perturb_index].copy()
        perturbed_y0[0] += 0.4  # Aumentar ignorancia
        perturbed_y0[1] += -0.3  # Disminuir apego
        perturbed_y0[2] += 0.5  # Aumentar aversión
        
        # Simulación perturbada
        pert_t_eval = t_eval[t_eval >= 40]
        sol_pert = solve_ivp(
            lambda t, y: vajrayana_model(t, y, params),
            [40, 100],
            perturbed_y0,
            t_eval=pert_t_eval,
            method='LSODA'
        )
        
        # Visualización
        plt.figure(figsize=(12, 8))
        
        # Trayectoria de sabiduría
        plt.plot(t_eval, sol_orig.y[3], 'm-', label='Sabiduría Original')
        plt.plot(sol_pert.t, sol_pert.y[3], 'm--', label='Sabiduría Perturbada')
        
        # Suma de karma negativo
        orig_negative = sol_orig.y[0] + sol_orig.y[1] + sol_orig.y[2]
        pert_negative = sol_pert.y[0] + sol_pert.y[1] + sol_pert.y[2]
        plt.plot(t_eval, orig_negative, 'k-', label='Karma Negativo Original')
        plt.plot(sol_pert.t, pert_negative, 'k--', label='Karma Negativo Perturbado')
        
        plt.axvline(x=40, color='r', linestyle=':', label='Evento de Perturbación')
        plt.title('Resiliencia Vajrayana: Transformación del Karma Negativo')
        plt.xlabel('Tiempo')
        plt.ylabel('Nivel')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('../figs/vajrayana_resilience.png', dpi=150)
        plt.show()
        
        # Métricas de recuperación
        recovery_time = None
        for i, t_val in enumerate(sol_pert.t):
            if pert_negative[i] <= orig_negative[perturb_index] * 1.1:  # Dentro del 10% del valor original
                recovery_time = t_val - 40
                break
        
        if recovery_time is not None:
            print(f"\nTiempo de Recuperación: {recovery_time:.1f} unidades")
        else:
            print("\nEl sistema no se recuperó completamente durante la simulación")
        
        elapsed = time.perf_counter() - start_time
        print(f"Prueba de resiliencia completada en {elapsed:.2f} segundos")
        return elapsed
        
    finally:
        progress.stop()

# =============================================
# 6. Función Principal
# =============================================
def main():
    print("\nIniciando simulación Vajrayana...")
    total_start = time.perf_counter()
    
    # 1. Simulación de transformación kármica
    trans_time = simulate_vajrayana()
    
    # 2. Prueba de resiliencia
    res_time = test_vajrayana_resilience()
    
    # Resumen de tiempos
    total_time = time.perf_counter() - total_start
    print("\nRESUMEN FINAL:")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print(f" - Simulación de transformación: {trans_time:.2f} segundos")
    print(f" - Prueba de resiliencia: {res_time:.2f} segundos")
    print("\nSimulación completada exitosamente!")

if __name__ == "__main__":
    main()