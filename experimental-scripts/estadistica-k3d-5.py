# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 13:11:24 2025

@author: eggra
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def analyze_chaos_absence(results_file):
    """Analiza estadísticamente la ausencia de comportamiento caótico"""
    # Cargar resultados
    df = pd.read_csv(results_file)
    total_simulations = len(df)
    
    # Contar dinámicas caóticas
    chaos_mask = df['dynamics_type'].str.contains('Caos', na=False)
    n_chaos = chaos_mask.sum()
    p_observed = n_chaos / total_simulations
    
    print("=== ANÁLISIS ESTADÍSTICO DE AUSENCIA DE CAOS ===")
    print(f"Total de simulaciones: {total_simulations}")
    print(f"Comportamientos caóticos detectados: {n_chaos}")
    print(f"Proporción observada: {p_observed:.6f} ({p_observed*100:.4f}%)")
    
    # 1. Regla de los tres para estimar el límite superior
    upper_bound = 3 / total_simulations if n_chaos == 0 else None
    if upper_bound:
        print(f"\nCon 95% de confianza (Regla de los tres):")
        print(f"La verdadera proporción de caos es menor que {upper_bound:.6f} ({upper_bound*100:.4f}%)")
    
    # 2. Intervalo de confianza exacto (Clopper-Pearson)
    if n_chaos == 0:
        ci_lower = 0.0
        ci_upper = 1 - (0.05)**(1/total_simulations)  # Límite superior del 95% CI
    else:
        ci_lower, ci_upper = stats.beta.interval(0.95, n_chaos+1, total_simulations-n_chaos+1)
    
    print(f"\nIntervalo de confianza exacto del 95% (Clopper-Pearson):")
    print(f"Probabilidad mínima de caos: {ci_lower:.8f} ({ci_lower*100:.6f}%)")
    print(f"Probabilidad máxima de caos: {ci_upper:.8f} ({ci_upper*100:.6f}%)")
    
    # 3. Prueba de potencia estadística
    # ¿Qué tamaño de efecto podríamos haber detectado?
    detectable_effect = stats.binom.ppf(0.95, total_simulations, 0.0001) / total_simulations
    
    print(f"\nPotencia estadística del estudio:")
    print(f"Con {total_simulations} simulaciones, podemos detectar:")
    print(f"- Caos que ocurra en al menos {detectable_effect*100:.4f}% de los casos (con 95% de confianza)")
    
    # 4. Análisis de sensibilidad de los parámetros
    if n_chaos == 0:
        print("\nAnálisis de parámetros en simulaciones complejas:")
        complex_mask = df['dynamics_type'].str.contains('complejo', case=False, na=False)
        complex_df = df[complex_mask]
        
        if len(complex_df) > 0:
            print("Parámetros en sistemas periódicos complejos (más cercanos al caos):")
            for param in ['gamma', 'alpha_aa', 'kappa_aa', 'delta_a', 'omega_a']:
                if param in complex_df.columns:
                    mean_val = complex_df[param].mean()
                    std_val = complex_df[param].std()
                    print(f"{param}: {mean_val:.3f} ± {std_val:.3f}")
        else:
            print("No se encontraron sistemas periódicos complejos")
    
    # 5. Visualización de la distribución de Lyapunov
    plt.figure(figsize=(10, 6))
    plt.hist(df['lyapunov_exp'], bins=100, alpha=0.7, color='blue')
    plt.axvline(x=0, color='red', linestyle='--', label='Umbral de Caos (λ>0)')
    plt.xlabel('Exponente de Lyapunov')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Exponentes de Lyapunov en las Simulaciones')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('lyapunov_distribution.png', dpi=150)
    plt.close()
    
    print("\nConclusión científica:")
    if n_chaos == 0:
        print(f"Con un 95% de confianza, la probabilidad de comportamiento caótico en este modelo")
        print(f"es menor que {min(ci_upper, upper_bound):.6f} ({min(ci_upper, upper_bound)*100:.4f}%).")
        print("El modelo muestra una fuerte tendencia a comportamientos periódicos estables.")
    else:
        print(f"Se detectó comportamiento caótico en {n_chaos} de {total_simulations} simulaciones.")
        print(f"La probabilidad estimada de caos es de {p_observed:.4f} (IC95%: [{ci_lower:.4f}, {ci_upper:.4f}]).")
    
    return {
        'total_simulations': total_simulations,
        'n_chaos': n_chaos,
        'p_observed': p_observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'detectable_effect': detectable_effect
    }

# Ejecutar análisis
if __name__ == "__main__":
    results = analyze_chaos_absence("karmic_chaos_extreme_scan.csv")
    
    # Guardar resultados estadísticos
    with open('chaos_absence_analysis.txt', 'w') as f:
        f.write("=== CONCLUSIÓN CIENTÍFICA ===\n")
        if results['n_chaos'] == 0:
            f.write(f"Tras {results['total_simulations']} simulaciones con parámetros extremos:\n")
            f.write("NO se detectó ningún caso de comportamiento caótico.\n\n")
            f.write(f"Con 95% de confianza, la probabilidad real de caos es menor que {results['ci_upper']:.6f}\n")
            f.write(f"Esto significa que en menos de 1 de cada {int(1/results['ci_upper'])} configuraciones\n")
            f.write("el modelo exhibiría comportamiento caótico.\n\n")
            f.write("El modelo muestra estabilidad estructural y tiende a comportamientos periódicos\n")
            f.write("incluso bajo condiciones paramétricas extremas.")
        else:
            f.write(f"Se detectó comportamiento caótico en {results['n_chaos']} de {results['total_simulations']} simulaciones.\n")
            f.write(f"Probabilidad estimada: {results['p_observed']:.4f} (IC95%: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}])\n\n")
            f.write("El modelo puede exhibir caos bajo configuraciones específicas,\n")
            f.write("pero este comportamiento es estadísticamente raro.")
    
    print("\nAnálisis completo guardado en 'chaos_absence_analysis.txt' y 'lyapunov_distribution.png'")