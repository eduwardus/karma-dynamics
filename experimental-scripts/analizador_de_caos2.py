# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 20:46:40 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages

# Diccionario de colores para cada tipo de comportamiento
behavior_colors = {
    'chaotic': '#d62728',     # Rojo intenso
    'cyclic_simple': '#1f77b4', # Azul estándar
    'cyclic_complex': '#ff7f0e', # Naranja
    'extinct': '#2ca02c',      # Verde
    'other': '#7f7f7f'         # Gris
}

def analizar_trayectoria(archivo, tipo_comportamiento):
    """Analiza una trayectoria y genera visualizaciones"""
    # Cargar datos
    trayectoria = np.load(archivo)
    n_puntos = len(trayectoria)
    
    # Calcular dt basado en t_max y el número de puntos
    t_max = 500  # Debe coincidir con el valor usado en la simulación
    dt = t_max / n_puntos
    
    # Crear array de tiempo correctamente dimensionado
    tiempo = np.linspace(0, t_max, n_puntos)
    
    # Configurar PDF de salida
    nombre_base = os.path.splitext(archivo)[0]
    pdf_path = f"{nombre_base}_analysis.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Título general
        fig = plt.figure(figsize=(10, 2))
        fig.suptitle(f'Análisis de Trayectoria {tipo_comportamiento.capitalize()}', fontsize=16)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # ===== 1. Series temporales =====
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f'Series Temporales - {os.path.basename(archivo)}', fontsize=16)
        
        componentes = ['Avidyā', 'Rāga', 'Dveṣa']
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(3):
            axs[i].plot(tiempo, trayectoria[:, i], color=colores[i])
            axs[i].set_ylabel(f'{componentes[i]} (I)', fontsize=12)
            axs[i].grid(alpha=0.3)
            
            # Resaltar regiones importantes
            if trayectoria[:, i].max() > 0.5:
                axs[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            # Detectar máximos locales
            max_indices, _ = find_peaks(trayectoria[:, i], height=0.1)
            axs[i].plot(tiempo[max_indices], trayectoria[max_indices, i], 'ro', markersize=3)
            
            # Destacar comportamiento final
            if trayectoria[-1, i] < 0.01:
                axs[i].axvline(x=tiempo[-1], color='g', linestyle='-', alpha=0.3, linewidth=3)
        
        axs[-1].set_xlabel('Tiempo', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 2. Espacio de Fase 3D =====
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Segmentar trayectoria para mejor visualización
        segment_size = min(2000, n_puntos)
        start_idx = max(0, n_puntos // 4)  # Empezar después del transitorio
        
        # Crear gradiente de color para la trayectoria
        puntos = np.arange(segment_size)
        
        # Graficar trayectoria segmentada
        for j in range(segment_size - 1):
            ax.plot(trayectoria[start_idx+j:start_idx+j+2, 0], 
                    trayectoria[start_idx+j:start_idx+j+2, 1], 
                    trayectoria[start_idx+j:start_idx+j+2, 2],
                    color=behavior_colors[tipo_comportamiento], alpha=0.8, linewidth=0.8)
        
        # Destacar puntos importantes
        ax.scatter(trayectoria[start_idx, 0], trayectoria[start_idx, 1], trayectoria[start_idx, 2],
                  c='red', s=50, label='Inicio', zorder=5)
        ax.scatter(trayectoria[start_idx+segment_size-1, 0], 
                  trayectoria[start_idx+segment_size-1, 1], 
                  trayectoria[start_idx+segment_size-1, 2],
                  c='blue', s=50, label='Fin', zorder=5)
        
        # Identificar puntos extremos
        max_idx = np.argmax(np.linalg.norm(trayectoria, axis=1))
        ax.scatter(trayectoria[max_idx, 0], trayectoria[max_idx, 1], trayectoria[max_idx, 2],
                  c='green', s=70, marker='*', label='Máximo', zorder=5)
        
        ax.set_xlabel('Avidyā', fontsize=12)
        ax.set_ylabel('Rāga', fontsize=12)
        ax.set_zlabel('Dveṣa', fontsize=12)
        ax.set_title(f'Espacio de Fase 3D - Comportamiento: {tipo_comportamiento}', fontsize=14)
        ax.legend()
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        pdf.savefig(fig)
        plt.close()
        
        # ===== 3. Comportamiento final =====
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comportamiento Asintótico', fontsize=16)
        
        # Último 20% de la simulación
        start_idx = int(n_puntos * 0.8)
        last_segment = trayectoria[start_idx:]
        tiempo_final = tiempo[start_idx:]
        
        for i in range(3):
            axs[i].plot(tiempo_final, last_segment[:, i], color=colores[i])
            axs[i].set_title(componentes[i])
            axs[i].set_xlabel('Tiempo')
            axs[i].set_ylabel('Intensidad')
            axs[i].grid(alpha=0.3)
            
            # Calcular tendencia
            if len(tiempo_final) > 10:
                coeffs = np.polyfit(tiempo_final, last_segment[:, i], 1)
                trend = np.poly1d(coeffs)(tiempo_final)
                axs[i].plot(tiempo_final, trend, 'k--', label='Tendencia')
                
                # Determinar si tiende a cero
                slope = coeffs[0]
                if slope < -1e-5:
                    axs[i].text(0.5, 0.9, 'Extinguiéndose', 
                               transform=axs[i].transAxes, ha='center', color='red')
                elif abs(slope) < 1e-5 and last_segment[-1, i] < 0.01:
                    axs[i].text(0.5, 0.9, 'Extinto', 
                               transform=axs[i].transAxes, ha='center', color='green')
                elif abs(slope) < 1e-5:
                    axs[i].text(0.5, 0.9, 'Estable', 
                               transform=axs[i].transAxes, ha='center', color='blue')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 4. Análisis Espectral (FFT) =====
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Análisis Espectral', fontsize=16)
        
        for i in range(3):
            # Calcular FFT
            yf = fft(trayectoria[:, i])
            xf = fftfreq(n_puntos, dt)[:n_puntos//2]
            
            # Graficar espectro de potencia
            axs[i].plot(xf, 2.0/n_puntos * np.abs(yf[0:n_puntos//2]), 
                       color=colores[i])
            axs[i].set_ylabel(f'Potencia {componentes[i]}', fontsize=12)
            axs[i].set_yscale('log')
            axs[i].grid(alpha=0.3)
            
            # Identificar frecuencias dominantes
            if len(xf) > 0:  # Asegurar que hay datos
                idx_max = np.argmax(2.0/n_puntos * np.abs(yf[0:n_puntos//2]))
                freq_dom = xf[idx_max]
                axs[i].axvline(x=freq_dom, color='r', linestyle='--', 
                              label=f'f = {freq_dom:.4f} Hz')
                axs[i].legend()
        
        axs[-1].set_xlabel('Frecuencia (Hz)', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 5. Distribución de Probabilidad =====
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Distribución de Probabilidad', fontsize=16)
        
        for i in range(3):
            # Crear estimación de densidad de kernel
            kde = gaussian_kde(trayectoria[:, i])
            x_vals = np.linspace(trayectoria[:, i].min(), trayectoria[:, i].max(), 200)
            densidad = kde(x_vals)
            
            # Graficar histograma y KDE
            axs[i].hist(trayectoria[:, i], bins=50, density=True, alpha=0.5, 
                       color=colores[i], label='Histograma')
            axs[i].plot(x_vals, densidad, 'k-', linewidth=2, label='KDE')
            
            # Marcar estadísticas importantes
            media = np.mean(trayectoria[:, i])
            mediana = np.median(trayectoria[:, i])
            axs[i].axvline(media, color='r', linestyle='--', label=f'Media: {media:.3f}')
            axs[i].axvline(mediana, color='g', linestyle='-.', label=f'Mediana: {mediana:.3f}')
            
            axs[i].set_title(componentes[i], fontsize=14)
            axs[i].set_xlabel('Intensidad', fontsize=12)
            axs[i].set_ylabel('Densidad', fontsize=12)
            axs[i].legend()
            axs[i].grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 6. Resumen Estadístico =====
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f'Resumen Estadístico - {os.path.basename(archivo)}', fontsize=16)
        
        # Crear tabla de estadísticas
        stats_data = []
        for i, comp in enumerate(componentes):
            comp_data = trayectoria[:, i]
            stats_data.append([
                comp,
                np.min(comp_data),
                np.max(comp_data),
                np.mean(comp_data),
                np.median(comp_data),
                np.std(comp_data),
                np.quantile(comp_data, 0.25),
                np.quantile(comp_data, 0.75),
                np.sum(comp_data > 0.5) / len(comp_data)  # Porcentaje sobre 0.5
            ])
        
        # Crear gráfico de tabla
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        col_labels = ['Componente', 'Mín', 'Máx', 'Media', 'Mediana', 'Desv Est', 
                     'Q1', 'Q3', '% > 0.5']
        table = ax.table(cellText=stats_data,
                        colLabels=col_labels,
                        loc='center',
                        cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        pdf.savefig(fig)
        plt.close()
    
    print(f"Análisis completo. Resultados guardados en: {pdf_path}")
    return pdf_path

def analizar_todas_trayectorias():
    """Busca y analiza todas las trayectorias en el directorio actual"""
    # Buscar archivos de trayectorias
    patrones = {
        'chaotic': "chaotic_trajectory_sim_*.npy",
        'cyclic_simple': "cyclic_simple_trajectory_sim_*.npy",
        'cyclic_complex': "cyclic_complex_trajectory_sim_*.npy",
        'extinct': "extinct_trajectory_sim_*.npy"
    }
    
    for tipo, patron in patrones.items():
        archivos_trayectorias = glob.glob(patron)
        
        if not archivos_trayectorias:
            print(f"No se encontraron trayectorias de tipo {tipo}.")
            continue
        
        print(f"\nEncontradas {len(archivos_trayectorias)} trayectorias de tipo {tipo}:")
        for archivo in archivos_trayectorias:
            print(f" - {archivo}")
        
        # Analizar cada trayectoria
        for archivo in archivos_trayectorias:
            print(f"\nAnalizando trayectoria: {archivo}")
            analizar_trayectoria(archivo, tipo)
    
    print("\n¡Análisis de todas las trayectorias completado!")

# ===== EJECUCIÓN PRINCIPAL =====
if __name__ == "__main__":
    print("===== ANALIZADOR DE TRAYECTORIAS KÁRMICAS =====")
    analizar_todas_trayectorias()