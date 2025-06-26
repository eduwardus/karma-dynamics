# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 20:12:24 2025

@author: eggra
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq
from scipy.stats import gaussian_kde
import os
import glob
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

def analizar_trayectoria_caotica(archivo):
    """Analiza una trayectoria caótica y genera visualizaciones"""
    # Cargar datos
    trayectoria = np.load(archivo)
    tiempo = np.arange(0, len(trayectoria) * dt, dt)
    
    # Configurar PDF de salida
    nombre_base = os.path.splitext(archivo)[0]
    pdf_path = f"{nombre_base}_analysis.pdf"
    
    with PdfPages(pdf_path) as pdf:
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
            max_indices = np.where((trayectoria[1:-1, i] > trayectoria[0:-2, i]) & 
                                  (trayectoria[1:-1, i] > trayectoria[2:, i]))[0] + 1
            axs[i].plot(tiempo[max_indices], trayectoria[max_indices, i], 'ro', markersize=3)
        
        axs[-1].set_xlabel('Tiempo', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 2. Espacio de Fase 3D =====
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Segmentar trayectoria para mejor visualización
        segment_size = min(2000, len(trayectoria))
        start_idx = max(0, len(trayectoria) // 4)  # Empezar después del transitorio
        
        # Crear gradiente de color para la trayectoria
        puntos = np.arange(segment_size)
        colores_3d = plt.cm.viridis(puntos / segment_size)
        
        # Graficar trayectoria segmentada
        for j in range(segment_size - 1):
            ax.plot(trayectoria[start_idx+j:start_idx+j+2, 0], 
                    trayectoria[start_idx+j:start_idx+j+2, 1], 
                    trayectoria[start_idx+j:start_idx+j+2, 2],
                    color=colores_3d[j], alpha=0.8, linewidth=0.8)
        
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
        ax.set_title('Espacio de Fase 3D', fontsize=14)
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
        
        # ===== 3. Proyecciones 2D =====
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Proyecciones del Espacio de Fase', fontsize=16)
        
        # Proyección Avidyā-Rāga
        axs[0].scatter(trayectoria[start_idx:start_idx+segment_size, 0], 
                      trayectoria[start_idx:start_idx+segment_size, 1],
                      c=puntos, cmap='viridis', s=5, alpha=0.7)
        axs[0].set_xlabel('Avidyā')
        axs[0].set_ylabel('Rāga')
        axs[0].grid(alpha=0.3)
        
        # Proyección Avidyā-Dveṣa
        axs[1].scatter(trayectoria[start_idx:start_idx+segment_size, 0], 
                      trayectoria[start_idx:start_idx+segment_size, 2],
                      c=puntos, cmap='viridis', s=5, alpha=0.7)
        axs[1].set_xlabel('Avidyā')
        axs[1].set_ylabel('Dveṣa')
        axs[1].grid(alpha=0.3)
        
        # Proyección Rāga-Dveṣa
        axs[2].scatter(trayectoria[start_idx:start_idx+segment_size, 1], 
                      trayectoria[start_idx:start_idx+segment_size, 2],
                      c=puntos, cmap='viridis', s=5, alpha=0.7)
        axs[2].set_xlabel('Rāga')
        axs[2].set_ylabel('Dveṣa')
        axs[2].grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close()
        
        # ===== 4. Análisis Espectral (FFT) =====
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Análisis Espectral', fontsize=16)
        
        for i in range(3):
            # Calcular FFT
            yf = fft(trayectoria[:, i])
            xf = fftfreq(len(trayectoria), dt)[:len(trayectoria)//2]
            
            # Graficar espectro de potencia
            axs[i].plot(xf, 2.0/len(trayectoria) * np.abs(yf[0:len(trayectoria)//2]), 
                       color=colores[i])
            axs[i].set_ylabel(f'Potencia {componentes[i]}', fontsize=12)
            axs[i].set_yscale('log')
            axs[i].grid(alpha=0.3)
            
            # Identificar frecuencias dominantes
            idx_max = np.argmax(2.0/len(trayectoria) * np.abs(yf[0:len(trayectoria)//2]))
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
        
        # ===== 6. Atractor de Lorenz =====
        # Solo para trayectorias con estructura similar a Lorenz
        if np.max(trayectoria) > 0.3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Segmentar y escalar para mejor visualización
            segmento = trayectoria[:min(5000, len(trayectoria))]
            segmento = segmento - np.min(segmento, axis=0)
            segmento = segmento / np.max(segmento, axis=0)
            
            # Crear gradiente de color
            puntos = np.arange(len(segmento))
            colores_attr = plt.cm.plasma(puntos / len(segmento))
            
            # Graficar trayectoria
            for j in range(len(segmento) - 1):
                ax.plot(segmento[j:j+2, 0], segmento[j:j+2, 1], segmento[j:j+2, 2],
                        color=colores_attr[j], alpha=0.7, linewidth=0.7)
            
            ax.set_xlabel('Avidyā')
            ax.set_ylabel('Rāga')
            ax.set_zlabel('Dveṣa')
            ax.set_title('Representación de Atractor', fontsize=14)
            ax.grid(True)
            
            pdf.savefig(fig)
            plt.close()
        
        # ===== 7. Resumen Estadístico =====
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

def analizar_todas_trayectorias_caoticas():
    """Busca y analiza todas las trayectorias caóticas en el directorio actual"""
    # Buscar archivos de trayectorias caóticas
    archivos_trayectorias = glob.glob("chaotic_trajectory_sim_*.npy")
    
    if not archivos_trayectorias:
        print("No se encontraron archivos de trayectorias caóticas.")
        return
    
    print(f"Encontradas {len(archivos_trayectorias)} trayectorias caóticas:")
    for archivo in archivos_trayectorias:
        print(f" - {archivo}")
    
    # Analizar cada trayectoria
    for archivo in archivos_trayectorias:
        print(f"\nAnalizando trayectoria: {archivo}")
        analizar_trayectoria_caotica(archivo)
    
    print("\n¡Análisis de todas las trayectorias completado!")

# ===== PARÁMETROS DEL SISTEMA (deben coincidir con la simulación) =====
dt = 0.02  # Paso temporal (debe ser el mismo que en la simulación)

# ===== EJECUCIÓN PRINCIPAL =====
if __name__ == "__main__":
    print("===== ANALIZADOR DE TRAYECTORIAS CAÓTICAS KÁRMICAS =====")
    analizar_todas_trayectorias_caoticas()