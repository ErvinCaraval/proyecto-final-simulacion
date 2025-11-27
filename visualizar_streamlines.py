"""
Visualización de Líneas de Corriente (Streamlines)
===================================================
Este script genera gráficos con líneas de corriente para visualizar
el patrón de flujo alrededor de los obstáculos.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Importar del código existente
sys.path.insert(0, os.path.dirname(__file__))
from campo_velocidadesV4 import (
    FlujoNewtonRaphson, NY, NX, V0_INITIAL, VY_TEST,
    VIGA_INF_X_MIN, VIGA_INF_X_MAX, VIGA_INF_Y_MIN, VIGA_INF_Y_MAX,
    VIGA_SUP_X_MIN, VIGA_SUP_X_MAX, VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX,
    interpolate_cubic_natural_manual
)

# Importar interpolación bicúbica del archivo V4.1
import importlib.util
spec = importlib.util.spec_from_file_location("campo_v4_1", "campo_velocidadesV4.1.py")
campo_v4_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campo_v4_1)
interpolate_bicubic_natural = campo_v4_1.interpolate_bicubic_natural

plt.style.use('dark_background')

def calcular_campo_vy(Vx_matrix):
    """
    Estima el campo de velocidades Vy usando la ecuación de continuidad.
    ∂Vx/∂x + ∂Vy/∂y = 0  →  ∂Vy/∂y = -∂Vx/∂x
    """
    Vy_matrix = np.full_like(Vx_matrix, VY_TEST)
    
    # Aplicar ecuación de continuidad de forma simplificada
    for j in range(1, NY - 1):
        for i in range(1, NX - 1):
            # Derivada de Vx respecto a x (diferencias centradas)
            dVx_dx = (Vx_matrix[j, i+1] - Vx_matrix[j, i-1]) / 2.0
            
            # Estimar Vy usando continuidad (simplificado)
            # En realidad necesitaríamos integrar, pero usamos aproximación
            Vy_matrix[j, i] = VY_TEST - 0.5 * dVx_dx
    
    # Forzar a cero en obstáculos
    Vy_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
    Vy_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
    
    return Vy_matrix

def crear_streamlines_basico(Vx, Vy, titulo, nombre_archivo):
    """Crea visualización básica con streamlines sin interpolación."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Crear grilla para streamplot
    x = np.arange(0, NX)
    y = np.arange(0, NY)
    X, Y = np.meshgrid(x, y)
    
    # Mapa de calor de velocidad total
    velocidad_total = np.sqrt(Vx**2 + Vy**2)
    im = ax.imshow(velocidad_total, cmap='plasma', origin='lower', 
                   extent=[0, NX, 0, NY], alpha=0.6, vmin=0, vmax=V0_INITIAL)
    
    # Líneas de corriente
    ax.streamplot(X[0, :], Y[:, 0], Vx, Vy, 
                  color='cyan', linewidth=1.5, density=1.5, 
                  arrowsize=1.5, arrowstyle='->')
    
    # Dibujar obstáculos
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, 
                               color='#FF0055', alpha=0.9, zorder=10))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, 
                               color='#FF0055', alpha=0.9, zorder=10))
    
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_xlabel('Posición X', fontsize=12)
    ax.set_ylabel('Posición Y', fontsize=12)
    ax.set_title(titulo, fontsize=14, weight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocidad Total (m/s)', fontsize=11)
    
    plt.tight_layout()
    ruta = os.path.join('analisis_avanzado', nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Guardado: {ruta}")

def crear_streamlines_interpolado(Vx_low, Vy_low, metodo_interp, titulo, nombre_archivo):
    """Crea visualización con streamlines usando interpolación spline."""
    
    # Interpolar Vx
    if metodo_interp == 'manual':
        Vx_high = interpolate_cubic_natural_manual(Vx_low)
    else:
        Vx_high = interpolate_bicubic_natural(Vx_low)
    
    # Interpolar Vy (mismo método)
    if metodo_interp == 'manual':
        Vy_high = interpolate_cubic_natural_manual(Vy_low)
    else:
        Vy_high = interpolate_bicubic_natural(Vy_low)
    
    NY_high, NX_high = Vx_high.shape
    
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Crear grilla de alta resolución
    x_high = np.linspace(0, NX-1, NX_high)
    y_high = np.linspace(0, NY-1, NY_high)
    X_high, Y_high = np.meshgrid(x_high, y_high)
    
    # Mapa de calor
    velocidad_total = np.sqrt(Vx_high**2 + Vy_high**2)
    im = ax.imshow(velocidad_total, cmap='inferno', origin='lower', 
                   extent=[0, NX, 0, NY], alpha=0.7, vmin=0, vmax=V0_INITIAL)
    
    # Streamlines con mayor densidad
    ax.streamplot(x_high, y_high, Vx_high, Vy_high, 
                  color='white', linewidth=1.2, density=2.0, 
                  arrowsize=1.2, arrowstyle='->', zorder=5)
    
    # Obstáculos
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, 
                               color='#00FFFF', alpha=0.3, 
                               edgecolor='cyan', linewidth=2, zorder=10))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, 
                               color='#00FFFF', alpha=0.3, 
                               edgecolor='cyan', linewidth=2, zorder=10))
    
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_xlabel('Posición X', fontsize=13)
    ax.set_ylabel('Posición Y', fontsize=13)
    ax.set_title(titulo, fontsize=15, weight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Velocidad Total (m/s)', fontsize=12)
    
    plt.tight_layout()
    ruta = os.path.join('analisis_avanzado', nombre_archivo)
    plt.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Guardado: {ruta}")

def generar_todas_streamlines():
    """Genera todas las visualizaciones de streamlines."""
    print("=" * 80)
    print("GENERACIÓN DE LÍNEAS DE CORRIENTE (STREAMLINES)")
    print("=" * 80)
    print("\n🌊 Ejecutando simulación...\n")
    
    # Ejecutar simulación
    solver = FlujoNewtonRaphson()
    resultado = solver.solve(linear_solver_method='conjugate-gradient', analisis_teorico=False)
    
    if not resultado['converged']:
        print("❌ Error: La simulación no convergió.")
        return
    
    Vx = resultado['solution']
    
    print("✓ Simulación completada")
    print(f"✓ Iteraciones: {resultado['iterations']}")
    print(f"✓ Tiempo: {resultado['time']:.2f}s\n")
    
    # Calcular campo Vy
    print("📐 Calculando campo de velocidades Vy...\n")
    Vy = calcular_campo_vy(Vx)
    
    # Generar visualizaciones
    print("🎨 Generando visualizaciones...\n")
    
    # 1. Streamlines básicas (sin interpolación)
    crear_streamlines_basico(
        Vx, Vy,
        'Líneas de Corriente - Resolución Original (50×5)',
        'streamlines_basico.png'
    )
    
    # 2. Streamlines con interpolación manual (V4)
    crear_streamlines_interpolado(
        Vx, Vy, 'manual',
        'Líneas de Corriente - Interpolación Spline Cúbico Natural 1D',
        'streamlines_v4_manual.png'
    )
    
    # 3. Streamlines con interpolación bicúbica (V4.1)
    crear_streamlines_interpolado(
        Vx, Vy, 'bicubica',
        'Líneas de Corriente - Interpolación Spline Bicúbico',
        'streamlines_v4_1_bicubica.png'
    )
    
    print("\n" + "=" * 80)
    print("✅ TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
    print("=" * 80)
    print("\nArchivos guardados en: analisis_avanzado/")
    print("  - streamlines_basico.png")
    print("  - streamlines_v4_manual.png")
    print("  - streamlines_v4_1_bicubica.png\n")

if __name__ == '__main__':
    generar_todas_streamlines()
