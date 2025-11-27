"""
Streamline Visualization
========================
Generates streamline plots to visualize flow patterns around obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import from existing code
sys.path.insert(0, os.path.dirname(__file__))
from campo_velocidadesV4 import (
    FlujoNewtonRaphson, NY, NX, V0_INITIAL, VY_TEST,
    VIGA_INF_X_MIN, VIGA_INF_X_MAX, VIGA_INF_Y_MIN, VIGA_INF_Y_MAX,
    VIGA_SUP_X_MIN, VIGA_SUP_X_MAX, VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX,
    interpolate_cubic_natural_manual
)

# Import bicubic interpolation from V4.1
import importlib.util
spec = importlib.util.spec_from_file_location("campo_v4_1", "campo_velocidadesV4.1.py")
campo_v4_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campo_v4_1)
interpolate_bicubic_natural = campo_v4_1.interpolate_bicubic_natural

plt.style.use('dark_background')

def calculate_vy_field(Vx_matrix):
    """
    Estimates the Vy velocity field using the continuity equation.
    dVx/dx + dVy/dy = 0  ->  dVy/dy = -dVx/dx
    """
    Vy_matrix = np.full_like(Vx_matrix, VY_TEST)
    
    # Apply continuity equation (simplified)
    for j in range(1, NY - 1):
        for i in range(1, NX - 1):
            # Derivative of Vx with respect to x (central differences)
            dVx_dx = (Vx_matrix[j, i+1] - Vx_matrix[j, i-1]) / 2.0
            
            # Estimate Vy using continuity
            Vy_matrix[j, i] = VY_TEST - 0.5 * dVx_dx
    
    # Force zero at obstacles
    Vy_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
    Vy_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
    
    return Vy_matrix

def create_streamlines_basic(Vx, Vy, title, filename):
    """Creates basic streamline visualization without interpolation."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create grid for streamplot
    x = np.arange(0, NX)
    y = np.arange(0, NY)
    X, Y = np.meshgrid(x, y)
    
    # Total velocity heatmap
    velocity_total = np.sqrt(Vx**2 + Vy**2)
    im = ax.imshow(velocity_total, cmap='plasma', origin='lower', 
                   extent=[0, NX, 0, NY], alpha=0.6, vmin=0, vmax=V0_INITIAL)
    
    # Streamlines
    ax.streamplot(X[0, :], Y[:, 0], Vx, Vy, 
                  color='cyan', linewidth=1.5, density=1.5, 
                  arrowsize=1.5, arrowstyle='->')
    
    # Draw obstacles
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
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Total Velocity (m/s)', fontsize=11)
    
    plt.tight_layout()
    path = os.path.join('analisis_avanzado', filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def create_streamlines_interpolated(Vx_low, Vy_low, interp_method, title, filename):
    """Creates streamline visualization using spline interpolation."""
    
    # Interpolate Vx
    if interp_method == 'manual':
        Vx_high = interpolate_cubic_natural_manual(Vx_low)
    else:
        Vx_high = interpolate_bicubic_natural(Vx_low)
    
    # Interpolate Vy (same method)
    if interp_method == 'manual':
        Vy_high = interpolate_cubic_natural_manual(Vy_low)
    else:
        Vy_high = interpolate_bicubic_natural(Vy_low)
    
    NY_high, NX_high = Vx_high.shape
    
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Create high resolution grid
    x_high = np.linspace(0, NX-1, NX_high)
    y_high = np.linspace(0, NY-1, NY_high)
    X_high, Y_high = np.meshgrid(x_high, y_high)
    
    # Heatmap
    velocity_total = np.sqrt(Vx_high**2 + Vy_high**2)
    im = ax.imshow(velocity_total, cmap='inferno', origin='lower', 
                   extent=[0, NX, 0, NY], alpha=0.7, vmin=0, vmax=V0_INITIAL)
    
    # Streamlines with higher density
    ax.streamplot(x_high, y_high, Vx_high, Vy_high, 
                  color='white', linewidth=1.2, density=2.0, 
                  arrowsize=1.2, arrowstyle='->', zorder=5)
    
    # Obstacles
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
    ax.set_xlabel('X Position', fontsize=13)
    ax.set_ylabel('Y Position', fontsize=13)
    ax.set_title(title, fontsize=15, weight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Total Velocity (m/s)', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join('analisis_avanzado', filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

def generate_all_streamlines():
    """Generates all streamline visualizations."""
    print("=" * 80)
    print("STREAMLINE GENERATION")
    print("=" * 80)
    print("\nRunning simulation...\n")
    
    # Run simulation
    solver = FlujoNewtonRaphson()
    result = solver.solve(linear_solver_method='conjugate-gradient', theoretical_analysis=False)
    
    if not result['converged']:
        print("Error: Simulation did not converge.")
        return
    
    Vx = result['solution']
    
    print("Simulation completed")
    print(f"Iterations: {result['iterations']}")
    print(f"Time: {result['time']:.2f}s\n")
    
    # Calculate Vy field
    print("Calculating Vy velocity field...\n")
    Vy = calculate_vy_field(Vx)
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    # 1. Basic Streamlines (no interpolation)
    create_streamlines_basic(
        Vx, Vy,
        'Streamlines - Original Resolution (50x5)',
        'streamlines_basico.png'
    )
    
    # 2. Streamlines with manual interpolation (V4)
    create_streamlines_interpolated(
        Vx, Vy, 'manual',
        'Streamlines - 1D Natural Cubic Spline Interpolation',
        'streamlines_v4_manual.png'
    )
    
    # 3. Streamlines with bicubic interpolation (V4.1)
    create_streamlines_interpolated(
        Vx, Vy, 'bicubica',
        'Streamlines - Bicubic Spline Interpolation',
        'streamlines_v4_1_bicubica.png'
    )
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("\nFiles saved in: analisis_avanzado/")
    print("  - streamlines_basico.png")
    print("  - streamlines_v4_manual.png")
    print("  - streamlines_v4_1_bicubica.png\n")

if __name__ == '__main__':
    generate_all_streamlines()
