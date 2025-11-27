import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Domain Parameters
NX, NY = 50, 5
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 10, 20
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 3, 5

# Create figure
fig, ax = plt.subplots(figsize=(14, 4))

# Draw domain
ax.add_patch(patches.Rectangle((0, 0), NX, NY, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3))

# Draw obstacles
ax.add_patch(patches.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                                VIGA_INF_X_MAX - VIGA_INF_X_MIN, 
                                VIGA_INF_Y_MAX - VIGA_INF_Y_MIN,
                                linewidth=2, edgecolor='black', facecolor='gray'))

ax.add_patch(patches.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                                VIGA_SUP_X_MAX - VIGA_SUP_X_MIN, 
                                VIGA_SUP_Y_MAX - VIGA_SUP_Y_MIN,
                                linewidth=2, edgecolor='black', facecolor='gray'))

# Flow arrows
for y in np.linspace(0.5, NY-0.5, 5):
    ax.arrow(-2, y, 1.5, 0, head_width=0.3, head_length=0.5, fc='red', ec='red', linewidth=2)

# Boundary conditions
ax.text(-5, NY/2, 'Inlet\n$v_x = 1.0$ m/s\n$v_y = 0$', fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.text(NX+5, NY/2, 'Outlet\nZero Gradient', fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(NX/2, -1.5, 'Bottom Wall: No Slip ($v_x = v_y = 0$)', fontsize=10, ha='center')
ax.text(NX/2, NY+1.5, 'Top Wall: No Slip ($v_x = v_y = 0$)', fontsize=10, ha='center')

# Obstacle labels
ax.text((VIGA_INF_X_MIN + VIGA_INF_X_MAX)/2, (VIGA_INF_Y_MIN + VIGA_INF_Y_MAX)/2, 
        'Beam 1', fontsize=12, ha='center', va='center', color='white', weight='bold')
ax.text((VIGA_SUP_X_MIN + VIGA_SUP_X_MAX)/2, (VIGA_SUP_Y_MIN + VIGA_SUP_Y_MAX)/2, 
        'Beam 2', fontsize=12, ha='center', va='center', color='white', weight='bold')

# Axes and dimensions
ax.set_xlabel('x (units)', fontsize=12)
ax.set_ylabel('y (units)', fontsize=12)
ax.set_title('Computational Domain - Channel with Obstacles', fontsize=14, weight='bold')

# Dimensions
ax.annotate('', xy=(NX, -2.5), xytext=(0, -2.5),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='blue'))
ax.text(NX/2, -3, f'$L_x = {NX}$ units', fontsize=11, ha='center', color='blue')

ax.annotate('', xy=(-3, NY), xytext=(-3, 0),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='blue'))
ax.text(-4.5, NY/2, f'$L_y = {NY}$ units', fontsize=11, ha='center', va='center', rotation=90, color='blue')

ax.set_xlim(-8, NX+8)
ax.set_ylim(-4, NY+3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figuras/dominio_computacional.png', dpi=150, bbox_inches='tight')
print("Computational domain generated: figuras/dominio_computacional.png")
plt.close()
