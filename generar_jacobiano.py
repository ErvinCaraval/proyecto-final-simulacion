import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix

# Crear una matriz Jacobiana simulada con estructura de 5 diagonales
n = 105  # Tamaño de la matriz (número de incógnitas en la malla 5x50)
J = lil_matrix((n, n))

# Llenar con patrón de 5 diagonales (diagonal principal + 4 diagonales adyacentes)
for i in range(n):
    J[i, i] = 1  # Diagonal principal
    if i > 0:
        J[i, i-1] = 1  # Diagonal inferior
    if i < n-1:
        J[i, i+1] = 1  # Diagonal superior
    if i >= 21:  # Aproximadamente el ancho de la malla
        J[i, i-21] = 1
    if i < n-21:
        J[i, i+21] = 1

# Convertir a formato denso para visualización
J_dense = J.toarray()

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))

# Visualizar la estructura dispersa
ax.spy(J_dense, markersize=2, color='black')
ax.set_title('Estructura Dispersa de la Matriz Jacobiana\n(Patrón de 5 Diagonales)', fontsize=14, weight='bold')
ax.set_xlabel('Columna (índice de variable)', fontsize=12)
ax.set_ylabel('Fila (índice de ecuación)', fontsize=12)

# Agregar texto explicativo
textstr = f'Dimensión: {n}×{n}\nElementos no nulos: {J.nnz}\nDensidad: {J.nnz/(n*n)*100:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('figuras/estructura_jacobiano.png', dpi=150, bbox_inches='tight')
print("✅ Estructura del Jacobiano generada: figuras/estructura_jacobiano.png")
plt.close()
