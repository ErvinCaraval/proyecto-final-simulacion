import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix

# Create a simulated Jacobian matrix with 5-diagonal structure
n = 105  # Matrix size (number of unknowns in 5x50 mesh)
J = lil_matrix((n, n))

# Fill with 5-diagonal pattern
for i in range(n):
    J[i, i] = 1  # Main diagonal
    if i > 0:
        J[i, i-1] = 1  # Lower diagonal
    if i < n-1:
        J[i, i+1] = 1  # Upper diagonal
    if i >= 21:  # Approx mesh width
        J[i, i-21] = 1
    if i < n-21:
        J[i, i+21] = 1

# Convert to dense for visualization
J_dense = J.toarray()

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Visualize sparse structure
ax.spy(J_dense, markersize=2, color='black')
ax.set_title('Jacobian Matrix Sparse Structure\n(5-Diagonal Pattern)', fontsize=14, weight='bold')
ax.set_xlabel('Column (Variable Index)', fontsize=12)
ax.set_ylabel('Row (Equation Index)', fontsize=12)

# Explanatory text
textstr = f'Dimension: {n}x{n}\nNon-zero elements: {J.nnz}\nDensity: {J.nnz/(n*n)*100:.2f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('figuras/estructura_jacobiano.png', dpi=150, bbox_inches='tight')
print("Jacobian structure generated: figuras/estructura_jacobiano.png")
plt.close()
