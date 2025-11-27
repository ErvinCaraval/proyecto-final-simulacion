"""
Mesh Independence Analysis
==========================
Evaluates the sensitivity of results to mesh resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Import necessary components
sys.path.insert(0, os.path.dirname(__file__))
from scipy.sparse import lil_matrix, diags, identity
from numpy.linalg import norm

plt.style.use('dark_background')

# Base Global Parameters
VY_TEST = 0.1
V0_INITIAL = 1.0
TOLERANCE = 1e-8
MAX_ITER = 500
LINEAR_TOLERANCE = 1e-4
MAX_LINEAR_ITER = 5000

class ParametricSolver:
    """Solver that accepts variable mesh parameters."""
    
    def __init__(self, ny, nx):
        self.NY = ny
        self.NX = nx
        
        # Scale obstacles proportionally
        self.VIGA_INF_Y_MIN = 0
        self.VIGA_INF_Y_MAX = max(1, int(2 * ny / 5))
        self.VIGA_INF_X_MIN = int(20 * nx / 50)
        self.VIGA_INF_X_MAX = int(30 * nx / 50)
        
        self.VIGA_SUP_Y_MIN = ny - max(1, int(1 * ny / 5))
        self.VIGA_SUP_Y_MAX = ny
        self.VIGA_SUP_X_MIN = int(40 * nx / 50)
        self.VIGA_SUP_X_MAX = nx
        
        self._incognita_map = {}
        self._preparar_mapa_incognitas()
        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)
    
    def _es_incognita(self, i, j):
        if not (1 <= i <= self.NX - 2 and 1 <= j <= self.NY - 2): 
            return False
        if j == 1 and self.VIGA_INF_X_MIN <= i < self.VIGA_INF_X_MAX: 
            return False
        if j == 2 and i >= self.VIGA_INF_X_MIN: 
            return False
        return True
    
    def _preparar_mapa_incognitas(self):
        count = 0
        for j in range(self.NY):
            for i in range(self.NX):
                if self._es_incognita(i, j): 
                    self._incognita_map[(i, j)] = count
                    count += 1
    
    def _map_to_linear_index(self, i, j): 
        return self._incognita_map.get((i, j), None)
    
    def _inicializar_matriz_velocidades(self, v_init):
        V_matrix = np.full((self.NY, self.NX), v_init)
        V_matrix[self.NY - 1, :] = V0_INITIAL
        V_matrix[:, 0] = V0_INITIAL
        V_matrix[0, :] = 0.0
        V_matrix[:, self.NX - 1] = 0.0
        V_matrix[self.VIGA_INF_Y_MIN:self.VIGA_INF_Y_MAX, 
                 self.VIGA_INF_X_MIN:self.VIGA_INF_X_MAX] = 0.0
        V_matrix[self.VIGA_SUP_Y_MIN:self.VIGA_SUP_Y_MAX, 
                 self.VIGA_SUP_X_MIN:self.VIGA_SUP_X_MAX] = 0.0
        V_matrix[2, self.VIGA_INF_X_MIN:] = 0.0
        
        for j in range(self.NY):
            for i in range(self.NX):
                if self._es_incognita(i, j): 
                    V_matrix[j, i] = v_init * (j / (self.NY - 1))
        return V_matrix
    
    def ensamblar_sistema_newton(self, V_current):
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        rhs = np.zeros(self.N_INCÓGNITAS)
        m = 0
        for j in range(self.NY):
            for i in range(self.NX):
                if not self._es_incognita(i, j): continue
                V_c = V_current[j, i]
                V_r = V_current[j, i+1]
                V_l = V_current[j, i-1]
                V_u = V_current[j+1, i]
                V_d = V_current[j-1, i]
                
                rhs[m] = -(4*V_c - (V_r+V_l+V_u+V_d) + 
                          4*V_c*(V_r-V_l) + 4*VY_TEST*(V_u-V_d))
                J[m, m] = 4 + 4*V_r - 4*V_l
                
                n_r = self._map_to_linear_index(i+1, j)
                n_l = self._map_to_linear_index(i-1, j)
                n_u = self._map_to_linear_index(i, j+1)
                n_d = self._map_to_linear_index(i, j-1)
                
                if n_r is not None: J[m, n_r] = -1 + 4*V_c
                if n_l is not None: J[m, n_l] = -1 - 4*V_c
                if n_u is not None: J[m, n_u] = -1 + 4*VY_TEST
                if n_d is not None: J[m, n_d] = -1 - 4*VY_TEST
                m += 1
        return J.tocsr(), rhs
    
    def _conjugate_gradient(self, J, rhs):
        A = J.T @ J + 1e-8 * identity(self.N_INCÓGNITAS)
        b = J.T @ rhs
        x = np.zeros_like(rhs, dtype=float)
        r = b - A @ x
        p = r.copy()
        rs_old = float(r.T @ r)
        
        for i in range(MAX_LINEAR_ITER):
            Ap = A @ p
            denom = float(p.T @ Ap)
            if abs(denom) < 1e-20: break
            alpha = rs_old / denom
            x += alpha * p
            r -= alpha * Ap
            rs_new = float(r.T @ r)
            if np.sqrt(rs_new) < LINEAR_TOLERANCE: return x
            p = r + (rs_new / rs_old) * p
            p = p.astype(float)
            rs_old = rs_new
        return x
    
    def solve(self):
        V_matrix = self.V_k.copy()
        start_time = time.time()
        
        for k in range(1, MAX_ITER + 1):
            J, rhs = self.ensamblar_sistema_newton(V_matrix)
            Delta_V_vector = self._conjugate_gradient(J, rhs)
            
            if np.isnan(Delta_V_vector).any() or np.max(np.abs(Delta_V_vector)) > 100:
                return None
            
            V_new_matrix = V_matrix.copy()
            m = 0
            for j in range(self.NY):
                for i in range(self.NX):
                    if self._es_incognita(i, j): 
                        V_new_matrix[j, i] += 0.6 * Delta_V_vector[m]
                        m += 1
            
            V_matrix = np.clip(V_new_matrix, 0, V0_INITIAL)
            max_cambio = np.max(np.abs(Delta_V_vector))
            
            if max_cambio < TOLERANCE:
                return {
                    'solution': V_matrix,
                    'time': time.time() - start_time,
                    'iterations': k,
                    'converged': True
                }
        
        return {
            'solution': V_matrix,
            'time': time.time() - start_time,
            'iterations': MAX_ITER,
            'converged': False
        }

def analyze_mesh_independence():
    """Executes mesh independence analysis."""
    print("=" * 80)
    print("MESH INDEPENDENCE ANALYSIS")
    print("=" * 80)
    
    # Mesh configurations to test
    configurations = [
        (5, 50, "Coarse (Original)"),
        (10, 100, "Medium"),
        (15, 150, "Fine")
    ]
    
    results = []
    
    for ny, nx, name in configurations:
        print(f"\n{'=' * 80}")
        print(f"Testing mesh: {name} ({ny} x {nx} = {ny*nx} cells)")
        print(f"{'=' * 80}")
        
        solver = ParametricSolver(ny, nx)
        print(f"  Unknowns: {solver.N_INCÓGNITAS}")
        print(f"  Running simulation...")
        
        result = solver.solve()
        
        if result is None or not result['converged']:
            print(f"  Did not converge")
            results.append(None)
            continue
        
        V_sol = result['solution']
        
        # Key metrics
        v_max = np.max(V_sol)
        v_avg = np.mean(V_sol[V_sol > 0])
        
        # Monitor point (center of channel, before first obstacle)
        i_monitor = int(0.3 * nx)
        j_monitor = int(0.5 * ny)
        v_monitor = V_sol[j_monitor, i_monitor]
        
        info = {
            'name': name,
            'ny': ny,
            'nx': nx,
            'v_max': v_max,
            'v_avg': v_avg,
            'v_monitor': v_monitor,
            'iterations': result['iterations'],
            'time': result['time']
        }
        
        results.append(info)
        
        print(f"  Converged in {result['iterations']} iterations")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  V_max: {v_max:.6f}")
        print(f"  V_avg: {v_avg:.6f}")
        print(f"  V_monitor: {v_monitor:.6f}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("MESH CONVERGENCE REPORT")
    print("=" * 80)
    
    report = f"""
{'=' * 80}
MESH INDEPENDENCE ANALYSIS
{'=' * 80}

1. TESTED CONFIGURATIONS
"""
    
    for i, res in enumerate(results):
        if res is None:
            continue
        report += f"""
   Mesh {i+1}: {res['name']}
   - Resolution: {res['ny']} x {res['nx']} = {res['ny']*res['nx']} cells
   - Iterations: {res['iterations']}
   - Time: {res['time']:.2f}s
   - V_max: {res['v_max']:.6f} m/s
   - V_avg: {res['v_avg']:.6f} m/s
   - V_monitor: {res['v_monitor']:.6f} m/s
"""
    
    # Calculate differences
    if len([r for r in results if r is not None]) >= 2:
        report += "\n2. CONVERGENCE ANALYSIS\n"
        
        valid_res = [r for r in results if r is not None]
        
        for i in range(len(valid_res) - 1):
            r1 = valid_res[i]
            r2 = valid_res[i+1]
            
            diff_max = abs(r2['v_max'] - r1['v_max']) / r1['v_max'] * 100
            diff_avg = abs(r2['v_avg'] - r1['v_avg']) / r1['v_avg'] * 100
            diff_mon = abs(r2['v_monitor'] - r1['v_monitor']) / r1['v_monitor'] * 100
            
            report += f"""
   {r1['name']} -> {r2['name']}:
   - Change in V_max:      {diff_max:.2f}%
   - Change in V_avg:      {diff_avg:.2f}%
   - Change in V_monitor:  {diff_mon:.2f}%
"""
        
        # Conclusion
        last_diff = diff_mon
        if last_diff < 1.0:
            conclusion = "EXCELLENT: Difference < 1% -> Mesh independent solution"
        elif last_diff < 5.0:
            conclusion = "GOOD: Difference < 5% -> Acceptable convergence"
        else:
            conclusion = "WARNING: Difference > 5% -> Further refinement required"
        
        report += f"""
3. CONCLUSION

   {conclusion}
   
   Last observed difference: {last_diff:.2f}%
   
   RECOMMENDATION:
   {'Current mesh is sufficient for this academic analysis.' if last_diff < 5 else 'Finer mesh recommended for production results.'}

"""
    
    report += f"""
4. COMPUTATIONAL COST

   Observed scaling:
"""
    
    for res in [r for r in results if r is not None]:
        cost_per_cell = res['time'] / (res['ny'] * res['nx'])
        report += f"   - {res['name']}: {cost_per_cell*1000:.2f} ms/cell\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    print(report)
    
    # Save report
    report_path = os.path.join('analisis_avanzado', 'reporte_independencia_malla.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}\n")
    
    # Generate convergence plot
    if len([r for r in results if r is not None]) >= 2:
        generate_convergence_plot([r for r in results if r is not None])

def generate_convergence_plot(results):
    """Generates mesh convergence plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    cells = [r['ny'] * r['nx'] for r in results]
    v_max = [r['v_max'] for r in results]
    v_monitor = [r['v_monitor'] for r in results]
    names = [r['name'] for r in results]
    
    # Plot 1: Velocity Convergence
    ax1.plot(cells, v_max, 'o-', label='V_max', linewidth=2, markersize=8, color='cyan')
    ax1.plot(cells, v_monitor, 's-', label='V_monitor', linewidth=2, markersize=8, color='yellow')
    ax1.set_xlabel('Number of Cells', fontsize=12)
    ax1.set_ylabel('Velocity (m/s)', fontsize=12)
    ax1.set_title('Mesh Convergence', fontsize=14, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Computational Cost
    times = [r['time'] for r in results]
    ax2.bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Time (s)', fontsize=12)
    ax2.set_title('Computational Cost', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join('analisis_avanzado', 'convergencia_malla.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {path}\n")

if __name__ == '__main__':
    analyze_mesh_independence()
