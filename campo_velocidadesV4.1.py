import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve, eigs
from numpy.linalg import cond, norm
import time
from scipy import interpolate
import os

# Global Configuration
NY, NX = 5, 50
VY_TEST = 0.1
MAX_ITER = 500
TOLERANCE = 1e-8
MAX_LINEAR_ITER = 5000
LINEAR_TOLERANCE = 1e-4
V0_INITIAL = 1.0

# Obstacle Definitions
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

OUTPUT_DIR = "graficas_V4_1"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clear_previous_plots():
    """Cleans up the output directory before running new simulations."""
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(f"Error removing {filename}: {e}")

class NewtonRaphsonFlowSolver:
    """
    Solves the fluid flow equations using the Newton-Raphson method.
    Handles system assembly, boundary conditions, and linear solvers.
    """
    def __init__(self):
        self._unknown_map = {}
        self._map_unknowns()
        self.V_k = self._initialize_velocity_field(V0_INITIAL)
        self.n_unknowns = len(self._unknown_map)

    def _is_unknown(self, i, j):
        """Determines if a grid point (i, j) is a variable to be solved."""
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2): return False
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX: return False
        return True
    
    def _map_unknowns(self):
        """Maps grid coordinates to linear system indices."""
        count = 0
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    self._unknown_map[(i, j)] = count
                    count += 1
    
    def _get_linear_index(self, i, j):
        return self._unknown_map.get((i, j), None)

    def _initialize_velocity_field(self, v_init):
        """Sets initial conditions and boundary constraints."""
        V_matrix = np.full((NY, NX), v_init)
        # Boundary conditions
        V_matrix[NY - 1, :] = V0_INITIAL
        V_matrix[:, 0] = V0_INITIAL
        V_matrix[0, :] = 0.0
        V_matrix[:, NX - 1] = 0.0
        
        # Obstacles
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        
        # Smooth inlet for top obstacle
        V_matrix[4, 30:40] = np.linspace(V0_INITIAL, 0.0, 10)
        
        # Linear initialization for internal points
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1))
        return V_matrix

    def assemble_system(self, V_current):
        """Constructs the Jacobian matrix and RHS vector for the current iteration."""
        J = lil_matrix((self.n_unknowns, self.n_unknowns))
        rhs = np.zeros(self.n_unknowns)
        m = 0
        for j in range(NY):
            for i in range(NX):
                if not self._is_unknown(i, j): continue
                
                # Stencil values
                V_c = V_current[j, i]
                V_r = V_current[j, i+1]
                V_l = V_current[j, i-1]
                V_u = V_current[j+1, i]
                V_d = V_current[j-1, i]
                
                # Residual calculation
                rhs[m] = -(4*V_c - (V_r+V_l+V_u+V_d) + 4*V_c*(V_r-V_l) + 4*VY_TEST*(V_u-V_d))
                
                # Jacobian diagonal
                J[m, m] = 4 + 4*V_r - 4*V_l
                
                # Jacobian off-diagonals
                n_r = self._get_linear_index(i+1, j)
                n_l = self._get_linear_index(i-1, j)
                n_u = self._get_linear_index(i, j+1)
                n_d = self._get_linear_index(i, j-1)
                
                if n_r is not None: J[m, n_r] = -1 + 4*V_c
                if n_l is not None: J[m, n_l] = -1 - 4*V_c
                if n_u is not None: J[m, n_u] = -1 + 4*VY_TEST
                if n_d is not None: J[m, n_d] = -1 - 4*VY_TEST
                m += 1
        return J.tocsr(), rhs

    def _calculate_spectral_radius(self, J, method='jacobi'):
        """Computes the spectral radius to analyze convergence properties."""
        try:
            D_diag = J.diagonal()
            if np.any(np.abs(D_diag) < 1e-12): return float('nan')
            
            D = diags(D_diag)
            L = J.tril(k=-1)
            U = J.triu(k=1)
            
            if method == 'jacobi':
                D_inv = diags(1.0 / D_diag)
                T = -D_inv @ (L + U)
            elif method == 'gauss-seidel':
                D_plus_L = csc_matrix(D + L)
                T_U = spsolve(D_plus_L, U)
                T = -T_U
            else:
                return float('nan')
                
            eigenvalues = eigs(T, k=1, which='LM', return_eigenvectors=False)
            return np.abs(eigenvalues[0])
        except Exception:
            return float('nan')

    def _richardson(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER, omega=0.1):
        x = np.zeros_like(rhs)
        for _ in range(max_iter):
            residual = rhs - J @ x
            if norm(residual) < tol: break
            x += omega * residual
        return x

    def _jacobi(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        x = np.zeros_like(rhs, dtype=float)
        D = J.diagonal().astype(float)
        D[np.abs(D) < 1e-12] = 1e-12
        L_plus_U = J - diags(D)
        D_inv = 1.0 / D
        
        for _ in range(max_iter):
            x_new = ((rhs - L_plus_U @ x) * D_inv)
            if norm(x_new - x) < tol: return x_new
            x = x_new
        return x

    def _gauss_seidel(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        x = np.zeros_like(rhs, dtype=float)
        n = self.n_unknowns
        diag = np.array([J[i, i] for i in range(n)], dtype=float)
        diag[np.abs(diag) < 1e-12] = 1e-12
        J_csr = J.tocsr()
        
        for _ in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                row_start, row_end = J_csr.indptr[i], J_csr.indptr[i+1]
                cols, data = J_csr.indices[row_start:row_end], J_csr.data[row_start:row_end]
                
                sigma = 0.0
                for col, val in zip(cols, data):
                    if col != i:
                        sigma += val * (x[col] if col < i else x_old[col])
                
                x[i] = (rhs[i] - sigma) / diag[i]
                
            if norm(x - x_old) < tol: return x
        return x

    def _gradient_descent(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        A = J.T @ J + 1e-8 * identity(self.n_unknowns)
        b = J.T @ rhs
        x = np.zeros_like(rhs, dtype=float)
        for _ in range(max_iter):
            r = b - A @ x
            if norm(r) < tol: return x
            denom = r.T @ A @ r
            alpha = (r.T @ r) / denom if abs(denom) > 1e-20 else 1e-6
            x += alpha * r
        return x

    def _conjugate_gradient(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        A = J.T @ J + 1e-8 * identity(self.n_unknowns)
        b = J.T @ rhs
        x = np.zeros_like(rhs, dtype=float)
        r = b - A @ x
        p = r.copy()
        rs_old = float(r.T @ r)
        
        for _ in range(max_iter):
            Ap = A @ p
            denom = float(p.T @ Ap)
            if abs(denom) < 1e-20: break
            alpha = rs_old / denom
            x += alpha * p
            r -= alpha * Ap
            rs_new = float(r.T @ r)
            if np.sqrt(rs_new) < tol: return x
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def solve_linear_system(self, J, rhs, method):
        solvers = {
            'richardson': self._richardson,
            'jacobi': self._jacobi,
            'gauss-seidel': self._gauss_seidel,
            'gradient-descent': self._gradient_descent,
            'conjugate-gradient': self._conjugate_gradient
        }
        if method not in solvers:
            raise ValueError(f"Unknown method: {method}")
        return solvers[method](J, rhs)
    
    def solve(self, linear_solver_method, theoretical_analysis=False):
        V_matrix = self.V_k.copy()
        start_time = time.time()
        converged = False
        final_k = 0
        cond_history, rs_history = [], []
        
        for k in range(1, MAX_ITER + 1):
            J, rhs = self.assemble_system(V_matrix)
            
            if theoretical_analysis:
                try:
                    cond_num = cond(J.toarray())
                    cond_history.append(cond_num)
                    print(f"    Iteration {k}: Condition Number = {cond_num:.2e}")
                except Exception:
                    cond_history.append(float('inf'))
                
                if linear_solver_method in ['jacobi', 'gauss-seidel']:
                    rs = self._calculate_spectral_radius(J, method=linear_solver_method)
                    rs_history.append(rs)
                    print(f"    Iteration {k}: Spectral Radius ({linear_solver_method}) = {rs:.4f}")
            
            Delta_V = self.solve_linear_system(J, rhs, method=linear_solver_method)
            
            if np.isnan(Delta_V).any() or np.max(np.abs(Delta_V)) > 100:
                print(f"Divergence or NaN detected in {linear_solver_method}.")
                final_k = k
                break
            
            V_new = V_matrix.copy()
            m = 0
            for j in range(NY):
                for i in range(NX):
                    if self._is_unknown(i, j):
                        V_new[j, i] += 0.6 * Delta_V[m] # Relaxation factor
                        m += 1
            
            V_matrix = np.clip(V_new, 0, V0_INITIAL)
            max_change = np.max(np.abs(Delta_V))
            print(f"Iteration {k} ({linear_solver_method}): Max change = {max_change:.8f}")
            
            if max_change < TOLERANCE:
                print(f"Convergence reached in {k} iterations.")
                converged = True
                final_k = k
                break
            final_k = k
            
        return {
            "solution": V_matrix,
            "time": time.time() - start_time,
            "converged": converged,
            "iterations": final_k,
            "has_nan": np.isnan(V_matrix).any(),
            "cond_history": cond_history,
            "rs_history": rs_history
        }

def interpolate_bicubic(V_low_res):
    """Applies bicubic interpolation for smoother visualization."""
    INTERP_RESOLUTION = 10 
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1
    
    x_low = np.linspace(0, NX - 1, NX)
    y_low = np.linspace(0, NY - 1, NY)
    x_high = np.linspace(0, NX - 1, NX_HIGH)
    y_high = np.linspace(0, NY - 1, NY_HIGH)
    
    interp_func = interpolate.RectBivariateSpline(y_low, x_low, V_low_res, kx=3, ky=3)
    return interp_func(y_high, x_high)

def plot_solution(V_final, vy_value, method_name):
    V_interpolated = interpolate_bicubic(V_final)
    fig, ax = plt.subplots(figsize=(18, 8))
    
    cax = ax.imshow(V_interpolated, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)
    
    fig.colorbar(cax, label='Velocity (Vx) [Interpolated]')
    
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red'))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red'))
    
    ax.set_title(f'Solution: {method_name.upper()} (Bicubic Spline)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, f"result_spline_{method_name}.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: '{filename}'")

def plot_solution_no_spline(V_final, vy_value, method_name):
    fig, ax = plt.subplots(figsize=(22, 10))
    cax = ax.imshow(V_final, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL, interpolation='nearest')
    
    fig.colorbar(cax, label='Velocity (Vx)', shrink=0.8)
    
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red', alpha=0.7, linewidth=2))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red', alpha=0.7, linewidth=2))
    
    fontsize = max(4, min(12, 300 / NX))
    
    for j in range(NY):
        for i in range(NX):
            x_pos = i + 0.5
            y_pos = j + 0.5
            velocity = V_final[j, i]
            text_color = 'white' if velocity < 0.5 else 'black'
            ax.text(x_pos, y_pos, f'{V_final[j, i]:.2f}', 
                   ha='center', va='center', fontsize=fontsize, color=text_color, 
                   weight='bold')
    
    ax.set_xticks(np.arange(0, NX + 1, 5))
    ax.set_yticks(np.arange(0, NY + 1, 1))
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_title(f'Solution: {method_name.upper()} (No Spline - Raw Values)', fontsize=14, weight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, f"result_no_spline_{method_name}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: '{filename}'")

def analyze_results(results):
    print("\n--- ITERATIVE METHODS COMPARISON ---")
    print(f"{'Method':<20} | {'Converged':<10} | {'Time (s)':<12} | {'Iterations':<12} | {'Analysis'}")
    print("-" * 80)
    
    valid_methods = {m: d for m, d in results.items() if d["converged"] and not d["has_nan"]}
    
    for method, data in results.items():
        analysis_str = ""
        if data["cond_history"]: analysis_str += f"Final Cond: {data['cond_history'][-1]:.2e}"
        if data["rs_history"]: analysis_str += f" | Final RS: {data['rs_history'][-1]:.4f}"
        
        print(f"{method:<20} | {'Yes' if data['converged'] else 'No':<10} | {data['time']:<12.4f} | {data['iterations']:<12} | {analysis_str}")
    print("-" * 80)
    
    if not valid_methods:
        print("\nResult: No iterative method converged successfully.")
    else:
        best_method = min(valid_methods, key=lambda m: valid_methods[m]['time'])
        print(f"\nBest practical method: '{best_method.upper()}' (fastest convergence).")

if __name__ == '__main__':
    try:
        from scipy import interpolate
    except ImportError:
        print("Error: 'scipy' is required for Cubic Spline interpolation.")
        exit()
    
    print("Cleaning previous plots...")
    clear_previous_plots()

    methods_to_test = ['jacobi', 'gauss-seidel', 'richardson', 'gradient-descent', 'conjugate-gradient']
    methods_for_deep_analysis = ['jacobi', 'gauss-seidel'] 
    
    results = {}
    
    for method in methods_to_test:
        print(f"\n--- STARTING SIMULATION: {method.upper()} ---")
        run_analysis = method in methods_for_deep_analysis
        if run_analysis: print("   (Deep theoretical analysis enabled)")
        
        solver = NewtonRaphsonFlowSolver()
        results[method] = solver.solve(linear_solver_method=method, theoretical_analysis=run_analysis)
    
    analyze_results(results)
    
    print("\n--- GENERATING PLOTS ---")
    
    for method, data in results.items():
        if data["converged"] and not data["has_nan"]: 
            plot_solution(data["solution"], VY_TEST, method)
            plot_solution_no_spline(data["solution"], VY_TEST, method)
        else:
            print(f"Skipping plot for {method.upper()}: Did not converge or contains NaN.")