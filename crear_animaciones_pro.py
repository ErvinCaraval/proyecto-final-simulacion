import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import lil_matrix, diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from scipy.interpolate import CubicSpline, RectBivariateSpline
import os

# Aesthetic Configuration
plt.style.use('dark_background')
COLOR_MAP = 'plasma'
OBSTACLE_COLOR = '#FF0055'
TEXT_COLOR = '#FFFFFF'
GRID_COLOR = '#333333'

# Global Parameters
NY, NX = 5, 50
VY_TEST = 0.1
V0_INITIAL = 1.0
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

OUTPUT_DIR = "animaciones_pro"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class AnimatedNewtonRaphsonFlow:
    def __init__(self):
        self._unknown_map = {}
        self._map_unknowns()
        self.V_k = self._initialize_velocity_field(V0_INITIAL)
        self.n_unknowns = len(self._unknown_map)

    def _is_unknown(self, i, j):
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2): return False
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX: return False
        if j == 2 and i >= VIGA_INF_X_MIN: return False
        return True
    
    def _map_unknowns(self):
        count = 0
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    self._unknown_map[(i, j)] = count
                    count += 1
    
    def _get_linear_index(self, i, j):
        return self._unknown_map.get((i, j), None)

    def _initialize_velocity_field(self, v_init):
        V_matrix = np.full((NY, NX), v_init)
        V_matrix[NY - 1, :], V_matrix[:, 0] = V0_INITIAL, V0_INITIAL
        V_matrix[0, :], V_matrix[:, NX - 1] = 0.0, 0.0
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        V_matrix[2, VIGA_INF_X_MIN:] = 0.0
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1))
        return V_matrix

    def assemble_system(self, V_current):
        J = lil_matrix((self.n_unknowns, self.n_unknowns))
        rhs = np.zeros(self.n_unknowns)
        m = 0
        for j in range(NY):
            for i in range(NX):
                if not self._is_unknown(i, j): continue
                V_c = V_current[j, i]
                V_r = V_current[j, i+1]
                V_l = V_current[j, i-1]
                V_u = V_current[j+1, i]
                V_d = V_current[j-1, i]
                
                rhs[m] = -(4*V_c - (V_r+V_l+V_u+V_d) + 4*V_c*(V_r-V_l) + 4*VY_TEST*(V_u-V_d))
                J[m, m] = 4 + 4*V_r - 4*V_l
                
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

    def _conjugate_gradient(self, J, rhs, tol=1e-4, max_iter=5000):
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

    def generate_convergence_history(self, max_frames=60):
        history = [self.V_k.copy()]
        V_matrix = self.V_k.copy()
        
        print("Simulating evolution...")
        for k in range(1, max_frames + 1):
            J, rhs = self.assemble_system(V_matrix)
            Delta_V = self._conjugate_gradient(J, rhs)
            
            V_new = V_matrix.copy()
            m = 0
            for j in range(NY):
                for i in range(NX):
                    if self._is_unknown(i, j):
                        V_new[j, i] += 0.6 * Delta_V[m]
                        m += 1
            
            V_matrix = np.clip(V_new, 0, V0_INITIAL)
            history.append(V_matrix.copy())
            
            max_change = np.max(np.abs(Delta_V))
            if max_change < 1e-6:
                print(f"Convergence reached at frame {k}")
                for _ in range(60): history.append(V_matrix.copy())
                break
                
        return history

def interpolate_history(history, extra_steps=19):
    smooth_history = []
    for i in range(len(history) - 1):
        v_start = history[i]
        v_end = history[i+1]
        
        if np.allclose(v_start, v_end):
            smooth_history.append(v_start)
            continue
            
        for step in range(extra_steps + 1):
            alpha = step / (extra_steps + 1)
            v_interp = v_start * (1 - alpha) + v_end * alpha
            smooth_history.append(v_interp)
    smooth_history.append(history[-1])
    return smooth_history

def interpolate_cubic_natural_manual(V_low_res):
    INTERP_RESOLUTION = 10 
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1
    x_low = np.linspace(0, NX - 1, NX); y_low = np.linspace(0, NY - 1, NY)
    x_high = np.linspace(0, NX - 1, NX_HIGH); y_high = np.linspace(0, NY - 1, NY_HIGH)
    
    V_mixed = np.zeros((NY, NX_HIGH))
    for j in range(NY):
        cs_x = CubicSpline(x_low, V_low_res[j, :], bc_type='natural')
        V_mixed[j, :] = cs_x(x_high)
        
    V_high_res = np.zeros((NY_HIGH, NX_HIGH))
    for i_prime in range(NX_HIGH):
        cs_y = CubicSpline(y_low, V_mixed[:, i_prime], bc_type='natural')
        V_high_res[:, i_prime] = cs_y(y_high)
    return V_high_res

def interpolate_bicubic_natural(V_low_res):
    INTERP_RESOLUTION = 10 
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1
    x_low = np.linspace(0, NX - 1, NX); y_low = np.linspace(0, NY - 1, NY)
    x_high = np.linspace(0, NX - 1, NX_HIGH); y_high = np.linspace(0, NY - 1, NY_HIGH)
    interp_func = RectBivariateSpline(y_low, x_low, V_low_res, kx=3, ky=3)
    return interp_func(y_high, x_high)

def create_animation(interpolation_type, filename):
    print(f"Generating extended animation: {filename}...")
    solver = AnimatedNewtonRaphsonFlow()
    raw_history = solver.generate_convergence_history(max_frames=50)
    
    history = interpolate_history(raw_history, extra_steps=19)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_xlabel("X Position", color=TEXT_COLOR)
    ax.set_ylabel("Y Position", color=TEXT_COLOR)
    ax.set_title(f"CFD Simulation - {interpolation_type.upper()}", color=TEXT_COLOR, fontsize=14, weight='bold')
    
    rect1 = plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), VIGA_INF_X_MAX-VIGA_INF_X_MIN, VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, 
                          color=OBSTACLE_COLOR, alpha=0.8, zorder=10)
    rect2 = plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, 
                          color=OBSTACLE_COLOR, alpha=0.8, zorder=10)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    if interpolation_type == "v4_manual":
        initial_data = interpolate_cubic_natural_manual(history[0])
    else:
        initial_data = interpolate_bicubic_natural(history[0])
        
    im = ax.imshow(initial_data, cmap=COLOR_MAP, origin='lower', extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL, animated=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Flow Velocity', color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
    
    text_iter = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold')

    def update(frame):
        V_frame = history[frame]
        
        if interpolation_type == "v4_manual":
            V_interp = interpolate_cubic_natural_manual(V_frame)
        else:
            V_interp = interpolate_bicubic_natural(V_frame)
            
        im.set_array(V_interp)
        
        iter_real = int(frame / 20) 
        text_iter.set_text(f'Iteration: {iter_real}')
        return im, text_iter

    ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)
    
    full_path = os.path.join(OUTPUT_DIR, filename)
    ani.save(full_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"Animation saved to: {full_path}")

if __name__ == "__main__":
    create_animation("v4_manual", "simulation_v4_manual.gif")
    create_animation("v4_1_bicubic", "simulation_v4_1_bicubic.gif")
