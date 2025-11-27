import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import lil_matrix, diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from scipy.interpolate import CubicSpline, RectBivariateSpline
import os

# --- CONFIGURACIÓN ESTÉTICA "PRO" ---
plt.style.use('dark_background')
COLOR_MAP = 'plasma' # 'inferno', 'magma', 'plasma', 'viridis'
OBSTACLE_COLOR = '#FF0055' # Neon Red/Pink
TEXT_COLOR = '#FFFFFF'
GRID_COLOR = '#333333'

# --- PARÁMETROS GLOBALES ---
NY, NX = 5, 50
VY_TEST = 0.1
V0_INITIAL = 1.0
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

CARPETA_SALIDA = "animaciones_pro"
if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

# --- CLASE DE SIMULACIÓN (ADAPTADA PARA ANIMACIÓN) ---
class FlujoNewtonRaphsonAnimado:
    def __init__(self):
        self._incognita_map = {}
        self._preparar_mapa_incognitas()
        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)

    def _es_incognita(self, i, j):
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2): return False
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX: return False
        if j == 2 and i >= VIGA_INF_X_MIN: return False
        return True
    
    def _preparar_mapa_incognitas(self):
        count = 0
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j): self._incognita_map[(i, j)] = count; count += 1
    
    def _map_to_linear_index(self, i, j): return self._incognita_map.get((i, j), None)

    def _inicializar_matriz_velocidades(self, v_init):
        V_matrix = np.full((NY, NX), v_init)
        V_matrix[NY - 1, :], V_matrix[:, 0] = V0_INITIAL, V0_INITIAL
        V_matrix[0, :], V_matrix[:, NX - 1] = 0.0, 0.0
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        V_matrix[2, VIGA_INF_X_MIN:] = 0.0
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j): V_matrix[j, i] = v_init * (j / (NY - 1))
        return V_matrix

    def ensamblar_sistema_newton(self, V_current):
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        rhs = np.zeros(self.N_INCÓGNITAS)
        m = 0
        for j in range(NY):
            for i in range(NX):
                if not self._es_incognita(i, j): continue
                V_c,V_r,V_l,V_u,V_d = V_current[j,i],V_current[j,i+1],V_current[j,i-1],V_current[j+1,i],V_current[j-1,i]
                rhs[m] = -(4*V_c - (V_r+V_l+V_u+V_d) + 4*V_c*(V_r-V_l) + 4*VY_TEST*(V_u-V_d))
                J[m, m] = 4 + 4*V_r - 4*V_l
                n_r, n_l = self._map_to_linear_index(i+1, j), self._map_to_linear_index(i-1, j)
                n_u, n_d = self._map_to_linear_index(i, j+1), self._map_to_linear_index(i, j-1)
                if n_r is not None: J[m, n_r] = -1 + 4*V_c
                if n_l is not None: J[m, n_l] = -1 - 4*V_c
                if n_u is not None: J[m, n_u] = -1 + 4*VY_TEST
                if n_d is not None: J[m, n_d] = -1 - 4*VY_TEST
                m += 1
        return J.tocsr(), rhs

    def _conjugate_gradient(self, J, rhs, tol=1e-4, max_iter=5000):
        A = J.T @ J + 1e-8 * identity(self.N_INCÓGNITAS); b = J.T @ rhs; x = np.zeros_like(rhs, dtype=float)
        r = b - A @ x; p, rs_old = r.copy(), float(r.T @ r)
        for i in range(max_iter):
            Ap = A @ p; denom = float(p.T @ Ap)
            if abs(denom) < 1e-20: break
            alpha = rs_old / denom; x += alpha * p; r -= alpha * Ap; rs_new = float(r.T @ r)
            if np.sqrt(rs_new) < tol: return x
            p = r + (rs_new / rs_old) * p; p = p.astype(float)
            rs_old = rs_new
        return x

    def generar_historial_convergencia(self, max_frames=60):
        """Genera una lista de matrices V que representan la evolución de la solución."""
        historial = [self.V_k.copy()]
        V_matrix = self.V_k.copy()
        
        # Usamos Conjugate Gradient porque es el más estable y visualmente interesante
        print("Simulando evolución...")
        for k in range(1, max_frames + 1):
            J, rhs = self.ensamblar_sistema_newton(V_matrix)
            Delta_V_vector = self._conjugate_gradient(J, rhs)
            
            V_new_matrix, m = V_matrix.copy(), 0
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j): V_new_matrix[j, i] += 0.6 * Delta_V_vector[m]; m += 1 # Factor relajación 0.6
            
            V_matrix = np.clip(V_new_matrix, 0, V0_INITIAL)
            historial.append(V_matrix.copy())
            
            max_cambio = np.max(np.abs(Delta_V_vector))
            if max_cambio < 1e-6:
                print(f"Convergencia alcanzada en frame {k}")
                # Repetir el último frame para pausar el final (AUMENTADO A 60 FRAMES)
                for _ in range(60): historial.append(V_matrix.copy())
                break
                
        return historial

def interpolar_historial(historial, pasos_extra=19):
    """Genera frames intermedios para suavizar y alargar la animación (Super Slow Motion)."""
    historial_suave = []
    for i in range(len(historial) - 1):
        v_start = historial[i]
        v_end = historial[i+1]
        # Si son idénticos (pausa final), solo agregarlos
        if np.allclose(v_start, v_end):
            historial_suave.append(v_start)
            continue
            
        for step in range(pasos_extra + 1):
            alpha = step / (pasos_extra + 1)
            v_interp = v_start * (1 - alpha) + v_end * alpha
            historial_suave.append(v_interp)
    historial_suave.append(historial[-1])
    return historial_suave

# --- FUNCIONES DE INTERPOLACIÓN ---

def interpolate_cubic_natural_manual(V_low_res):
    """V4: Interpolación Manual 1D (X luego Y)"""
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
    """V4.1: Interpolación Bicúbica (Más suave)"""
    INTERP_RESOLUTION = 10 
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1
    x_low = np.linspace(0, NX - 1, NX); y_low = np.linspace(0, NY - 1, NY)
    x_high = np.linspace(0, NX - 1, NX_HIGH); y_high = np.linspace(0, NY - 1, NY_HIGH)
    interp_func = RectBivariateSpline(y_low, x_low, V_low_res, kx=3, ky=3)
    return interp_func(y_high, x_high)

# --- GENERADOR DE ANIMACIÓN ---

def crear_animacion(tipo_interpolacion, nombre_archivo):
    print(f"Generando animación SUPER extendida: {nombre_archivo}...")
    solver = FlujoNewtonRaphsonAnimado()
    historial_raw = solver.generar_historial_convergencia(max_frames=50)
    
    # APLICAR INTERPOLACIÓN TEMPORAL (SUPER SLOW MOTION)
    # 19 pasos extra entre cada iteración = 20x más frames de simulación
    historial = interpolar_historial(historial_raw, pasos_extra=19)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Configurar ejes y fondo
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_xlabel("Posición X", color=TEXT_COLOR)
    ax.set_ylabel("Posición Y", color=TEXT_COLOR)
    ax.set_title(f"Simulación CFD - {tipo_interpolacion.upper()}", color=TEXT_COLOR, fontsize=14, weight='bold')
    
    # Dibujar obstáculos (Estático)
    rect1 = plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), VIGA_INF_X_MAX-VIGA_INF_X_MIN, VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, 
                          color=OBSTACLE_COLOR, alpha=0.8, zorder=10)
    rect2 = plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, 
                          color=OBSTACLE_COLOR, alpha=0.8, zorder=10)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Inicializar imagen
    if tipo_interpolacion == "v4_manual":
        data_inicial = interpolate_cubic_natural_manual(historial[0])
    else:
        data_inicial = interpolate_bicubic_natural(historial[0])
        
    im = ax.imshow(data_inicial, cmap=COLOR_MAP, origin='lower', extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL, animated=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Velocidad de Flujo', color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
    
    text_iter = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold')

    def update(frame):
        V_frame = historial[frame]
        
        if tipo_interpolacion == "v4_manual":
            V_interp = interpolate_cubic_natural_manual(V_frame)
        else:
            V_interp = interpolate_bicubic_natural(V_frame)
            
        im.set_array(V_interp)
        
        # Calcular número de iteración real aproximado
        # Cada 20 frames es 1 iteración real (1 original + 19 extra)
        iter_real = int(frame / 20) 
        text_iter.set_text(f'Iteración: {iter_real}')
        return im, text_iter

    ani = FuncAnimation(fig, update, frames=len(historial), interval=100, blit=True)
    
    ruta_completa = os.path.join(CARPETA_SALIDA, nombre_archivo)
    ani.save(ruta_completa, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"✅ Animación guardada en: {ruta_completa}")

if __name__ == "__main__":
    # Generar V4 (Manual)
    crear_animacion("v4_manual", "simulacion_v4_manual.gif")
    
    # Generar V4.1 (Bicúbica)
    crear_animacion("v4_1_bicubica", "simulacion_v4_1_bicubica.gif")
