# 1. IMPORTS (Al principio de todo)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags, identity, csc_matrix
from scipy.sparse.linalg import spsolve, eigs
from numpy.linalg import cond, norm
import time
from scipy.interpolate import RectBivariateSpline, CubicSpline # Añadido CubicSpline
from scipy import interpolate
import os 

# 2. PARÁMETROS GLOBALES (Justo después de los imports y antes de la clase)
NY, NX = 5, 50
VY_TEST = 0.1
MAX_ITER = 500
TOLERANCE = 1e-8 # Tolerancia de convergencia global (Newton-Raphson)
# AUMENTO DE MAX_ITER E INCLUSO DE LA TOLERANCIA DEL SOLVER LINEAL (1e-4) para forzar la convergencia de Gauss-Seidel
MAX_LINEAR_ITER = 5000 
LINEAR_TOLERANCE = 1e-4 
V0_INITIAL = 1.0
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

# Crear carpeta para gráficas
CARPETA_GRAFICAS = "graficas_V4"
if not os.path.exists(CARPETA_GRAFICAS):
    os.makedirs(CARPETA_GRAFICAS)

# Función para limpiar gráficas previas
def limpiar_graficas_previas():
    """Elimina todas las gráficas previas de la carpeta."""
    if os.path.exists(CARPETA_GRAFICAS):
        for archivo in os.listdir(CARPETA_GRAFICAS):
            ruta_archivo = os.path.join(CARPETA_GRAFICAS, archivo)
            try:
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
            except Exception as e:
                print(f"No se pudo eliminar '{archivo}': {e}")


# 3. DEFINICIÓN DE LA CLASE
class FlujoNewtonRaphson:
    def __init__(self):
        self._incognita_map = {}
        self._preparar_mapa_incognitas()
        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)

    def _es_incognita(self, i, j):
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2): return False
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX: return False
        # Forzar a cero las celdas en j=2 para toda la región donde la viga influye
        # if j == 2 and i >= VIGA_INF_X_MIN: return False
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
        # Forzar a cero la fila j=2 desde el inicio de la viga en adelante (zona de influencia) - ELIMINADO
        # V_matrix[2, VIGA_INF_X_MIN:] = 0.0
        
        # Suavizar la entrada al obstáculo superior en la fila 4 (Top)
        # Descenso lineal desde X=30 hasta X=40 (donde empieza el obstáculo y es 0)
        V_matrix[4, 30:40] = np.linspace(V0_INITIAL, 0.0, 10)
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

    def _calcular_radio_espectral(self, J, method='jacobi'):
        try:
            D_diag = J.diagonal()
            if np.any(np.abs(D_diag) < 1e-12): return float('nan')
            D = diags(D_diag); L = J.tril(k=-1); U = J.triu(k=1)
            if method == 'jacobi':
                D_inv_diag = 1.0 / D_diag; D_inv = diags(D_inv_diag); T = -D_inv @ (L + U)
            elif method == 'gauss-seidel':
                D_plus_L = csc_matrix(D + L); T_U = spsolve(D_plus_L, U); T = -T_U
            else: return float('nan')
            eigenvalues = eigs(T, k=1, which='LM', return_eigenvectors=False)
            return np.abs(eigenvalues[0])
        except Exception: return float('nan')

    def _richardson(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER, omega=0.1):
        x = np.zeros_like(rhs); residuos_norm = []
        for i in range(max_iter):
            residual = rhs - J @ x; norm_res = norm(residual); residuos_norm.append(norm_res)
            if norm_res < tol: break
            x += omega * residual
        if len(residuos_norm) > 5:
            tasas = [residuos_norm[i+1]/residuos_norm[i] for i in range(len(residuos_norm)-5, len(residuos_norm)-1)]
            tasa_promedio = np.mean(tasas) if tasas else float('nan')
            print(f"    └─ Richardson: Tasa de convergencia práctica estimada ≈ {tasa_promedio:.4f}")
        return x

    def _jacobi(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER, omega=1.0): # OMEGA FIJADO A 1.0 (JACOBI PURO)
        x = np.zeros_like(rhs, dtype=float); D = J.diagonal().astype(float); D[np.abs(D) < 1e-12] = 1e-12
        L_plus_U = J - diags(D); D_inv = 1.0 / D
        for it in range(max_iter):
            # Jacobi puro (omega=1.0)
            x_new = ((rhs - L_plus_U @ x) * D_inv) 
            if norm(x_new - x) < tol: return x_new
            x = x_new
        return x

    def _gauss_seidel_puro(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER): # GAUSS-SEIDEL PURO (SIN RELAJACIÓN)
        x = np.zeros_like(rhs, dtype=float); n = self.N_INCÓGNITAS
        
        diag = np.array([J[i, i] for i in range(n)], dtype=float)
        diag[np.abs(diag) < 1e-12] = 1e-12 
        
        J_csr = J.tocsr()
        
        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                row_start, row_end = J_csr.indptr[i], J_csr.indptr[i+1]
                cols, data = J_csr.indices[row_start:row_end], J_csr.data[row_start:row_end]
                
                sigma = 0.0
                for col, val in zip(cols, data):
                    if col != i:
                        if col < i:
                            sigma += val * x[col]  # Valor actualizado
                        else:
                            sigma += val * x_old[col] # Valor anterior
                
                # Fórmula de Gauss-Seidel Pura (x[i] = (rhs[i] - sigma) / D[i,i])
                x[i] = (rhs[i] - sigma) / diag[i]
                
            if norm(x - x_old) < tol: return x
        return x

    def _gradient_descent(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        A = J.T @ J + 1e-8 * identity(self.N_INCÓGNITAS); b = J.T @ rhs; x = np.zeros_like(rhs, dtype=float)
        for i in range(max_iter):
            r = b - A @ x;
            if norm(r) < tol: return x
            denom = r.T @ A @ r; alpha = (r.T @ r) / denom if abs(denom) > 1e-20 else 1e-6
            x += alpha * r
        return x

    def _conjugate_gradient(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        A = J.T @ J + 1e-8 * identity(self.N_INCÓGNITAS); b = J.T @ rhs; x = np.zeros_like(rhs, dtype=float)
        r = b - A @ x; p, rs_old = r.copy(), float(r.T @ r)
        for i in range(max_iter):
            Ap = A @ p; denom = float(p.T @ Ap)
            if abs(denom) < 1e-20: break
            alpha = rs_old / denom; x += alpha * p; r -= alpha * Ap; rs_new = float(r.T @ r)
            if np.sqrt(rs_new) < tol: return x
            p = r + (rs_new / rs_old) * p; p = p.astype(float) # Asegurar float
            rs_old = rs_new
        return x

    def solve_linear_system(self, J, rhs, method):
        # Mapeo de métodos, apuntando Gauss-Seidel a la versión pura
        solvers = {'richardson': self._richardson, 'jacobi': self._jacobi, 'gauss-seidel': self._gauss_seidel_puro, 'gradient-descent': self._gradient_descent, 'conjugate-gradient': self._conjugate_gradient}
        if method not in solvers: raise ValueError(f"Método '{method}' no reconocido.")
        
        # Llamar al solver, usando _gauss_seidel_puro para 'gauss-seidel'
        return solvers[method](J, rhs)
    
    def solve(self, linear_solver_method, analisis_teorico=False):
        V_matrix = self.V_k.copy(); start_time = time.time(); converged, final_k = False, 0; historial_cond, historial_rs = [], []
        for k in range(1, MAX_ITER + 1):
            J, rhs = self.ensamblar_sistema_newton(V_matrix)
            if analisis_teorico:
                try:
                    cond_num = cond(J.toarray()); historial_cond.append(cond_num)
                    print(f"    ├─ Iteración {k}: Número de Condición ≈ {cond_num:.2e}")
                except Exception: historial_cond.append(float('inf'))
                if linear_solver_method in ['jacobi', 'gauss-seidel']:
                    rs = self._calcular_radio_espectral(J, method=linear_solver_method); historial_rs.append(rs)
                    print(f"    ├─ Iteración {k}: Radio Espectral ({linear_solver_method}) ≈ {rs:.4f}")
            Delta_V_vector = self.solve_linear_system(J, rhs, method=linear_solver_method)
            
            # Revisar si hay NaN o si el vector de cambio es demasiado grande (divergencia)
            if np.isnan(Delta_V_vector).any() or np.max(np.abs(Delta_V_vector)) > 100: 
                print(f"❌ Divergencia o NaN detectado en el solver lineal de {linear_solver_method}.")
                final_k=k; break
            
            V_new_matrix, m = V_matrix.copy(), 0
            for j in range(NY):
                for i in range(NX):
                    # Factor de relajación del método de Newton (mantener en 0.6)
                    if self._es_incognita(i, j): V_new_matrix[j, i] += 0.6 * Delta_V_vector[m]; m += 1
            V_matrix = np.clip(V_new_matrix, 0, V0_INITIAL); max_cambio = np.max(np.abs(Delta_V_vector))
            print(f"Iteración {k} ({linear_solver_method}): Cambio máximo = {max_cambio:.8f}")
            if max_cambio < TOLERANCE:
                print(f"✅ Convergencia global alcanzada en {k} iteraciones."); converged, final_k = True, k; break
            final_k = k
        return { "solution": V_matrix, "time": time.time() - start_time, "converged": converged, "iterations": final_k, "has_nan": np.isnan(V_matrix).any(), "historial_cond": historial_cond, "historial_rs": historial_rs }

# ----------------------------------------------------------------------
# --- FUNCIÓN DE INTERPOLACIÓN CÚBICA NATURAL (MANUAL EN 2 PASOS) ---
# ----------------------------------------------------------------------

def interpolate_cubic_natural_manual(V_low_res):
    """Aplica el Spline Cúbico Natural (1D) en dos pasos (X luego Y) para suavizar la visualización."""
    
    INTERP_RESOLUTION = 10 
    NY, NX = V_low_res.shape
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1
    
    x_low = np.linspace(0, NX - 1, NX)
    y_low = np.linspace(0, NY - 1, NY)
    x_high = np.linspace(0, NX - 1, NX_HIGH)
    y_high = np.linspace(0, NY - 1, NY_HIGH)
    
    # 1. Paso: Interpolar a lo largo de X (Filas)
    # V_mixed: Alta resolución en X, Baja resolución en Y
    V_mixed = np.zeros((NY, NX_HIGH))
    for j in range(NY):
        # Spline Cúbico Natural (bc_type='natural') para cada fila
        cs_x = CubicSpline(x_low, V_low_res[j, :], bc_type='natural')
        V_mixed[j, :] = cs_x(x_high)
        
    # 2. Paso: Interpolar a lo largo de Y (Columnas)
    # V_high_res: Alta resolución en Y, usando los datos de V_mixed
    V_high_res = np.zeros((NY_HIGH, NX_HIGH))
    for i_prime in range(NX_HIGH):
        # Spline Cúbico Natural (bc_type='natural') para cada columna
        cs_y = CubicSpline(y_low, V_mixed[:, i_prime], bc_type='natural')
        V_high_res[:, i_prime] = cs_y(y_high)
        
    return V_high_res

# 4. FUNCIONES AUXILIARES
def plot_solution(V_final, vy_value, method_name):
    # Aplicar Interpolación Spline Cúbico Natural (Manual)
    V_interpolated = interpolate_cubic_natural_manual(V_final)
    
    fig, ax = plt.subplots(figsize=(18, 8)); 
    
    # Usar la matriz interpolada para la visualización suave
    cax = ax.imshow(V_interpolated, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)
    
    fig.colorbar(cax, label='Velocidad (Vx) [Interpolada]'); 
    
    # Dibujar obstáculos
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), VIGA_INF_X_MAX-VIGA_INF_X_MIN, VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red')); 
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red'))
    
    # Título actualizado para reflejar el método de interpolación
    ax.set_title(f'Solución con: {method_name.upper()} (Visualización suavizada por Spline Cúbico Natural 1D)'); 
    ax.set_xlabel('X'); ax.set_ylabel('Y'); 
    plt.tight_layout(); 
    ruta_archivo = os.path.join(CARPETA_GRAFICAS, f"resultado_spline_natural_1D_{method_name}.png")
    plt.savefig(ruta_archivo); 
    plt.close(fig)
    print(f"Gráfico guardado: '{ruta_archivo}'")

def plot_solution_sin_spline(V_final, vy_value, method_name):
    """Genera un mapa de calor SIN spline (sin interpolación) que muestra el valor de velocidad en cada celda."""
    # Aumentar tamaño de figura para mejor visualización de números
    fig, ax = plt.subplots(figsize=(22, 10))
    
    # Usar la matriz original sin interpolación
    cax = ax.imshow(V_final, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL, interpolation='nearest')
    
    # Colorbar
    cbar = fig.colorbar(cax, label='Velocidad (Vx)', shrink=0.8)
    
    # Dibujar obstáculos
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), VIGA_INF_X_MAX-VIGA_INF_X_MIN, VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red', alpha=0.7, linewidth=2))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red', alpha=0.7, linewidth=2))
    
    # Calcular tamaño de fuente dinámico según el tamaño de la grilla
    fontsize = max(4, min(12, 300 / NX))
    
    # Agregar valores de velocidad en cada celda
    for j in range(NY):
        for i in range(NX):
            # Calcular la posición central de cada celda
            x_pos = i + 0.5
            y_pos = j + 0.5
            # Determinar color de texto basado en intensidad del fondo
            velocity = V_final[j, i]
            text_color = 'white' if velocity < 0.5 else 'black'
            # Mostrar el valor de Vx con 2 decimales
            ax.text(x_pos, y_pos, f'{V_final[j, i]:.2f}', 
                   ha='center', va='center', fontsize=fontsize, color=text_color, 
                   weight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='none', edgecolor='none', alpha=0))
    
    # Configuración de los ejes
    ax.set_xticks(np.arange(0, NX + 1, 5))
    ax.set_yticks(np.arange(0, NY + 1, 1))
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_title(f'Solución con: {method_name.upper()} (Mapa de Calor SIN Spline - Con Valores Vx)', fontsize=14, weight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    plt.tight_layout()
    ruta_archivo = os.path.join(CARPETA_GRAFICAS, f"resultado_sin_spline_{method_name}.png")
    plt.savefig(ruta_archivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico guardado: '{ruta_archivo}'")

def analizar_y_mostrar_resultados(results):
    print("\n\n--- ANÁLISIS COMPARATIVO DE MÉTODOS ITERATIVOS ---"); print("=" * 80)
    print(f"{'Método':<20} | {'Convergió':<10} | {'Tiempo (s)':<12} | {'Iteraciones':<12} | {'Análisis Teórico'}"); print("-" * 80)
    validos = {m: d for m, d in results.items() if d["converged"] and not d["has_nan"]}
    for method, data in results.items():
        analisis_str = ""
        if data["historial_cond"]: analisis_str += f"Cond. final: {data['historial_cond'][-1]:.2e}"
        if data["historial_rs"]: analisis_str += f" | RS final: {data['historial_rs'][-1]:.4f}"
        print(f"{method:<20} | {'Sí' if data['converged'] else 'No':<10} | {data['time']:<12.4f} | {data['iterations']:<12} | {analisis_str}")
    print("=" * 80)
    if not validos: print("\n🏆 Resultado: Ningún método iterativo convergió exitosamente.")
    else:
        mejor_metodo = min(validos, key=lambda m: validos[m]['time'])
        print(f"\n🏆 El mejor método práctico es '{mejor_metodo.upper()}' (el más rápido en converger).")

# 5. BLOQUE PRINCIPAL
if __name__ == '__main__':
    
    # El spline requiere este import adicional que faltaba en el código original
    try:
        from scipy import interpolate
    except ImportError:
        print("Error: La librería 'scipy' no está instalada. Es necesaria para el Spline Cúbico.")
        exit()
    
    # Limpiar gráficas previas
    print("🧹 Eliminando gráficas previas...")
    limpiar_graficas_previas()

    metodos_iterativos_a_probar = [ 'jacobi', 'gauss-seidel', 'richardson', 'gradient-descent', 'conjugate-gradient' ]
    metodos_para_analisis_profundo = ['jacobi', 'gauss-seidel'] 
    
    resultados_iterativos = {}
    
    for metodo in metodos_iterativos_a_probar:
        print(f"\n--- INICIANDO SIMULACIÓN CON: {metodo.upper()} ---")
        realizar_analisis = metodo in metodos_para_analisis_profundo
        if realizar_analisis: print("   (Análisis teórico profundo activado)")
        solver = FlujoNewtonRaphson()
        resultados_iterativos[metodo] = solver.solve(linear_solver_method=metodo, analisis_teorico=realizar_analisis)
    
    analizar_y_mostrar_resultados(resultados_iterativos)
    
    print("\n--- GENERANDO GRÁFICOS (Suavizados por Spline Cúbico Natural 1D Manual) ---")
    
    # Genera el gráfico para todos los métodos que converjan correctamente
    for metodo, data in resultados_iterativos.items():
        if data["converged"] and not data["has_nan"]: 
            # Se llama a la función plot_solution que ahora usa interpolate_cubic_natural_manual
            plot_solution(data["solution"], VY_TEST, metodo)
            # Generar también el mapa sin spline con valores de velocidad
            plot_solution_sin_spline(data["solution"], VY_TEST, metodo)
        else:
            print(f"🚫 Saltando gráfico para {metodo.upper()}: No convergió o la solución contiene NaN.")