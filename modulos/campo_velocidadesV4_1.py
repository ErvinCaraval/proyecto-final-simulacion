# Importación de librerías necesarias
import numpy as np  # Operaciones numéricas y matrices
import matplotlib.pyplot as plt  # Visualización de gráficas
from scipy.sparse import lil_matrix, diags, identity, csc_matrix  # Matrices dispersas para sistemas grandes
from scipy.sparse.linalg import spsolve, eigs  # Solvers y cálculo de eigenvalores
from numpy.linalg import cond, norm  # Número de condición y normas vectoriales
import time  # Medición de tiempos de ejecución
from scipy import interpolate  # Interpolación bicúbica
import os  # Manejo de directorios y archivos

# ========== Configuración Global del Dominio ==========
NY, NX = 5, 50  # Número de puntos de la malla (NY filas, NX columnas)
VY_TEST = 0.1  # Velocidad vertical de prueba (componente Vy del flujo)
MAX_ITER = 500  # Número máximo de iteraciones para Newton-Raphson
TOLERANCE = 1e-8  # Tolerancia de convergencia (criterio de parada)
MAX_LINEAR_ITER = 5000  # Iteraciones máximas para solucionadores lineales iterativos
LINEAR_TOLERANCE = 1e-4  # Tolerancia para convergencia de solucionadores lineales
V0_INITIAL = 1.0  # Velocidad inicial/de entrada del flujo

# ========== Definición de Obstáculos ==========
# Obstáculo inferior (viga horizontal inferior)
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2  # Rango vertical del obstáculo inferior
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30  # Rango horizontal del obstáculo inferior

# Obstáculo superior (viga horizontal superior)
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5  # Rango vertical del obstáculo superior
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50  # Rango horizontal del obstáculo superior

# Directorio de salida para gráficas
OUTPUT_DIR = "graficas_V4_1"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clear_previous_plots():
    """
    Limpia el directorio de salida eliminando gráficas anteriores.
    Esto asegura que solo se guarden los resultados de la simulación actual.
    """
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Eliminar archivo existente
            except OSError as e:
                print(f"Error removing {filename}: {e}")

class NewtonRaphsonFlowSolver:
    """
    Resuelve las ecuaciones de flujo de fluidos usando el método de Newton-Raphson.
    
    Esta clase implementa:
    - Ensamblaje del sistema jacobiano para ecuaciones no lineales de flujo
    - Aplicación de condiciones de contorno (entrada, salida, obstáculos)
    - 5 métodos de solución iterativos para sistemas lineales
    - Análisis de convergencia (número de condición, radio espectral)
    """
    def __init__(self):
        """Inicializa el solver con el campo de velocidad y mapeo de incógnitas."""
        self._unknown_map = {}  # Diccionario que mapea coordenadas (i,j) a índices lineales
        self._map_unknowns()  # Crear mapeo de variables incógnitas
        self.V_k = self._initialize_velocity_field(V0_INITIAL)  # Campo de velocidad inicial
        self.n_unknowns = len(self._unknown_map)  # Total de incógnitas a resolver

    def _is_unknown(self, i, j):
        """
        Determina si un punto de la malla (i, j) es una incógnita a resolver.
        
        Args:
            i: Índice horizontal (columna)
            j: Índice vertical (fila)
        
        Returns:
            True si el punto es una incógnita, False si es condición de contorno u obstáculo
        """
        # Excluir bordes (condiciones de contorno conocidas)
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2): return False
        # Excluir puntos dentro del obstáculo inferior
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX: return False
        return True
    
    def _map_unknowns(self):
        """
        Crea un mapeo de coordenadas de malla a índices del sistema lineal.
        Esto permite convertir entre la malla 2D y el vector 1D de incógnitas.
        """
        count = 0
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    self._unknown_map[(i, j)] = count  # Asignar índice lineal
                    count += 1
    
    def _get_linear_index(self, i, j):
        """Obtiene el índice lineal para un punto de malla (i, j)."""
        return self._unknown_map.get((i, j), None)

    def _initialize_velocity_field(self, v_init):
        """
        Establece las condiciones iniciales y de contorno del campo de velocidad.
        
        Args:
            v_init: Valor de velocidad inicial para puntos internos
        
        Returns:
            Matriz NY×NX con el campo de velocidad inicializado
        """
        V_matrix = np.full((NY, NX), v_init)  # Inicialización uniforme
        
        # ===== Condiciones de Contorno =====
        V_matrix[NY - 1, :] = V0_INITIAL  # Borde superior: entrada de flujo
        V_matrix[:, 0] = V0_INITIAL  # Borde izquierdo: entrada de flujo
        V_matrix[0, :] = 0.0  # Borde inferior: pared (velocidad cero)
        V_matrix[:, NX - 1] = 0.0  # Borde derecho: salida libre
        
        # ===== Obstáculos (velocidad cero en paredes sólidas) =====
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0  # Obstáculo inferior
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0  # Obstáculo superior
        
        # ===== Transición suave cerca del obstáculo superior =====
        V_matrix[4, 30:40] = np.linspace(V0_INITIAL, 0.0, 10)  # Rampa de velocidad
        
        # ===== Inicialización lineal de puntos internos =====
        # Los puntos internos tienen una distribución lineal en Y para mejor convergencia
        for j in range(NY):
            for i in range(NX):
                if self._is_unknown(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1))  # Interpolación lineal
        return V_matrix

    def assemble_system(self, V_current):
        """
        Ensambla la matriz Jacobiana y el vector RHS para la iteración actual de Newton-Raphson.
        
        La ecuación discretizada es:
        F(V) = 4*V_c - (V_r + V_l + V_u + V_d) + 4*V_c*(V_r - V_l) + 4*Vy*(V_u - V_d) = 0
        
        donde:
        - V_c: velocidad en el centro
        - V_r, V_l: velocidades derecha e izquierda (eje X)
        - V_u, V_d: velocidades arriba y abajo (eje Y)
        
        Args:
            V_current: Campo de velocidad actual (matriz NY×NX)
        
        Returns:
            J: Matriz Jacobiana (derivadas parciales de F respecto a V)
            
        """
        J = lil_matrix((self.n_unknowns, self.n_unknowns))  # Matriz dispersa eficiente
        rhs = np.zeros(self.n_unknowns)  # Vector lado derecho
        m = 0  # Contador de ecuaciones
        
        for j in range(NY):
            for i in range(NX):
                if not self._is_unknown(i, j): continue  # Saltar condiciones de contorno
                
                # ===== Stencil de 5 puntos (valores vecinos) =====
                V_c = V_current[j, i]  # Centro
                V_r = V_current[j, i+1]  # Derecha
                V_l = V_current[j, i-1]  # Izquierda
                V_u = V_current[j+1, i]  # Arriba
                V_d = V_current[j-1, i]  # Abajo
                
                
                # Incluye términos de difusión y advección
                rhs[m] = -(4*V_c - (V_r+V_l+V_u+V_d) + 4*V_c*(V_r-V_l) + 4*VY_TEST*(V_u-V_d))
                
                # ===== Jacobiano: diagonal principal (derivada respecto a V_c) =====
                J[m, m] = 4 + 4*V_r - 4*V_l
                
                # ===== Jacobiano: elementos fuera de la diagonal =====
                # Obtener índices lineales de vecinos
                n_r = self._get_linear_index(i+1, j)  # Índice del vecino derecho
                n_l = self._get_linear_index(i-1, j)  # Índice del vecino izquierdo
                n_u = self._get_linear_index(i, j+1)  # Índice del vecino superior
                n_d = self._get_linear_index(i, j-1)  # Índice del vecino inferior
                
                # Llenar elementos del Jacobiano si el vecino es incógnita
                if n_r is not None: J[m, n_r] = -1 + 4*V_c  # ∂F/∂V_r
                if n_l is not None: J[m, n_l] = -1 - 4*V_c  # ∂F/∂V_l
                if n_u is not None: J[m, n_u] = -1 + 4*VY_TEST  # ∂F/∂V_u
                if n_d is not None: J[m, n_d] = -1 - 4*VY_TEST  # ∂F/∂V_d
                m += 1
        
        return J.tocsr(), rhs  # Convertir a formato CSR (más eficiente para operaciones)

    def _calculate_spectral_radius(self, J, method='jacobi'):
        """
        Calcula el radio espectral de la matriz de iteración para analizar convergencia.
        
        El radio espectral ρ(T) < 1 garantiza convergencia del método iterativo.
        Cuanto menor sea ρ(T), más rápida es la convergencia.
        
        Args:
            J: Matriz Jacobiana
            method: 'jacobi' o 'gauss-seidel'
        
        Returns:
            Radio espectral (eigenvalor de mayor magnitud de la matriz de iteración)
        """
        try:
            D_diag = J.diagonal()  # Diagonal de J
            if np.any(np.abs(D_diag) < 1e-12): return float('nan')  # Evitar división por cero
            
            # Descomponer J = D + L + U (Diagonal + Triangular inferior + Triangular superior)
            D = diags(D_diag)  # Parte diagonal
            L = J.tril(k=-1)  # Parte triangular inferior (sin diagonal)
            U = J.triu(k=1)  # Parte triangular superior (sin diagonal)
            
            # Construir matriz de iteración T según el método
            if method == 'jacobi':
                # T_jacobi = -D^(-1) * (L + U)
                D_inv = diags(1.0 / D_diag)
                T = -D_inv @ (L + U)
            elif method == 'gauss-seidel':
                # T_gs = -(D + L)^(-1) * U
                D_plus_L = csc_matrix(D + L)
                T_U = spsolve(D_plus_L, U)
                T = -T_U
            else:
                return float('nan')
            
            # Calcular eigenvalor de mayor magnitud
            eigenvalues = eigs(T, k=1, which='LM', return_eigenvectors=False)
            return np.abs(eigenvalues[0])  # Radio espectral
        except Exception:
            return float('nan')

    def _richardson(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER, omega=0.1):
        """
        Método de Richardson para resolver Jx = rhs.
        
        Iteración: x^(k+1) = x^(k) + ω * (rhs - J*x^(k))
        donde ω es el parámetro de relajación.
        
        Args:
            J: Matriz del sistema
            rhs: Lado derecho del sistema
            tol: Tolerancia de convergencia
            max_iter: Número máximo de iteraciones
            omega: Parámetro de relajación (controla tamaño del paso)
        
        Returns:
            Solución aproximada x
        """
        x = np.zeros_like(rhs)  # Inicialización en cero
        for _ in range(max_iter):
            r = rhs - J @ x  
            if norm(r) < tol: break  # Verificar convergencia
            x += omega * r  # Actualizar solución
        return x

    def _jacobi(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        """
        Método iterativo de Jacobi para resolver Jx = rhs.
        
        Descompone J = D + (L + U) y resuelve:
        x^(k+1) = D^(-1) * (rhs - (L + U)*x^(k))
        
        Características:
        - Fácil de paralelizar (cada componente se actualiza independientemente)
        - Convergencia garantizada si J es estrictamente diagonal dominante
        - Converge más lento que Gauss-Seidel en general
        
        Args:
            J: Matriz del sistema
            rhs: Lado derecho del sistema
            tol: Tolerancia de convergencia
            max_iter: Número máximo de iteraciones
        
        Returns:
            Solución aproximada x
        """
        x = np.zeros_like(rhs, dtype=float)
        D = J.diagonal().astype(float)  # Extraer diagonal
        D[np.abs(D) < 1e-12] = 1e-12  # Evitar división por cero
        L_plus_U = J - diags(D)  # L + U (parte fuera de la diagonal)
        D_inv = 1.0 / D  # Inversa de la diagonal
        
        for _ in range(max_iter):
            x_new = ((rhs - L_plus_U @ x) * D_inv)  # Fórmula de Jacobi
            if norm(x_new - x) < tol: return x_new  # Verificar convergencia
            x = x_new  # Actualizar para siguiente iteración
        return x

    def _gauss_seidel(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        """
        Método iterativo de Gauss-Seidel para resolver Jx = rhs.
        
        Similar a Jacobi pero usa valores actualizados inmediatamente:
        x_i^(k+1) = (rhs_i - Σ(j<i) J_ij*x_j^(k+1) - Σ(j>i) J_ij*x_j^(k)) / J_ii
        
        Características:
        - Converge aproximadamente el doble de rápido que Jacobi
        - No paralelizable fácilmente (actualizaciones secuenciales)
        - Requiere menos memoria (no necesita almacenar x_old completo)
        
        Args:
            J: Matriz del sistema
            rhs: Lado derecho del sistema
            tol: Tolerancia de convergencia
            max_iter: Número máximo de iteraciones
        
        Returns:
            Solución aproximada x
        """
        x = np.zeros_like(rhs, dtype=float)
        n = self.n_unknowns
        diag = np.array([J[i, i] for i in range(n)], dtype=float)  # Diagonal
        diag[np.abs(diag) < 1e-12] = 1e-12  # Evitar división por cero
        J_csr = J.tocsr()  # Formato CSR para acceso eficiente por filas
        
        for _ in range(max_iter):
            x_old = x.copy()  # Guardar para verificar convergencia
            for i in range(n):
                # Acceder a elementos no ceros de la fila i
                row_start, row_end = J_csr.indptr[i], J_csr.indptr[i+1]
                cols, data = J_csr.indices[row_start:row_end], J_csr.data[row_start:row_end]
                
                sigma = 0.0
                for col, val in zip(cols, data):
                    if col != i:  # Saltar elemento diagonal
                        # Usar x actualizado si col < i, sino usar x_old
                        sigma += val * (x[col] if col < i else x_old[col])
                
                x[i] = (rhs[i] - sigma) / diag[i]  # Actualizar componente i
                
            if norm(x - x_old) < tol: return x  # Verificar convergencia
        return x

    def _gradient_descent(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        """
        Método de descenso de gradiente para resolver Jx = rhs.
        
        Convierte el sistema en un problema de minimización:
        min ||Jx - rhs||^2  =>  min x^T*A*x - 2*b^T*x  donde A = J^T*J, b = J^T*rhs
        
        
        
        Args:
            J: Matriz del sistema
            rhs: Lado derecho del sistema
            tol: Tolerancia de convergencia
            max_iter: Número máximo de iteraciones
        
        Returns:
            Solución aproximada x
        """
        # Formar sistema de ecuaciones normales (A es simétrica definida positiva)
        A = J.T @ J + 1e-8 * identity(self.n_unknowns)  # Regularización para estabilidad
        b = J.T @ rhs
        x = np.zeros_like(rhs, dtype=float)
        
        for _ in range(max_iter):
            r = b - A @ x  # 
            if norm(r) < tol: return x  # Convergencia alcanzada
            
            # Calcular tamaño de paso óptimo: α = (r^T*r) / (r^T*A*r)
            denom = r.T @ A @ r
            alpha = (r.T @ r) / denom if abs(denom) > 1e-20 else 1e-6
            x += alpha * r  # Actualizar en dirección del gradiente
        return x

    def _conjugate_gradient(self, J, rhs, tol=LINEAR_TOLERANCE, max_iter=MAX_LINEAR_ITER):
        """
        Método de gradiente conjugado para resolver Jx = rhs.
        
        Similar a descenso de gradiente pero usa direcciones conjugadas ortogonales,
        lo que garantiza convergencia en a lo más n iteraciones (n = dimensión).
        
        Ventajas:
        - Converge mucho más rápido que descenso de gradiente
        - No requiere almacenar toda la matriz A
        - Ideal para sistemas grandes y dispersos
        
        Args:
            J: Matriz del sistema
            rhs: Lado derecho del sistema
            tol: Tolerancia de convergencia
            max_iter: Número máximo de iteraciones
        
        Returns:
            Solución aproximada x
        """
        # Formar sistema de ecuaciones normales
        A = J.T @ J + 1e-8 * identity(self.n_unknowns)  # A debe ser simétrica def. positiva
        b = J.T @ rhs
        x = np.zeros_like(rhs, dtype=float)
        r = b - A @ x  # 
        p = r.copy()  # Dirección de búsqueda inicial
        rs_old = float(r.T @ r)  # 
        
        for _ in range(max_iter):
            Ap = A @ p  # Producto matriz-vector
            denom = float(p.T @ Ap)  # p^T * A * p
            if abs(denom) < 1e-20: break  # Evitar división por cero
            
            alpha = rs_old / denom  # Tamaño de paso óptimo
            x += alpha * p  # Actualizar solución
            r -= alpha * Ap  # 
            rs_new = float(r.T @ r)  #
            
            if np.sqrt(rs_new) < tol: return x  # Convergencia alcanzada
            
            # Calcular nueva dirección conjugada
            p = r + (rs_new / rs_old) * p  # Fórmula de Fletcher-Reeves
            rs_old = rs_new
        return x

    def solve_linear_system(self, J, rhs, method):
        """
        Resuelve el sistema lineal Jx = rhs usando el método especificado.
        
        Args:
            J: Matriz Jacobiana del sistema
            rhs: Lado derecho del sistema
            method: Método iterativo a usar:
                - 'richardson': Método de Richardson (simple pero lento)
                - 'jacobi': Método de Jacobi (paralelizable)
                - 'gauss-seidel': Gauss-Seidel (más rápido que Jacobi)
                - 'gradient-descent': Descenso de gradiente
                - 'conjugate-gradient': Gradiente conjugado (más eficiente)
        
        Returns:
            Solución del sistema lineal
        """
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
        """
        Resuelve el problema de flujo usando el método de Newton-Raphson.
        
        Algoritmo:
        1. Ensamblar sistema Jacobiano J y vector rhs
        2. Resolver J*ΔV = rhs usando método iterativo especificado
        3. Actualizar campo de velocidad: V^(k+1) = V^(k) + ω*ΔV
        4. Repetir hasta convergencia o máximo de iteraciones
        
        Args:
            linear_solver_method: Método para resolver sistemas lineales en cada iteración
            theoretical_analysis: Si True, calcula número de condición y radio espectral
        
        Returns:
            Diccionario con:
                - solution: Campo de velocidad final
                - time: Tiempo de ejecución
                - converged: Booleano indicando convergencia
                - iterations: Número de iteraciones realizadas
                - has_nan: Booleano indicando si hay valores NaN
                - cond_history: Historial de números de condición (si theoretical_analysis=True)
                - rs_history: Historial de radios espectrales (si theoretical_analysis=True)
        """
        V_matrix = self.V_k.copy()  # Copiar campo inicial
        start_time = time.time()  # Iniciar cronómetro
        converged = False
        final_k = 0
        cond_history, rs_history = [], []  # Historiales para análisis teórico
        
        # ===== Ciclo principal de Newton-Raphson =====
        for k in range(1, MAX_ITER + 1):
            # Paso 1: Ensamblar sistema linealizado
            J, rhs = self.assemble_system(V_matrix)
            
            # Análisis teórico (opcional, costoso computacionalmente)
            if theoretical_analysis:
                try:
                    cond_num = cond(J.toarray())  # Número de condición de J
                    cond_history.append(cond_num)
                    print(f"    Iteration {k}: Condition Number = {cond_num:.2e}")
                except Exception:
                    cond_history.append(float('inf'))
                
                # Radio espectral (solo para Jacobi y Gauss-Seidel)
                if linear_solver_method in ['jacobi', 'gauss-seidel']:
                    rs = self._calculate_spectral_radius(J, method=linear_solver_method)
                    rs_history.append(rs)
                    print(f"    Iteration {k}: Spectral Radius ({linear_solver_method}) = {rs:.4f}")
            
            # Paso 2: Resolver sistema lineal J*ΔV = rhs
            Delta_V = self.solve_linear_system(J, rhs, method=linear_solver_method)
            
            # Verificar divergencia o valores inválidos
            if np.isnan(Delta_V).any() or np.max(np.abs(Delta_V)) > 100:
                print(f"Divergence or NaN detected in {linear_solver_method}.")
                final_k = k
                break
            
            # Paso 3: Actualizar campo de velocidad con relajación
            V_new = V_matrix.copy()
            m = 0
            for j in range(NY):
                for i in range(NX):
                    if self._is_unknown(i, j):
                        V_new[j, i] += 0.6 * Delta_V[m]  # Factor de relajación 0.6 para estabilidad
                        m += 1
            
            # Paso 4: Asegurar que velocidades estén en rango físico [0, V0]
            V_matrix = np.clip(V_new, 0, V0_INITIAL)
            
            # Verificar convergencia
            max_change = np.max(np.abs(Delta_V))
            print(f"Iteration {k} ({linear_solver_method}): Max change = {max_change:.8f}")
            
            if max_change < TOLERANCE:
                print(f"Convergence reached in {k} iterations.")
                converged = True
                final_k = k
                break
            final_k = k
        
        # Retornar resultados completos
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
    """
    Aplica interpolación bicúbica para visualización suave del campo de velocidad.
    
    La interpolación bicúbica es una extensión 2D de la interpolación cúbica.
    Usa polinomios cúbicos en ambas direcciones (X e Y) simultáneamente,
    produciendo superficies más suaves que métodos separables 1D.
    
    Args:
        V_low_res: Campo de velocidad de baja resolución (NY × NX)
    
    Returns:
        Campo de velocidad interpolado de alta resolución (NY_HIGH × NX_HIGH)
        donde la resolución se incrementa por factor INTERP_RESOLUTION
    """
    INTERP_RESOLUTION = 10  # Factor de incremento de resolución
    NY_HIGH = (NY - 1) * INTERP_RESOLUTION + 1  # Nueva cantidad de puntos en Y
    NX_HIGH = (NX - 1) * INTERP_RESOLUTION + 1  # Nueva cantidad de puntos en X
    
    # Crear grillas de coordenadas
    x_low = np.linspace(0, NX - 1, NX)  # Coordenadas X originales
    y_low = np.linspace(0, NY - 1, NY)  # Coordenadas Y originales
    x_high = np.linspace(0, NX - 1, NX_HIGH)  # Coordenadas X de alta resolución
    y_high = np.linspace(0, NY - 1, NY_HIGH)  # Coordenadas Y de alta resolución
    
    # Crear función de interpolación bicúbica (kx=3, ky=3 = cúbico en ambas direcciones)
    interp_func = interpolate.RectBivariateSpline(y_low, x_low, V_low_res, kx=3, ky=3)
    return interp_func(y_high, x_high)  # Evaluar en grilla de alta resolución

def plot_solution(V_final, vy_value, method_name):
    """
    Genera gráfica del campo de velocidad con interpolación bicúbica.
    
    Crea una visualización suave y de alta resolución del flujo,
    mostrando cómo varía la velocidad en el dominio y alrededor de obstáculos.
    
    Args:
        V_final: Campo de velocidad solución (resolución original)
        vy_value: Valor de velocidad vertical (no usado actualmente)
        method_name: Nombre del método usado (para título y nombre de archivo)
    """
    V_interpolated = interpolate_bicubic(V_final)  # Interpolar a alta resolución
    fig, ax = plt.subplots(figsize=(18, 8))  # Crear figura grande
    
    # Mostrar campo de velocidad como mapa de calor
    cax = ax.imshow(V_interpolated, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)
    
    # Agregar barra de colores
    fig.colorbar(cax, label='Velocity (Vx) [Interpolated]')
    
    # Dibujar obstáculos como rectángulos rojos
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red'))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red'))
    
    # Configurar etiquetas y título
    ax.set_title(f'Solution: {method_name.upper()} (Bicubic Spline)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    
    # Guardar figura
    filename = os.path.join(OUTPUT_DIR, f"result_spline_{method_name}.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: '{filename}'")

def plot_solution_no_spline(V_final, vy_value, method_name):
    """
    Genera gráfica del campo de velocidad sin interpolación (valores crudos).
    
    Muestra la solución numérica exacta en la malla computacional,
    con valores numéricos anotados en cada celda para inspección detallada.
    
    Args:
        V_final: Campo de velocidad solución (resolución original)
        vy_value: Valor de velocidad vertical (no usado actualmente)
        method_name: Nombre del método usado (para título y nombre de archivo)
    """
    fig, ax = plt.subplots(figsize=(22, 10))  # Figura extra grande para mostrar valores
    
    # Mostrar campo de velocidad sin interpolación
    cax = ax.imshow(V_final, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL, interpolation='nearest')
    
    # Barra de colores
    fig.colorbar(cax, label='Velocity (Vx)', shrink=0.8)
    
    # Dibujar obstáculos semitransparentes
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX-VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX-VIGA_INF_Y_MIN, color='red', alpha=0.7, linewidth=2))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX-VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX-VIGA_SUP_Y_MIN, color='red', alpha=0.7, linewidth=2))
    
    # Calcular tamaño de fuente adaptativo
    fontsize = max(4, min(12, 300 / NX))  # Más pequeño para mallas grandes
    
    # Anotar valores numéricos en cada celda
    for j in range(NY):
        for i in range(NX):
            x_pos = i + 0.5  # Centro de la celda
            y_pos = j + 0.5
            velocity = V_final[j, i]
            # Usar color de texto que contraste con el fondo
            text_color = 'white' if velocity < 0.5 else 'black'
            ax.text(x_pos, y_pos, f'{V_final[j, i]:.2f}', 
                   ha='center', va='center', fontsize=fontsize, color=text_color, 
                   weight='bold')
    
    # Configurar ejes y grilla
    ax.set_xticks(np.arange(0, NX + 1, 5))  # Marcas cada 5 unidades
    ax.set_yticks(np.arange(0, NY + 1, 1))  # Marcas en cada unidad vertical
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)  # Grilla sutil
    
    # Títulos y etiquetas
    ax.set_title(f'Solution: {method_name.upper()} (No Spline - Raw Values)', fontsize=14, weight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    plt.tight_layout()
    
    # Guardar con alta resolución
    filename = os.path.join(OUTPUT_DIR, f"result_no_spline_{method_name}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: '{filename}'")

def analyze_results(results):
    """
    Analiza y compara el rendimiento de todos los métodos iterativos probados.
    
    Imprime una tabla comparativa mostrando:
    - Convergencia (sí/no)
    - Tiempo de ejecución
    - Número de iteraciones
    - Análisis teórico (número de condición, radio espectral)
    
    Identifica el mejor método basado en tiempo de convergencia.
    
    Args:
        results: Diccionario con resultados de cada método
                {nombre_metodo: {"converged", "time", "iterations", ...}}
    """
    print("\n--- ITERATIVE METHODS COMPARISON ---")
    print(f"{'Method':<20} | {'Converged':<10} | {'Time (s)':<12} | {'Iterations':<12} | {'Analysis'}")
    print("-" * 80)
    
    # Filtrar solo métodos que convergieron exitosamente
    valid_methods = {m: d for m, d in results.items() if d["converged"] and not d["has_nan"]}
    
    # Imprimir estadísticas de cada método
    for method, data in results.items():
        analysis_str = ""
        # Agregar datos de análisis teórico si están disponibles
        if data["cond_history"]: 
            analysis_str += f"Final Cond: {data['cond_history'][-1]:.2e}"
        if data["rs_history"]: 
            analysis_str += f" | Final RS: {data['rs_history'][-1]:.4f}"
        
        # Fila de la tabla
        print(f"{method:<20} | {'Yes' if data['converged'] else 'No':<10} | {data['time']:<12.4f} | {data['iterations']:<12} | {analysis_str}")
    print("-" * 80)
    
    # Determinar mejor método
    if not valid_methods:
        print("\nResult: No iterative method converged successfully.")
    else:
        # Mejor método = más rápido entre los que convergieron
        best_method = min(valid_methods, key=lambda m: valid_methods[m]['time'])
        print(f"\nBest practical method: '{best_method.upper()}' (fastest convergence).")

# ========== Programa Principal ==========
if __name__ == '__main__':
    """
    Ejecuta simulaciones CFD comparando diferentes m\u00e9todos iterativos.
    
    Proceso:
    1. Limpiar gr\u00e1ficas anteriores
    2. Ejecutar simulaciones con cada m\u00e9todo iterativo
    3. Analizar y comparar resultados
    4. Generar visualizaciones
    """
    
    # Verificar dependencias
    try:
        from scipy import interpolate
    except ImportError:
        print("Error: 'scipy' is required for Cubic Spline interpolation.")
        exit()
    
    # Paso 1: Limpiar archivos anteriores
    print("Cleaning previous plots...")
    clear_previous_plots()

    # M\u00e9todos a probar
    methods_to_test = ['jacobi', 'gauss-seidel', 'richardson', 'gradient-descent', 'conjugate-gradient']
    methods_for_deep_analysis = ['jacobi', 'gauss-seidel']  # Solo estos se analizan en detalle (costoso)
    
    results = {}  # Almacenar resultados de cada m\u00e9todo
    
    # Paso 2: Ejecutar simulaciones con cada m\u00e9todo
    for method in methods_to_test:
        print(f"\n--- STARTING SIMULATION: {method.upper()} ---")
        run_analysis = method in methods_for_deep_analysis
        if run_analysis: 
            print("   (Deep theoretical analysis enabled)")
        
        # Crear solver y resolver
        solver = NewtonRaphsonFlowSolver()
        results[method] = solver.solve(linear_solver_method=method, theoretical_analysis=run_analysis)
    
    # Paso 3: Analizar resultados comparativamente
    analyze_results(results)
    
    # Paso 4: Generar visualizaciones
    print("\n--- GENERATING PLOTS ---")
    
    for method, data in results.items():
        if data["converged"] and not data["has_nan"]: 
            # Solo graficar soluciones v\u00e1lidas
            plot_solution(data["solution"], VY_TEST, method)  # Con interpolaci\u00f3n bic\u00fabica
            plot_solution_no_spline(data["solution"], VY_TEST, method)  # Sin interpolaci\u00f3n
        else:
            print(f"Skipping plot for {method.upper()}: Did not converge or contains NaN.")