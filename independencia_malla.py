"""
Análisis de Independencia de Malla
===================================
Este script evalúa la sensibilidad de los resultados a la resolución
de la malla, probando con diferentes tamaños de grilla.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Importar componentes necesarios
sys.path.insert(0, os.path.dirname(__file__))
from scipy.sparse import lil_matrix, diags, identity
from numpy.linalg import norm

plt.style.use('dark_background')

# Parámetros globales base
VY_TEST = 0.1
V0_INITIAL = 1.0
TOLERANCE = 1e-8
MAX_ITER = 500
LINEAR_TOLERANCE = 1e-4
MAX_LINEAR_ITER = 5000

class SolverParametrico:
    """Solver que acepta parámetros de malla variables."""
    
    def __init__(self, ny, nx):
        self.NY = ny
        self.NX = nx
        
        # Escalar obstáculos proporcionalmente
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

def analizar_independencia_malla():
    """Ejecuta análisis de independencia de malla."""
    print("=" * 80)
    print("ANÁLISIS DE INDEPENDENCIA DE MALLA")
    print("=" * 80)
    
    # Configuraciones de malla a probar
    configuraciones = [
        (5, 50, "Gruesa (Original)"),
        (10, 100, "Media"),
        (15, 150, "Fina")
    ]
    
    resultados = []
    
    for ny, nx, nombre in configuraciones:
        print(f"\n{'─' * 80}")
        print(f"Probando malla: {nombre} ({ny} × {nx} = {ny*nx} celdas)")
        print(f"{'─' * 80}")
        
        solver = SolverParametrico(ny, nx)
        print(f"  Incógnitas: {solver.N_INCÓGNITAS}")
        print(f"  Ejecutando simulación...")
        
        resultado = solver.solve()
        
        if resultado is None or not resultado['converged']:
            print(f"  ❌ No convergió")
            resultados.append(None)
            continue
        
        V_sol = resultado['solution']
        
        # Métricas clave
        v_max = np.max(V_sol)
        v_promedio = np.mean(V_sol[V_sol > 0])
        
        # Punto de monitoreo (centro del canal, antes del primer obstáculo)
        i_monitor = int(0.3 * nx)
        j_monitor = int(0.5 * ny)
        v_monitor = V_sol[j_monitor, i_monitor]
        
        info = {
            'nombre': nombre,
            'ny': ny,
            'nx': nx,
            'v_max': v_max,
            'v_promedio': v_promedio,
            'v_monitor': v_monitor,
            'iteraciones': resultado['iterations'],
            'tiempo': resultado['time']
        }
        
        resultados.append(info)
        
        print(f"  ✓ Convergió en {resultado['iterations']} iteraciones")
        print(f"  ✓ Tiempo: {resultado['time']:.2f}s")
        print(f"  ✓ V_max: {v_max:.6f}")
        print(f"  ✓ V_promedio: {v_promedio:.6f}")
        print(f"  ✓ V_monitor: {v_monitor:.6f}")
    
    # Generar reporte
    print("\n" + "=" * 80)
    print("REPORTE DE CONVERGENCIA DE MALLA")
    print("=" * 80)
    
    reporte = f"""
{'=' * 80}
ANÁLISIS DE INDEPENDENCIA DE MALLA
{'=' * 80}

1. CONFIGURACIONES PROBADAS
"""
    
    for i, res in enumerate(resultados):
        if res is None:
            continue
        reporte += f"""
   Malla {i+1}: {res['nombre']}
   - Resolución: {res['ny']} × {res['nx']} = {res['ny']*res['nx']} celdas
   - Iteraciones: {res['iteraciones']}
   - Tiempo: {res['tiempo']:.2f}s
   - V_max: {res['v_max']:.6f} m/s
   - V_promedio: {res['v_promedio']:.6f} m/s
   - V_monitor: {res['v_monitor']:.6f} m/s
"""
    
    # Calcular diferencias
    if len([r for r in resultados if r is not None]) >= 2:
        reporte += "\n2. ANÁLISIS DE CONVERGENCIA\n"
        
        res_validos = [r for r in resultados if r is not None]
        
        for i in range(len(res_validos) - 1):
            r1 = res_validos[i]
            r2 = res_validos[i+1]
            
            diff_max = abs(r2['v_max'] - r1['v_max']) / r1['v_max'] * 100
            diff_prom = abs(r2['v_promedio'] - r1['v_promedio']) / r1['v_promedio'] * 100
            diff_mon = abs(r2['v_monitor'] - r1['v_monitor']) / r1['v_monitor'] * 100
            
            reporte += f"""
   {r1['nombre']} → {r2['nombre']}:
   - Cambio en V_max:      {diff_max:.2f}%
   - Cambio en V_promedio: {diff_prom:.2f}%
   - Cambio en V_monitor:  {diff_mon:.2f}%
"""
        
        # Conclusión
        ultima_diff = diff_mon
        if ultima_diff < 1.0:
            conclusion = "✓ EXCELENTE: Diferencia < 1% → Solución independiente de malla"
        elif ultima_diff < 5.0:
            conclusion = "✓ BUENA: Diferencia < 5% → Convergencia aceptable"
        else:
            conclusion = "⚠ ADVERTENCIA: Diferencia > 5% → Se requiere mayor refinamiento"
        
        reporte += f"""
3. CONCLUSIÓN

   {conclusion}
   
   Última diferencia observada: {ultima_diff:.2f}%
   
   RECOMENDACIÓN:
   {'La malla actual es suficiente para este análisis académico.' if ultima_diff < 5 else 'Se recomienda usar una malla más fina para resultados de producción.'}

"""
    
    reporte += f"""
4. COSTO COMPUTACIONAL

   Escalamiento observado:
"""
    
    for res in [r for r in resultados if r is not None]:
        costo_por_celda = res['tiempo'] / (res['ny'] * res['nx'])
        reporte += f"   - {res['nombre']}: {costo_por_celda*1000:.2f} ms/celda\n"
    
    reporte += "\n" + "=" * 80 + "\n"
    
    print(reporte)
    
    # Guardar reporte
    ruta_reporte = os.path.join('analisis_avanzado', 'reporte_independencia_malla.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(f"✅ Reporte guardado en: {ruta_reporte}\n")
    
    # Generar gráfico de convergencia
    if len([r for r in resultados if r is not None]) >= 2:
        generar_grafico_convergencia([r for r in resultados if r is not None])

def generar_grafico_convergencia(resultados):
    """Genera gráfico de convergencia de malla."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    celdas = [r['ny'] * r['nx'] for r in resultados]
    v_max = [r['v_max'] for r in resultados]
    v_monitor = [r['v_monitor'] for r in resultados]
    nombres = [r['nombre'] for r in resultados]
    
    # Gráfico 1: Convergencia de velocidades
    ax1.plot(celdas, v_max, 'o-', label='V_max', linewidth=2, markersize=8, color='cyan')
    ax1.plot(celdas, v_monitor, 's-', label='V_monitor', linewidth=2, markersize=8, color='yellow')
    ax1.set_xlabel('Número de Celdas', fontsize=12)
    ax1.set_ylabel('Velocidad (m/s)', fontsize=12)
    ax1.set_title('Convergencia de Malla', fontsize=14, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Tiempo de cómputo
    tiempos = [r['tiempo'] for r in resultados]
    ax2.bar(nombres, tiempos, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('Tiempo (s)', fontsize=12)
    ax2.set_title('Costo Computacional', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    ruta = os.path.join('analisis_avanzado', 'convergencia_malla.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Gráfico guardado: {ruta}\n")

if __name__ == '__main__':
    analizar_independencia_malla()
