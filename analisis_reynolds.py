"""
Análisis del Número de Reynolds
================================
Este script calcula el número de Reynolds para caracterizar el régimen de flujo
según lo solicitado en el problema original.
"""

import numpy as np
import sys
import os

# Importar la clase del código existente
sys.path.insert(0, os.path.dirname(__file__))
from campo_velocidadesV4 import FlujoNewtonRaphson, NY, NX, V0_INITIAL, VY_TEST

# Parámetros físicos (según el problema)
RHO = 1.0  # kg/m³ (densidad del fluido)
NU = 1.0   # m²/s (viscosidad cinemática)
MU = NU * RHO  # Pa·s (viscosidad dinámica)

def calcular_reynolds():
    """
    Calcula el número de Reynolds del flujo.
    Re = (ρ * V * L) / μ
    donde:
    - V = velocidad característica
    - L = longitud característica
    """
    print("=" * 80)
    print("ANÁLISIS DEL NÚMERO DE REYNOLDS")
    print("=" * 80)
    print("\n📊 Ejecutando simulación para obtener campo de velocidades...\n")
    
    # Ejecutar simulación
    solver = FlujoNewtonRaphson()
    resultado = solver.solve(linear_solver_method='conjugate-gradient', analisis_teorico=False)
    
    if not resultado['converged']:
        print("❌ Error: La simulación no convergió. No se puede calcular Reynolds.")
        return
    
    V_solution = resultado['solution']
    
    # Calcular velocidades características
    V_max = np.max(V_solution)
    V_promedio = np.mean(V_solution[V_solution > 0])  # Promedio excluyendo obstáculos
    V_entrada = V0_INITIAL
    
    # Longitudes características
    L_altura = NY  # Altura del canal
    L_longitud = NX  # Longitud del canal
    
    # Calcular diferentes números de Reynolds
    Re_max = (RHO * V_max * L_altura) / MU
    Re_promedio = (RHO * V_promedio * L_altura) / MU
    Re_entrada = (RHO * V_entrada * L_altura) / MU
    
    # Generar reporte
    reporte = f"""
{'=' * 80}
REPORTE DE ANÁLISIS DE REYNOLDS
{'=' * 80}

1. PARÁMETROS FÍSICOS
   - Densidad (ρ):              {RHO} kg/m³
   - Viscosidad cinemática (ν): {NU} m²/s
   - Viscosidad dinámica (μ):   {MU} Pa·s

2. GEOMETRÍA
   - Altura del canal (L):      {L_altura} unidades
   - Longitud del canal:        {L_longitud} unidades
   - Resolución de malla:       {NY} × {NX}

3. VELOCIDADES CARACTERÍSTICAS
   - Velocidad de entrada:      {V_entrada:.4f} m/s
   - Velocidad máxima:          {V_max:.4f} m/s
   - Velocidad promedio:        {V_promedio:.4f} m/s
   - Componente vertical (Vy):  {VY_TEST:.4f} m/s

4. NÚMEROS DE REYNOLDS CALCULADOS
   
   Re (basado en V_entrada) = {Re_entrada:.2f}
   Re (basado en V_max)     = {Re_max:.2f}
   Re (basado en V_promedio)= {Re_promedio:.2f}

5. INTERPRETACIÓN FÍSICA

   Régimen de Flujo:
   """
    
    # Clasificación del régimen
    Re_ref = Re_entrada  # Usamos la velocidad de entrada como referencia
    
    if Re_ref < 2000:
        regimen = "LAMINAR"
        descripcion = """
   ✓ Re < 2000 → FLUJO LAMINAR
   
   El flujo es ordenado y predecible. Las capas de fluido se deslizan
   suavemente unas sobre otras sin mezclarse. Este régimen justifica:
   
   - El uso de métodos iterativos para resolver las ecuaciones
   - La convergencia relativamente rápida de los solvers
   - La estabilidad numérica observada en la simulación
   
   NOTA: En este régimen, los términos no lineales de Navier-Stokes
   tienen una contribución pequeña pero no despreciable, por lo que
   el enfoque de Newton-Raphson es apropiado.
        """
    elif Re_ref < 4000:
        regimen = "TRANSICIÓN"
        descripcion = """
   ⚠ 2000 < Re < 4000 → FLUJO EN TRANSICIÓN
   
   El flujo está en una zona intermedia entre laminar y turbulento.
   Pueden aparecer pequeñas perturbaciones que crecen o se amortiguan.
   
   - Mayor sensibilidad a las condiciones de frontera
   - Posible aparición de inestabilidades locales
   - Requiere mayor cuidado en la discretización espacial
        """
    else:
        regimen = "TURBULENTO"
        descripcion = """
   ⚠ Re > 4000 → FLUJO TURBULENTO
   
   El flujo es caótico y presenta remolinos a múltiples escalas.
   
   ADVERTENCIA: La simulación actual NO incluye modelos de turbulencia
   (como k-ε o LES). Los resultados deben interpretarse con precaución.
   
   Para este régimen se recomienda:
   - Usar modelos de turbulencia apropiados
   - Aumentar significativamente la resolución de malla
   - Considerar simulaciones transitorias (no estacionarias)
        """
    
    reporte += descripcion
    
    reporte += f"""

6. VALIDACIÓN DEL ENFOQUE NUMÉRICO

   El problema original sugería verificar si los términos no lineales
   pueden despreciarse. Basándonos en Re = {Re_ref:.2f}:
   
   """
    
    if Re_ref < 1:
        reporte += """   ✓ Re << 1: Los términos no lineales son despreciables.
     Se podría usar un solver lineal directo (Stokes flow).
   """
    elif Re_ref < 100:
        reporte += """   ✓ Re < 100: Los términos no lineales son pequeños pero presentes.
     El enfoque de Newton-Raphson es apropiado y eficiente.
   """
    else:
        reporte += """   ⚠ Re > 100: Los términos no lineales son significativos.
     El enfoque de Newton-Raphson es NECESARIO (no opcional).
     No se puede linealizar el problema sin perder precisión.
   """
    
    reporte += f"""

7. CONCLUSIONES

   - Régimen de flujo: {regimen}
   - Número de Reynolds de referencia: {Re_ref:.2f}
   - Enfoque numérico: {'Apropiado' if Re_ref < 4000 else 'Requiere mejoras'}
   - Convergencia observada: {'Sí' if resultado['converged'] else 'No'}
   - Iteraciones necesarias: {resultado['iterations']}

{'=' * 80}
"""
    
    # Imprimir en consola
    print(reporte)
    
    # Guardar en archivo
    ruta_reporte = os.path.join('analisis_avanzado', 'reporte_reynolds.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print(f"\n✅ Reporte guardado en: {ruta_reporte}\n")
    
    return {
        'Re_entrada': Re_entrada,
        'Re_max': Re_max,
        'Re_promedio': Re_promedio,
        'regimen': regimen
    }

if __name__ == '__main__':
    calcular_reynolds()
