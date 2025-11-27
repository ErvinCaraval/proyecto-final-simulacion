"""
Reynolds Number Analysis
========================
Calculates the Reynolds number to characterize the flow regime.
"""

import numpy as np
import sys
import os

# Import the solver class
sys.path.insert(0, os.path.dirname(__file__))
from campo_velocidadesV4 import NewtonRaphsonFlowSolver, NY, NX, V0_INITIAL, VY_TEST

# Physical Parameters
RHO = 1.0  # kg/m^3
NU = 1.0   # m^2/s
MU = NU * RHO  # Pa*s

def calculate_reynolds():
    """
    Calculates the Reynolds number: Re = (rho * V * L) / mu
    """
    print("=" * 80)
    print("REYNOLDS NUMBER ANALYSIS")
    print("=" * 80)
    print("\nRunning simulation to obtain velocity field...\n")
    
    solver = NewtonRaphsonFlowSolver()
    result = solver.solve(linear_solver_method='conjugate-gradient', theoretical_analysis=False)
    
    if not result['converged']:
        print("Error: Simulation did not converge. Cannot calculate Reynolds number.")
        return
    
    V_solution = result['solution']
    
    # Characteristic velocities
    V_max = np.max(V_solution)
    V_avg = np.mean(V_solution[V_solution > 0])
    V_inlet = V0_INITIAL
    
    # Characteristic lengths
    L_height = NY
    L_length = NX
    
    # Calculate Reynolds numbers
    Re_max = (RHO * V_max * L_height) / MU
    Re_avg = (RHO * V_avg * L_height) / MU
    Re_inlet = (RHO * V_inlet * L_height) / MU
    
    report = f"""
{'=' * 80}
REYNOLDS ANALYSIS REPORT
{'=' * 80}

1. PHYSICAL PARAMETERS
   - Density (rho):             {RHO} kg/m^3
   - Kinematic Viscosity (nu):  {NU} m^2/s
   - Dynamic Viscosity (mu):    {MU} Pa*s

2. GEOMETRY
   - Channel Height (L):        {L_height} units
   - Channel Length:            {L_length} units
   - Mesh Resolution:           {NY} x {NX}

3. CHARACTERISTIC VELOCITIES
   - Inlet Velocity:            {V_inlet:.4f} m/s
   - Max Velocity:              {V_max:.4f} m/s
   - Average Velocity:          {V_avg:.4f} m/s
   - Vertical Component (Vy):   {VY_TEST:.4f} m/s

4. CALCULATED REYNOLDS NUMBERS
   
   Re (based on V_inlet) = {Re_inlet:.2f}
   Re (based on V_max)   = {Re_max:.2f}
   Re (based on V_avg)   = {Re_avg:.2f}

5. PHYSICAL INTERPRETATION

   Flow Regime:
   """
    
    Re_ref = Re_inlet
    
    if Re_ref < 2000:
        regime = "LAMINAR"
        description = """
   Re < 2000 -> LAMINAR FLOW
   
   The flow is ordered and predictable. Fluid layers slide smoothly 
   over one another without mixing. This regime justifies:
   
   - The use of iterative methods.
   - The relatively fast convergence of solvers.
   - The numerical stability observed.
   
   NOTE: In this regime, non-linear Navier-Stokes terms have a small 
   but non-negligible contribution, making Newton-Raphson appropriate.
        """
    elif Re_ref < 4000:
        regime = "TRANSITION"
        description = """
   2000 < Re < 4000 -> TRANSITION FLOW
   
   The flow is in an intermediate zone between laminar and turbulent.
   Small perturbations may grow or decay.
   
   - Higher sensitivity to boundary conditions.
   - Possible appearance of local instabilities.
   - Requires careful spatial discretization.
        """
    else:
        regime = "TURBULENT"
        description = """
   Re > 4000 -> TURBULENT FLOW
   
   The flow is chaotic with multi-scale eddies.
   
   WARNING: Current simulation does NOT include turbulence models 
   (like k-epsilon or LES). Results should be interpreted with caution.
   
   Recommendations:
   - Use appropriate turbulence models.
   - Significantly increase mesh resolution.
   - Consider transient simulations.
        """
    
    report += description
    
    report += f"""

6. NUMERICAL APPROACH VALIDATION

   Based on Re = {Re_ref:.2f}:
   
   """
    
    if Re_ref < 1:
        report += """   Re << 1: Non-linear terms are negligible (Stokes flow).
     A direct linear solver could be used.
   """
    elif Re_ref < 100:
        report += """   Re < 100: Non-linear terms are small but present.
     Newton-Raphson is appropriate and efficient.
   """
    else:
        report += """   Re > 100: Non-linear terms are significant.
     Newton-Raphson is NECESSARY.
     Linearization would lose precision.
   """
    
    report += f"""

7. CONCLUSIONS

   - Flow Regime: {regime}
   - Reference Reynolds: {Re_ref:.2f}
   - Numerical Approach: {'Appropriate' if Re_ref < 4000 else 'Needs improvement'}
   - Convergence: {'Yes' if result['converged'] else 'No'}
   - Iterations: {result['iterations']}

{'=' * 80}
"""
    
    print(report)
    
    report_path = os.path.join('analisis_avanzado', 'reporte_reynolds.txt')
    if not os.path.exists('analisis_avanzado'):
        os.makedirs('analisis_avanzado')
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}\n")
    
    return {
        'Re_inlet': Re_inlet,
        'Re_max': Re_max,
        'Re_avg': Re_avg,
        'regime': regime
    }

if __name__ == '__main__':
    calculate_reynolds()
