#!/usr/bin/env python3
"""
Main Entry Point - CFD Simulation Project
==========================================
Script principal para ejecutar las simulaciones y visualizaciones del proyecto.
"""

import sys
import os

def print_menu():
    """Muestra el menu principal."""
    print("\n" + "=" * 80)
    print(" " * 20 + "PROYECTO FINAL - SIMULACION CFD")
    print("=" * 80)
    print("\nOpciones disponibles:")
    print("  1. Ejecutar simulaciones y generar graficas (V4 y V4.1)")
    print("  2. Generar animaciones profesionales")
    print("  3. Visualizar streamlines")
    print("  4. Ejecutar todo el pipeline completo")
    print("  0. Salir")
    print("=" * 80)

def run_simulations():
    """Ejecuta las simulaciones de campo de velocidades."""
    print("\n" + "=" * 80)
    print("EJECUTANDO SIMULACIONES")
    print("=" * 80)
    
    print("\n[1/2] Ejecutando campo_velocidadesV4.py...")
    print("-" * 80)
    os.system("python modulos/campo_velocidadesV4.py")
    
    print("\n[2/2] Ejecutando campo_velocidadesV4_1.py...")
    print("-" * 80)
    os.system("python modulos/campo_velocidadesV4_1.py")
    
    print("\n" + "=" * 80)
    print("SIMULACIONES COMPLETADAS")
    print("=" * 80)

def run_animations():
    """Genera las animaciones profesionales."""
    print("\n" + "=" * 80)
    print("GENERANDO ANIMACIONES")
    print("=" * 80)
    os.system("python modulos/crear_animaciones_pro.py")

def run_streamlines():
    """Genera las visualizaciones de streamlines."""
    print("\n" + "=" * 80)
    print("GENERANDO STREAMLINES")
    print("=" * 80)
    os.system("python modulos/visualizar_streamlines.py")

def run_all():
    """Ejecuta todo el pipeline completo."""
    print("\n" + "=" * 80)
    print("EJECUTANDO PIPELINE COMPLETO")
    print("=" * 80)
    
    run_simulations()
    run_animations()
    run_streamlines()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETO FINALIZADO")
    print("=" * 80)
    print("\nResultados generados:")
    print("  - Graficas: graficas_V4/ y graficas_V4_1/")
    print("  - Animaciones: animaciones_pro/")
    print("  - Streamlines: analisis_avanzado/")
    print()

def main():
    """Funcion principal."""
    while True:
        print_menu()
        
        try:
            opcion = input("\nSeleccione una opcion: ").strip()
            
            if opcion == "0":
                print("\nSaliendo del programa...")
                print("=" * 80)
                sys.exit(0)
            
            elif opcion == "1":
                run_simulations()
            
            elif opcion == "2":
                run_animations()
            
            elif opcion == "3":
                run_streamlines()
            
            elif opcion == "4":
                run_all()
            
            else:
                print("\nOpcion no valida. Por favor seleccione una opcion del menu.")
        
        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario.")
            print("Saliendo del programa...")
            print("=" * 80)
            sys.exit(0)
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Por favor intente nuevamente.")
        
        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()
