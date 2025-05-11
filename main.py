"""
Script principal para ejecutar el algoritmo genético que genera y evalúa ideas de negocio.
"""

import argparse
import random
from typing import List
import matplotlib.pyplot as plt

from models import Idea
from utils import (generate_random_idea, simulate_llm_evaluation, 
                  simulate_llm_mutation, simulate_llm_crossover)
from genetic import run_genetic_algorithm

def parse_arguments():
    """
    Analiza los argumentos de línea de comandos para configurar la ejecución.
    
    Returns:
        argparse.Namespace: Objeto con los argumentos analizados.
    """
    parser = argparse.ArgumentParser(description='Algoritmo Genético para Generación de Ideas de Negocio')
    
    parser.add_argument('--population', type=int, default=20,
                        help='Tamaño de la población (default: 20)')
    parser.add_argument('--generations', type=int, default=10,
                        help='Número de generaciones (default: 10)')
    parser.add_argument('--elite', type=int, default=2,
                        help='Número de individuos elite a preservar (default: 2)')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='Tasa de cruce (default: 0.8)')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Tasa de mutación (default: 0.2)')
    parser.add_argument('--mutation-strength', type=float, default=0.5,
                        help='Intensidad de la mutación (default: 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla para generación de números aleatorios (default: None)')
    parser.add_argument('--no-plot', action='store_true',
                        help='No mostrar gráfico de evolución')
    
    return parser.parse_args()

def print_population_stats(population: List[Idea], generation: int = None):
    """
    Imprime estadísticas de la población actual.
    
    Args:
        population (List[Idea]): Población de ideas.
        generation (int, optional): Número de generación. Por defecto, None.
    """
    # Ordenar por fitness (de mayor a menor)
    sorted_population = sorted(population, key=lambda idea: idea.fitness, reverse=True)
    
    # Calcular estadísticas
    avg_fitness = sum(idea.fitness for idea in population) / len(population)
    best_fitness = sorted_population[0].fitness
    worst_fitness = sorted_population[-1].fitness
    
    # Imprimir encabezado
    if generation is not None:
        print(f"\n=== Generación {generation} ===")
    else:
        print("\n=== Población Inicial ===")
    
    print(f"Fitness promedio: {avg_fitness:.4f}")
    print(f"Mejor fitness: {best_fitness:.4f}")
    print(f"Peor fitness: {worst_fitness:.4f}")
    
    # Imprimir las mejores 3 ideas
    print("\nMejores ideas:")
    for i, idea in enumerate(sorted_population[:3], 1):
        print(f"\n{i}. Fitness: {idea.fitness:.4f}")
        print(f"   Problema: {idea.problem}")
        print(f"   Solución: {idea.solution}")

def plot_evolution(avg_fitness_history, args):
    """
    Genera un gráfico de la evolución del fitness promedio a lo largo de las generaciones.
    
    Args:
        avg_fitness_history (List[float]): Lista de fitness promedio por generación.
        args (argparse.Namespace): Argumentos de configuración.
    """
    generations = list(range(len(avg_fitness_history)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness_history, marker='o')
    plt.title('Evolución del Fitness Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.grid(True)
    
    # Añadimos información de configuración
    config_info = (
        f"Población: {args.population}, Generaciones: {args.generations}\n"
        f"Elite: {args.elite}, Cruce: {args.crossover_rate}, "
        f"Mutación: {args.mutation_rate}, Intensidad: {args.mutation_strength}"
    )
    plt.figtext(0.5, 0.01, config_info, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fitness_evolution.png')
    plt.show()

def main():
    """
    Función principal que ejecuta el algoritmo genético completo.
    """
    # Analizar argumentos de línea de comandos
    args = parse_arguments()
    
    # Establecer semilla si se proporciona
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Semilla aleatoria establecida: {args.seed}")
    
    # Imprimir configuración
    print("=== Configuración del Algoritmo Genético ===")
    print(f"Tamaño de población: {args.population}")
    print(f"Número de generaciones: {args.generations}")
    print(f"Tamaño de elite: {args.elite}")
    print(f"Tasa de cruce: {args.crossover_rate}")
    print(f"Tasa de mutación: {args.mutation_rate}")
    print(f"Intensidad de mutación: {args.mutation_strength}")
    
    # Ejecutar el algoritmo genético
    final_population, avg_fitness_history = run_genetic_algorithm(
        initializer_func=generate_random_idea,
        evaluator_func=simulate_llm_evaluation,
        crossover_func=simulate_llm_crossover,
        mutation_func=simulate_llm_mutation,
        population_size=args.population,
        num_generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_size=args.elite
    )
    
    # Imprimir estadísticas de la población final
    print_population_stats(final_population, args.generations)
    
    # Mostrar gráfico de evolución
    if not args.no_plot:
        try:
            plot_evolution(avg_fitness_history, args)
        except ImportError:
            print("Advertencia: Matplotlib no está instalado. No se puede generar el gráfico.")
            print("Instale matplotlib usando: pip install matplotlib")
    
    print("\n=== Proceso completado ===")
    print(f"Se generaron {args.generations} generaciones con {args.population} ideas por generación.")
    print("Mejores ideas guardadas en la población final.")

if __name__ == "__main__":
    main()