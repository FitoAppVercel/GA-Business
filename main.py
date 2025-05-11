"""
Script principal para ejecutar el algoritmo genético que genera y evalúa ideas de negocio.
"""

import argparse
import random
import os
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt

from models import Idea
# Importamos las funciones dummy pero las comentamos para referencia
from utils import (generate_random_idea, simulate_llm_evaluation, 
                  simulate_llm_mutation, simulate_llm_crossover)
# Importamos las funciones LLM reales
from llm import (generate_idea_with_llm, evaluate_fitness_with_llm,
                mutate_idea_with_llm, crossover_ideas_with_llm)
from genetic import run_genetic_algorithm, initialize_population, evaluate_population, evolve

# Crear la carpeta outputs si no existe
os.makedirs('outputs', exist_ok=True)

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

def print_population_stats(population: List[Idea], generation: int = None) -> Tuple[float, float]:
    """
    Imprime estadísticas de la población actual.
    
    Args:
        population (List[Idea]): Población de ideas.
        generation (int, optional): Número de generación. Por defecto, None.
        
    Returns:
        Tuple[float, float]: Mejor fitness y fitness promedio de la población.
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
        
    return best_fitness, avg_fitness

def save_fitness_log(best_fitness_history: List[float], avg_fitness_history: List[float]) -> None:
    """
    Guarda el historial de fitness en un archivo CSV.
    
    Args:
        best_fitness_history (List[float]): Lista con el mejor fitness de cada generación.
        avg_fitness_history (List[float]): Lista con el fitness promedio de cada generación.
    """
    csv_path = 'outputs/fitness_log.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Escribir encabezado
        writer.writerow(['Generación', 'Mejor Fitness', 'Fitness Promedio'])
        # Escribir datos por generación
        for gen, (best, avg) in enumerate(zip(best_fitness_history, avg_fitness_history)):
            writer.writerow([gen, f"{best:.6f}", f"{avg:.6f}"])
    
    print(f"Historial de fitness guardado en {csv_path}")

def plot_evolution(best_fitness_history: List[float], avg_fitness_history: List[float], args) -> None:
    """
    Genera un gráfico de la evolución del fitness a lo largo de las generaciones.
    
    Args:
        best_fitness_history (List[float]): Lista con el mejor fitness de cada generación.
        avg_fitness_history (List[float]): Lista con el fitness promedio de cada generación.
        args (argparse.Namespace): Argumentos de configuración.
    """
    generations = list(range(len(avg_fitness_history)))
    
    plt.figure(figsize=(10, 6))
    
    # Graficar ambas curvas: mejor fitness y fitness promedio
    plt.plot(generations, best_fitness_history, marker='o', color='green', label='Mejor Fitness')
    plt.plot(generations, avg_fitness_history, marker='s', color='blue', label='Fitness Promedio')
    
    plt.title('Evolución del Fitness por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    
    # Añadimos información de configuración
    config_info = (
        f"Población: {args.population}, Generaciones: {args.generations}\n"
        f"Elite: {args.elite}, Cruce: {args.crossover_rate}, "
        f"Mutación: {args.mutation_rate}, Intensidad: {args.mutation_strength}"
    )
    plt.figtext(0.5, 0.01, config_info, ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar en la carpeta outputs
    output_path = 'outputs/fitness_evolution.png'
    plt.savefig(output_path)
    print(f"Gráfico de evolución guardado en {output_path}")
    
    plt.show()

def get_population_stats(population: List[Idea]) -> Tuple[float, float]:
    """
    Calcula estadísticas de la población.
    
    Args:
        population (List[Idea]): Población de ideas.
        
    Returns:
        Tuple[float, float]: Mejor fitness y fitness promedio de la población.
    """
    # Ordenar por fitness (de mayor a menor)
    sorted_population = sorted(population, key=lambda idea: idea.fitness, reverse=True)
    
    # Calcular estadísticas
    avg_fitness = sum(idea.fitness for idea in population) / len(population)
    best_fitness = sorted_population[0].fitness
    
    return best_fitness, avg_fitness

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
    
    print("\n🧠 EJECUTANDO EN MODO LLM CON MISTRAL VÍA OLLAMA 🧠\n")
    
    # Imprimir configuración
    print("=== Configuración del Algoritmo Genético ===")
    print(f"Tamaño de población: {args.population}")
    print(f"Número de generaciones: {args.generations}")
    print(f"Tamaño de elite: {args.elite}")
    print(f"Tasa de cruce: {args.crossover_rate}")
    print(f"Tasa de mutación: {args.mutation_rate}")
    print(f"Intensidad de mutación: {args.mutation_strength}")
    
    # Ejecutar el algoritmo genético usando funciones LLM reales
    final_population, avg_fitness_history = run_genetic_algorithm(
        initializer_func=generate_idea_with_llm,  # Reemplaza generate_random_idea
        evaluator_func=evaluate_fitness_with_llm,  # Reemplaza simulate_llm_evaluation
        crossover_func=crossover_ideas_with_llm,  # Reemplaza simulate_llm_crossover
        mutation_func=mutate_idea_with_llm,  # Reemplaza simulate_llm_mutation
        population_size=args.population,
        num_generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_size=args.elite
    )
    
    # Calcular el mejor fitness para cada generación
    best_fitness_history = []
    
    # Recrear la evolución para calcular el mejor fitness
    # Usamos las funciones LLM reales
    population = initialize_population(args.population, generate_idea_with_llm)
    evaluate_population(population, evaluate_fitness_with_llm)
    
    # Obtener stats de la población inicial
    best_fitness, _ = get_population_stats(population)
    best_fitness_history.append(best_fitness)
    
    # Reconstruir el historial del mejor fitness
    for i in range(args.generations):
        # Evolucionamos para obtener la población en cada generación usando funciones LLM
        population = evolve(
            population,
            evaluate_fitness_with_llm,
            crossover_ideas_with_llm,
            mutate_idea_with_llm,
            args.crossover_rate,
            args.mutation_rate,
            args.mutation_strength,
            args.elite
        )
        
        # Calcular y almacenar el mejor fitness de esta generación
        best_fitness, _ = get_population_stats(population)
        best_fitness_history.append(best_fitness)
    
    # Imprimir estadísticas de la población final
    best_fitness, avg_fitness = print_population_stats(final_population, args.generations)
    
    # Guardar el historial de fitness en CSV
    save_fitness_log(best_fitness_history, avg_fitness_history)
    
    # Mostrar gráfico de evolución
    if not args.no_plot:
        try:
            plot_evolution(best_fitness_history, avg_fitness_history, args)
        except ImportError:
            print("Advertencia: Matplotlib no está instalado. No se puede generar el gráfico.")
            print("Instale matplotlib usando: pip3 install matplotlib")
    
    print("\n=== Proceso completado ===")
    print(f"Se generaron {args.generations} generaciones con {args.population} ideas por generación.")
    print(f"CSV guardado en: outputs/fitness_log.csv")
    print(f"Gráfico guardado en: outputs/fitness_evolution.png")
    print("\n🧠 El algoritmo genético se ejecutó con el modelo Mistral vía Ollama 🧠")

if __name__ == "__main__":
    main()